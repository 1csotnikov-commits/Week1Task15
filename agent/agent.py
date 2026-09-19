"""Agent — объектная сущность, инкапсулирующая LLM и слои памяти."""
from __future__ import annotations

import time
from typing import Any, Callable, Optional

from .config import AgentConfig
from .errors import LLMError, MemoryError, TaskError
from .fsm import Fsm, Stage, Transition, format_disallowed, normalize_fsm_data
from .invariants import InvariantStore
from .memory import MemoryStore, new_id, now_iso
from .profiles import ProfileStore
from .providers.base import LLMProvider
from .tasks import TaskManager

LAYER_LABELS = {
    "invariant": "ИНВАРИАНТЫ (INVARIANTS)",
    "profile": "ПРОФИЛЬ (PROFILE)",
    "long": "ДОЛГОВРЕМЕННАЯ ПАМЯТЬ (LONG-TERM)",
    "working": "РАБОЧАЯ ПАМЯТЬ (WORKING)",
    "short": "КРАТКОСРОЧНАЯ ПАМЯТЬ (SHORT-TERM)",
}

LONG_TYPES = ("decision", "knowledge")


class Agent:
    """Агент с явной многослойной моделью памяти.

    Каждый слой хранится отдельно; пользователь явно выбирает слой при
    сохранении. Все включённые слои подмешиваются в промт в порядке:
    system -> profile -> long-term -> working -> short-term -> текущий запрос.
    """

    def __init__(
        self,
        config: AgentConfig,
        provider: Optional[LLMProvider] = None,
        provider_factory: Optional[Callable[[], LLMProvider]] = None,
    ) -> None:
        self.config = config
        self.name = config.name
        self._provider = provider
        self._provider_factory = provider_factory

        self.store = MemoryStore(config.memory_dir)
        self.profiles = ProfileStore(config.memory_dir)
        self.invariants = InvariantStore(config.memory_dir)
        self.tasks = TaskManager(config.name, self.store)

        # Какие слои подмешиваются в промт. По умолчанию — все.
        self.usage = {
            "invariant": True,
            "profile": True,
            "long": True,
            "working": True,
            "short": True,
        }
        self.debug_enabled = False
        self.metrics_enabled = False
        self.prompt_save = False  # CLI-вариант A: спрашивать «сохранить куда?»
        self.invariant_check = False  # пост-проверка LLM-валидатором (по умолчанию off)

        self.last_debug: Optional[list[dict]] = None
        self.last_metrics: Optional[dict] = None
        self.last_invariant_check: Optional[dict] = None
        self.last_transition_check: Optional[dict] = None

        # Восстановление краткосрочной памяти и активной задачи из файла.
        self.short_term = self.store.load_short()
        saved_task = self.short_term.get("task")
        if saved_task and self.tasks.exists(saved_task):
            self.tasks.active_task = saved_task
        else:
            self.tasks.active_task = None
            self.short_term["task"] = None

    # ------------------------------------------------------------------ провайдер
    @property
    def provider(self) -> LLMProvider:
        if self._provider is None:
            if self._provider_factory is None:
                raise LLMError("У агента не задан LLM-провайдер.")
            self._provider = self._provider_factory()
        return self._provider

    # ------------------------------------------------------------- основной метод
    def ask(self, prompt: str) -> str:
        """Обработать запрос пользователя и вернуть ответ модели."""
        started = time.perf_counter()

        # 1. Сообщение пользователя в short-term.
        self.short_term["task"] = self.tasks.active_task
        self.short_term.setdefault("messages", [])
        self.short_term["messages"].append(self.store.make_message("user", prompt))

        # 2. Формирование промта с учётом включённых слоёв.
        messages, debug_sections = self._build_prompt(prompt)
        self.last_debug = debug_sections

        # 3. Запрос к LLM.
        result = self.provider.chat(messages)
        content = result.get("content") or ""

        # 4. Пост-проверка инвариантов (LLM-валидатор). Исходный ответ при
        #    нарушении пользователю не показывается — только текст отказа.
        self.last_invariant_check = None
        reply = content
        if (
            self.invariant_check
            and self.usage["invariant"]
            and self._invariants_list()
        ):
            check = self._validate_invariants(content)
            self.last_invariant_check = check
            if check.get("status") == "VIOLATION":
                reply = (
                    "Предложенное решение нарушает инвариант "
                    f"{check.get('invariant_id', '?')}."
                )

        # 5. Ответ в short-term.
        self.short_term["messages"].append(
            self.store.make_message("assistant", reply)
        )

        # 6. Сохранение short-term на диск (только при активной задаче).
        if self.tasks.active_task:
            self.store.save_short(self.short_term)

        # Метрики.
        self.last_metrics = self._make_metrics(result, time.perf_counter() - started)
        return reply

    # ------------------------------------------------------------------- промт
    def _build_prompt(self, current_request: str) -> tuple[list[dict], list[dict]]:
        sections: list[tuple[str, str]] = []
        debug_sections: list[dict] = []

        if self.usage["invariant"]:
            invariants = self._invariants_list()
            if invariants:
                sections.append(("invariant", self._format_invariants(invariants)))
                debug_sections.append(
                    {
                        "layer": "invariant",
                        "invariants": [
                            {"id": inv.get("id"), "text": inv.get("text")}
                            for inv in invariants
                        ],
                    }
                )

        if self.usage["profile"]:
            active_profile = self.profiles.active_name()
            if active_profile:
                try:
                    profile = self.profiles.load(active_profile)
                except MemoryError:
                    profile = None
                if profile:
                    sections.append(("profile", self._format_profile(profile)))
                    debug_sections.append(
                        {
                            "layer": "profile",
                            "name": profile.get("name"),
                            "fields": {
                                "style": profile.get("style", ""),
                                "format": profile.get("format", ""),
                                "constraints": profile.get("constraints", ""),
                            },
                            "notes": list(profile.get("notes", [])),
                        }
                    )

        if self.usage["long"]:
            records = self.store.load_long().get("records", [])
            if records:
                sections.append(("long", self._format_long(records)))
                debug_sections.append({"layer": "long", "records": list(records)})

        if self.usage["working"] and self.tasks.active_task:
            data = self.store.load_working(self.tasks.active_task)
            if data and data.get("records"):
                sections.append(("working", self._format_working(data)))
                debug_sections.append(
                    {
                        "layer": "working",
                        "task": data.get("task"),
                        "records": list(data["records"]),
                    }
                )

        state = self._build_state_block()
        if state:
            sections.append(("fsm", state[0]))
            debug_sections.append(state[1])

        if self.usage["short"]:
            messages = self.short_term.get("messages", [])
            if messages:
                sections.append(("short", self._format_short(messages)))
                debug_sections.append(
                    {
                        "layer": "short",
                        "task": self.short_term.get("task"),
                        "messages": list(messages),
                    }
                )

        blocks = []
        for layer, text in sections:
            if layer == "fsm":
                blocks.append(text)
            else:
                blocks.append(f"=== {LAYER_LABELS[layer]} ===\n{text}")
        blocks.append(f"=== ТЕКУЩИЙ ЗАПРОС ===\n{current_request}")
        user_content = "\n\n".join(blocks)

        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": user_content},
        ]
        return messages, debug_sections

    @staticmethod
    def _format_profile(profile: dict) -> str:
        lines = [f"Имя профиля: {profile.get('name')}"]
        lines.append(f"Стиль: {profile.get('style', '')}")
        lines.append(f"Формат: {profile.get('format', '')}")
        lines.append(f"Ограничения: {profile.get('constraints', '')}")
        notes = profile.get("notes", [])
        if notes:
            lines.append("Заметки:")
            lines.extend(f"- {n}" for n in notes)
        return "\n".join(lines)

    @staticmethod
    def _format_long(records: list[dict]) -> str:
        return "\n".join(f"[{r.get('type')}] {r.get('content')}" for r in records)

    @staticmethod
    def _format_working(data: dict) -> str:
        lines = [f"Задача: {data.get('task')}"]
        lines.extend(f"- {r.get('content')}" for r in data.get("records", []))
        return "\n".join(lines)

    @staticmethod
    def _format_short(messages: list[dict]) -> str:
        lines = []
        for m in messages:
            role = "Пользователь" if m.get("role") == "user" else "Ассистент"
            lines.append(f"{role}: {m.get('content')}")
        return "\n".join(lines)

    # ----------------------------------------------------------------- инварианты
    def _invariants_list(self) -> list[dict]:
        return self.invariants.list()

    @staticmethod
    def _format_invariants(invariants: list[dict]) -> str:
        lines = []
        for inv in invariants:
            lines.append(f"[{inv.get('category', 'other')}] {inv.get('text', '')}")
        return "\n".join(lines)

    def _validate_invariants(self, answer: str) -> dict:
        """Пост-проверка ответа LLM-валидатором.

        Возвращает {"status": "OK"} либо {"status": "VIOLATION", "invariant_id": ...}.
        """
        invariants = self._invariants_list()
        listing = "\n".join(
            f"{inv.get('id')}: {inv.get('text', '')}" for inv in invariants
        )
        prompt = (
            f"Вот список инвариантов:\n{listing}\n\n"
            f"Вот ответ ассистента:\n{answer}\n\n"
            "Нарушает ли этот ответ хотя бы один инвариант? Ответь строго: "
            '"OK" или "VIOLATION: <id инварианта>".'
        )
        result = self.provider.chat(
            [
                {
                    "role": "system",
                    "content": (
                        'Ты проверяешь ответ ассистента на нарушение инвариантов. '
                        'Отвечай строго "OK" или "VIOLATION: <id инварианта>".'
                    ),
                },
                {"role": "user", "content": prompt},
            ]
        )
        verdict = (result.get("content") or "").strip()
        if verdict.upper().startswith("OK"):
            return {"status": "OK"}
        if verdict.upper().startswith("VIOLATION"):
            invariant_id = ""
            if ":" in verdict:
                invariant_id = verdict.split(":", 1)[1].strip()
            return {"status": "VIOLATION", "invariant_id": invariant_id}
        # Не удалось однозначно распознать — считаем, что нарушения нет.
        return {"status": "OK"}

    @staticmethod
    def _make_metrics(result: dict, elapsed: float) -> dict:
        usage = result.get("usage") or {}
        return {
            "elapsed": round(elapsed, 3),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "model": result.get("model"),
        }

    def _build_state_block(self) -> Optional[tuple[str, dict]]:
        """Блок состояния задачи (FSM) для промта, если он применим.

        Блок добавляется только если у активной задачи есть этапы, задача не на
        паузе и не завершена. Возвращает (текст, отладочную секцию) или None.
        """
        if not self.tasks.active_task:
            return None
        fsm = self.fsm()
        if fsm is None or fsm.is_empty() or fsm.paused or fsm.completed:
            return None
        stage = fsm.active_stage()
        if stage is None:
            return None
        outgoing = fsm.outgoing_transitions(stage.name)
        lines = [
            "[СОСТОЯНИЕ ЗАДАЧИ]",
            f"Текущий этап: {stage.name}",
            f"Описание этапа: {stage.description}",
            f"Ожидаемое действие: {stage.expected_action}",
            "Допустимые переходы:",
        ]
        if outgoing:
            for t in outgoing:
                lines.append(f"  → {t.to}  (условие: {t.condition or '—'})")
        else:
            lines.append("  (нет — задача завершается)")
        lines.append(
            "\nРаботай в рамках текущего этапа. Не переходи к следующему, "
            "пока не выполнено условие соответствующего перехода."
        )
        text = "\n".join(lines)
        return text, {"layer": "fsm", "text": text}

    # ------------------------------------------------------------------- задачи
    def task_new(self, name: str) -> str:
        self.tasks.create_task(name)
        self._reset_short_for_task(name)
        return name

    def task_switch(self, name: str) -> str:
        self.tasks.switch_task(name)
        self._reset_short_for_task(name)
        return name

    def _reset_short_for_task(self, name: str) -> None:
        self.short_term = {"task": name, "messages": []}
        self.store.save_short(self.short_term)

    def task_list(self) -> list[str]:
        return self.tasks.list_tasks()

    def task_current(self) -> Optional[str]:
        return self.tasks.current_task()

    # --------------------------------------------------------------------- FSM
    def fsm(self) -> Optional[Fsm]:
        """Вернуть состояние конечного автомата активной задачи (или None).

        При первом чтении старой схемы (условие в этапе, нет transitions)
        выполняется миграция и результат сохраняется обратно.
        """
        data = self._working_data()
        if not data:
            return None
        return self._load_fsm(data)

    def _load_fsm(self, data: dict) -> Optional[Fsm]:
        raw = data.get("fsm")
        if not raw:
            return None
        normalized, changed = normalize_fsm_data(raw)
        if changed:
            data["fsm"] = normalized
            self._save_working(data)
        return Fsm.from_dict(normalized)

    def fsm_info(self) -> Optional[dict]:
        fsm = self.fsm()
        if fsm is None:
            return None
        data = fsm.to_dict()
        data["stages"] = sorted(data["stages"], key=lambda s: s["order"])
        return data

    def fsm_current_stage(self) -> Optional[dict]:
        fsm = self.fsm()
        if fsm is None:
            return None
        stage = fsm.active_stage()
        return stage.to_dict() if stage else None

    def fsm_current_stage_name(self) -> Optional[str]:
        fsm = self.fsm()
        return fsm.current_stage if fsm else None

    def _working_data(self) -> Optional[dict]:
        if not self.tasks.active_task:
            return None
        return self.store.load_working(self.tasks.active_task)

    def _require_working(self) -> dict:
        if not self.tasks.active_task:
            raise TaskError("Нет активной задачи.")
        data = self.store.load_working(self.tasks.active_task)
        if data is None:
            data = self.store.create_working(self.tasks.active_task)
        return data

    def _save_working(self, data: dict) -> None:
        self.store.save_working(self.tasks.active_task, data)

    def _fsm_from(self, data: dict) -> Fsm:
        fsm = self._load_fsm(data)
        if fsm is None or fsm.is_empty():
            raise TaskError("У задачи нет этапов. Задайте их через /task stages set.")
        return fsm

    def _fsm_require_active(self, fsm: Fsm) -> Stage:
        if fsm.completed:
            raise TaskError("Задача уже завершена.")
        stage = fsm.active_stage()
        if stage is None:
            raise TaskError("Нет активного этапа.")
        return stage

    def _save_fsm_state(self, data: dict, fsm: Fsm) -> bool:
        """Сохранить FSM; если текущий этап терминален — завершить задачу.

        Возвращает True, если задача была помечена завершённой этим вызовом.
        """
        completed = False
        if (
            not fsm.completed
            and not fsm.paused
            and fsm.current_stage
            and not fsm.outgoing_transitions(fsm.current_stage)
        ):
            fsm.completed = True
            data["records"] = []
            completed = True
        data["fsm"] = fsm.to_dict()
        self._save_working(data)
        return completed

    def fsm_stages_set(self, stages: list[dict]) -> list[str]:
        """Полностью перезаписать набор этапов.

        Очищает рабочую память, все статусы — pending, первый этап — active.
        Автоматически генерирует линейный граф переходов (условия пустые).
        Возвращает список имён этапов в порядке их order.
        """
        data = self._require_working()
        built: list[Stage] = []
        for i, raw in enumerate(stages):
            name = str(raw.get("name") or "").strip()
            if not name:
                raise TaskError("Имя этапа не может быть пустым.")
            built.append(
                Stage(
                    name=name,
                    description=str(raw.get("description") or "").strip(),
                    expected_action=str(raw.get("expected_action") or "").strip(),
                    status="active" if i == 0 else "pending",
                    order=i,
                )
            )
        fsm = Fsm(stages=built, current_stage=built[0].name if built else None)
        fsm.transitions = fsm.build_linear_transitions()
        data["records"] = []
        data["fsm"] = fsm.to_dict()
        self._save_working(data)
        return [s.name for s in built]

    def fsm_allowed_transitions(self) -> list[dict]:
        """Исходящие переходы из текущего этапа (как словари)."""
        data = self._require_working()
        fsm = self._fsm_from(data)
        self._fsm_require_active(fsm)
        current = fsm.active_stage()
        return [t.to_dict() for t in fsm.outgoing_transitions(current.name)]

    def fsm_move(self, to_name: str) -> str:
        """Перейти в этап ``to_name`` по разрешённому переходу."""
        data = self._require_working()
        fsm = self._fsm_from(data)
        current = self._fsm_require_active(fsm)
        transition = fsm.transition(current.name, to_name)
        if transition is None:
            allowed = [t.to for t in fsm.outgoing_transitions(current.name)]
            raise TaskError(format_disallowed(current.name, to_name, allowed))
        return self._apply_transition(fsm, transition, data)

    def fsm_back(self) -> str:
        """Вернуться на предыдущий этап — только при явном обратном переходе."""
        data = self._require_working()
        fsm = self._fsm_from(data)
        current = self._fsm_require_active(fsm)
        prev = fsm.prev_stage()
        if prev is None:
            raise TaskError("Это первый этап — назад переходить некуда.")
        transition = fsm.transition(current.name, prev.name)
        if transition is None:
            allowed = [t.to for t in fsm.outgoing_transitions(current.name)]
            raise TaskError(format_disallowed(current.name, prev.name, allowed))
        return self._apply_transition(fsm, transition, data)

    def _apply_transition(self, fsm: Fsm, transition: Transition, data: dict) -> str:
        """Выполнить разрешённый переход и сохранить состояние."""
        current = fsm.active_stage()
        target = fsm.stage_by_name(transition.to)
        if target is None:
            raise TaskError(f"Целевой этап «{transition.to}» не найден.")
        if current is not None:
            current.status = "completed" if target.order > current.order else "pending"
        target.status = "active"
        fsm.current_stage = target.name
        terminal = not fsm.outgoing_transitions(target.name)
        if terminal:
            fsm.completed = True
            data["records"] = []
        data["fsm"] = fsm.to_dict()
        self._save_working(data)
        if terminal:
            return (
                f"Переход к этапу «{target.name}». Этап не имеет исходящих переходов — "
                "задача завершена, рабочая память очищена."
            )
        return f"Переход к этапу «{target.name}»."

    # ------------------------------------------------------- управление переходами
    def fsm_transitions_set(self, transitions: list[dict]) -> list[dict]:
        """Полностью перезаписать граф переходов."""
        data = self._require_working()
        fsm = self._fsm_from(data)
        names = {s.name for s in fsm.stages}
        built: list[Transition] = []
        for raw in transitions:
            frm = str(raw.get("from") or "").strip()
            to = str(raw.get("to") or "").strip()
            condition = str(raw.get("condition") or "").strip()
            if not frm or not to:
                raise TaskError("Переход должен содержать from и to.")
            if frm not in names:
                raise TaskError(f"Этап «{frm}» не найден.")
            if to not in names:
                raise TaskError(f"Этап «{to}» не найден.")
            built.append(
                Transition(id=new_id(), from_stage=frm, to=to, condition=condition)
            )
        fsm.transitions = built
        self._save_fsm_state(data, fsm)
        return [t.to_dict() for t in built]

    def fsm_transition_add(self, from_stage: str, to: str, condition: str) -> dict:
        data = self._require_working()
        fsm = self._fsm_from(data)
        names = {s.name for s in fsm.stages}
        if from_stage not in names:
            raise TaskError(f"Этап «{from_stage}» не найден.")
        if to not in names:
            raise TaskError(f"Этап «{to}» не найден.")
        transition = Transition(
            id=new_id(), from_stage=from_stage, to=to, condition=condition
        )
        fsm.transitions.append(transition)
        self._save_fsm_state(data, fsm)
        return transition.to_dict()

    def fsm_transition_del(self, transition_id: str) -> bool:
        data = self._require_working()
        fsm = self._fsm_from(data)
        for i, t in enumerate(fsm.transitions):
            if t.id == transition_id:
                fsm.transitions.pop(i)
                self._save_fsm_state(data, fsm)
                return True
        return False

    def fsm_transition_edit(self, transition_id: str, condition: str) -> dict:
        data = self._require_working()
        fsm = self._fsm_from(data)
        t = fsm.transition_by_id(transition_id)
        if t is None:
            raise TaskError(f"Переход {transition_id} не найден.")
        t.condition = condition
        data["fsm"] = fsm.to_dict()
        self._save_working(data)
        return t.to_dict()

    def fsm_transitions_clear(self) -> str:
        data = self._require_working()
        fsm = self._fsm_from(data)
        fsm.transitions = []
        completed = self._save_fsm_state(data, fsm)
        if completed:
            return (
                "Все переходы очищены. Текущий этап не имеет исходящих переходов — "
                "задача завершена, рабочая память очищена."
            )
        return "Все переходы очищены."

    def fsm_pause(self) -> str:
        data = self._require_working()
        fsm = self._fsm_from(data)
        fsm.paused = True
        data["fsm"] = fsm.to_dict()
        self._save_working(data)
        return "Задача поставлена на паузу."

    def fsm_resume(self) -> str:
        data = self._require_working()
        fsm = self._fsm_from(data)
        fsm.paused = False
        data["fsm"] = fsm.to_dict()
        self._save_working(data)
        return "Задача снята с паузы."

    def check_transitions(self) -> Optional[dict]:
        """Проверить условия всех исходящих переходов текущего этапа через LLM.

        Возвращает словарь:
        - {"terminal": True, "message": ..., "checked": [], "triggered": []} —
          если текущий этап не имеет исходящих переходов (задача завершается);
        - {"terminal": False, "checked": [...], "triggered": [...]} — иначе.

        ``checked`` — переходы с непустым условием и результатом проверки
        (да/нет); ``triggered`` — переходы, чьё условие LLM счёл выполненным.
        None — если проверка неприменима (нет задачи/этапов/пауза/завершено/
        нет последнего обмена).
        """
        if not self.tasks.active_task:
            return None
        fsm = self.fsm()
        if fsm is None or fsm.is_empty() or fsm.paused or fsm.completed:
            return None
        outgoing = fsm.outgoing_transitions(fsm.current_stage)
        if not outgoing:
            data = self._require_working()
            fsm.completed = True
            data["records"] = []
            data["fsm"] = fsm.to_dict()
            self._save_working(data)
            result = {
                "terminal": True,
                "message": (
                    "Текущий этап не имеет исходящих переходов. "
                    "Задача завершена, рабочая память очищена."
                ),
                "checked": [],
                "triggered": [],
            }
            self.last_transition_check = result
            return result

        conditioned = [t for t in outgoing if t.condition]
        if not conditioned:
            result = {"terminal": False, "checked": [], "triggered": []}
            self.last_transition_check = result
            return result

        messages = self.short_term.get("messages", [])
        last_user = next(
            (m for m in reversed(messages) if m.get("role") == "user"), None
        )
        last_assistant = next(
            (m for m in reversed(messages) if m.get("role") == "assistant"), None
        )
        if last_user is None or last_assistant is None:
            return None

        checked: list[dict] = []
        triggered: list[dict] = []
        for t in conditioned:
            prompt = (
                f"Вот условие перехода: {t.condition}. "
                f"Вот последний обмен: {last_user.get('content', '')} / "
                f"{last_assistant.get('content', '')}. "
                'Ответь строго "да" или "нет".'
            )
            result = self.provider.chat(
                [
                    {
                        "role": "system",
                        "content": 'Ты проверяешь условие перехода. Отвечай строго "да" или "нет".',
                    },
                    {"role": "user", "content": prompt},
                ]
            )
            answer = (result.get("content") or "").strip().lower()
            satisfied = answer.startswith("да")
            checked.append(
                {
                    "id": t.id,
                    "from": t.from_stage,
                    "to": t.to,
                    "condition": t.condition,
                    "result": satisfied,
                }
            )
            if satisfied:
                triggered.append(t.to_dict())
        out = {"terminal": False, "checked": checked, "triggered": triggered}
        self.last_transition_check = out
        return out

    # ------------------------------------------------------------------- память
    def memory_show(self, layer: str) -> dict:
        if layer == "long":
            return self.store.load_long()
        if layer == "short":
            return self.short_term
        if layer == "working":
            if not self.tasks.active_task:
                raise MemoryError("Нет активной задачи — рабочая память недоступна.")
            return self.store.load_working(self.tasks.active_task) or {
                "task": self.tasks.active_task,
                "created_at": None,
                "records": [],
            }
        raise MemoryError(f"Неизвестный слой: {layer}")

    def memory_add(
        self,
        layer: str,
        content: str,
        record_type: Optional[str] = None,
        role: str = "user",
    ) -> dict:
        if layer == "long":
            if record_type not in LONG_TYPES:
                raise MemoryError("Для long укажите тип: decision | knowledge.")
            data = self.store.load_long()
            record = self.store.make_long_record(record_type, content)
            data["records"].append(record)
            self.store.save_long(data)
            return record

        if layer == "working":
            if not self.tasks.active_task:
                raise MemoryError("Нет активной задачи — добавить в рабочую память нельзя.")
            data = self.store.load_working(self.tasks.active_task) or self.store.create_working(
                self.tasks.active_task
            )
            record = self.store.make_working_record(content)
            data["records"].append(record)
            self.store.save_working(self.tasks.active_task, data)
            return record

        if layer == "short":
            record = self.store.make_message(role, content)
            self.short_term.setdefault("messages", []).append(record)
            if self.tasks.active_task:
                self.short_term["task"] = self.tasks.active_task
                self.store.save_short(self.short_term)
            return record

        raise MemoryError(f"Неизвестный слой: {layer}")

    def memory_del(self, layer: str, record_id: str) -> bool:
        if layer == "long":
            data = self.store.load_long()
            removed = self.store.remove_record(data["records"], record_id)
            if removed:
                self.store.save_long(data)
            return removed

        if layer == "working":
            if not self.tasks.active_task:
                raise MemoryError("Нет активной задачи.")
            data = self.store.load_working(self.tasks.active_task)
            if not data:
                return False
            removed = self.store.remove_record(data["records"], record_id)
            if removed:
                self.store.save_working(self.tasks.active_task, data)
            return removed

        if layer == "short":
            removed = self.store.remove_record(
                self.short_term.get("messages", []), record_id
            )
            if removed and self.tasks.active_task:
                self.store.save_short(self.short_term)
            return removed

        raise MemoryError(f"Неизвестный слой: {layer}")

    def memory_clear(self, layer: str) -> None:
        if layer == "long":
            self.store.save_long({"records": []})
            return

        if layer == "working":
            if not self.tasks.active_task:
                raise MemoryError("Нет активной задачи.")
            data = self.store.load_working(self.tasks.active_task) or {
                "task": self.tasks.active_task,
                "created_at": now_iso(),
                "records": [],
            }
            data["records"] = []
            self.store.save_working(self.tasks.active_task, data)
            return

        if layer == "short":
            self.short_term = {"task": self.tasks.active_task, "messages": []}
            if self.tasks.active_task:
                self.store.save_short(self.short_term)
            else:
                self.store.delete_short()
            return

        raise MemoryError(f"Неизвестный слой: {layer}")

    def memory_files(self) -> dict:
        active_profile = self.profiles.active_name()
        return {
            "invariants": str(self.invariants.invariants_path),
            "long": str(self.store.long_path),
            "short": str(self.store.short_path),
            "working": (
                str(self.store.working_path(self.tasks.active_task))
                if self.tasks.active_task
                else None
            ),
            "profiles_dir": str(self.profiles.profiles_dir),
            "active_profile": (
                str(self.profiles.active_profile_path) if active_profile else None
            ),
        }

    def memory_usage(self, layer: str, on_off: str) -> bool:
        if layer not in self.usage:
            raise MemoryError(f"Неизвестный слой: {layer}")
        if on_off not in ("on", "off"):
            raise MemoryError("Укажите on или off.")
        self.usage[layer] = on_off == "on"
        return self.usage[layer]

    # ----------------------------------------------------------------- инварианты
    def invariant_list(self) -> list[dict]:
        return self.invariants.list()

    def invariant_add(self, category: str, text: str) -> dict:
        return self.invariants.add(category, text)

    def invariant_del(self, invariant_id: str) -> bool:
        return self.invariants.delete(invariant_id)

    def invariant_edit(self, invariant_id: str, text: str) -> dict:
        return self.invariants.edit(invariant_id, text)

    def invariant_clear(self) -> None:
        self.invariants.clear()

    def invariant_check_set(self, on_off: str) -> bool:
        if on_off not in ("on", "off"):
            raise MemoryError("Укажите on или off.")
        self.invariant_check = on_off == "on"
        return self.invariant_check

    # ------------------------------------------------------------------ профили
    def profile_list(self) -> list[str]:
        return self.profiles.list_profiles()

    def profile_current(self) -> Optional[str]:
        return self.profiles.active_name()

    def profile_show(self, name: Optional[str] = None) -> dict:
        return self.profiles.load(name)

    def profile_new(self, name: str) -> dict:
        return self.profiles.create(name)

    def profile_switch(self, name: str) -> str:
        return self.profiles.switch(name)

    def profile_del(self, name: str) -> None:
        self.profiles.delete(name)

    def profile_set(self, name: str, field: str, value: str) -> dict:
        return self.profiles.set_field(name, field, value)

    def profile_unset(self, name: str, field: str) -> dict:
        return self.profiles.unset_field(name, field)

    def profile_add_note(self, name: str, text: str) -> dict:
        return self.profiles.add_note(name, text)

    def profile_del_note(self, name: str, index: int) -> dict:
        return self.profiles.del_note(name, index)



