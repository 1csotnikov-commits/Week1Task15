"""Конечный автомат (FSM) состояния задачи.

Модель описывает «этапы задачи» (они же «текущий шаг» / «ожидаемое действие» —
это один уровень). Внутри одного этапа отдельного «шага» нет.

Этап состоит из:
- name — короткое имя (используется в командах);
- description — что должно происходить на этапе (текст);
- expected_action — что ожидается от пользователя/агента на этом этапе (текст);
- status — pending | active | completed;
- order — порядковый номер.

Переход между этапами описывается отдельной сущностью Transition:
- id — UUID;
- from — имя этапа-источника;
- to — имя целевого этапа;
- condition — условие перехода на естественном языке (оценивает LLM);
  пустая строка означает, что переход не участвует в авто-проверке.

Состояние автомата хранится в поле ``fsm`` файла ``working_<task_name>.json``.
Если этапов нет — поле ``fsm`` отсутствует или равно null, и задача работает как
обычный диалог с рабочей памятью.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .memory import new_id

STAGE_STATUSES = ("pending", "active", "completed")


@dataclass
class Stage:
    """Один этап задачи."""

    name: str
    description: str = ""
    expected_action: str = ""
    status: str = "pending"
    order: int = 0

    @classmethod
    def from_dict(cls, data: dict) -> "Stage":
        return cls(
            name=str(data.get("name") or ""),
            description=str(data.get("description") or ""),
            expected_action=str(data.get("expected_action") or ""),
            status=str(data.get("status") or "pending"),
            order=int(data.get("order") or 0),
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "expected_action": self.expected_action,
            "status": self.status,
            "order": self.order,
        }


@dataclass
class Transition:
    """Один переход между этапами."""

    id: str = ""
    from_stage: str = ""
    to: str = ""
    condition: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "Transition":
        return cls(
            id=str(data.get("id") or ""),
            from_stage=str(data.get("from") or ""),
            to=str(data.get("to") or ""),
            condition=str(data.get("condition") or ""),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "from": self.from_stage,
            "to": self.to,
            "condition": self.condition,
        }


@dataclass
class Fsm:
    """Состояние конечного автомата задачи."""

    stages: list[Stage] = field(default_factory=list)
    transitions: list[Transition] = field(default_factory=list)
    current_stage: Optional[str] = None
    paused: bool = False
    completed: bool = False

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> Optional["Fsm"]:
        if not data:
            return None
        stages = [Stage.from_dict(s) for s in data.get("stages", [])]
        transitions = [Transition.from_dict(t) for t in data.get("transitions", [])]
        return cls(
            stages=stages,
            transitions=transitions,
            current_stage=data.get("current_stage"),
            paused=bool(data.get("paused", False)),
            completed=bool(data.get("completed", False)),
        )

    def to_dict(self) -> dict:
        return {
            "stages": [s.to_dict() for s in self.stages],
            "transitions": [t.to_dict() for t in self.transitions],
            "current_stage": self.current_stage,
            "paused": self.paused,
            "completed": self.completed,
        }

    def is_empty(self) -> bool:
        return not self.stages

    def ordered_stages(self) -> list[Stage]:
        return sorted(self.stages, key=lambda s: s.order)

    def stage_by_name(self, name: str) -> Optional[Stage]:
        for s in self.stages:
            if s.name == name:
                return s
        return None

    def active_stage(self) -> Optional[Stage]:
        return self.stage_by_name(self.current_stage) if self.current_stage else None

    def prev_stage(self) -> Optional[Stage]:
        ordered = self.ordered_stages()
        for i, s in enumerate(ordered):
            if s.name == self.current_stage and i - 1 >= 0:
                return ordered[i - 1]
        return None

    def outgoing_transitions(self, from_name: Optional[str]) -> list[Transition]:
        """Все переходы, исходящие из этапа ``from_name``."""
        if not from_name:
            return []
        return [t for t in self.transitions if t.from_stage == from_name]

    def transition(self, from_name: str, to_name: str) -> Optional[Transition]:
        for t in self.transitions:
            if t.from_stage == from_name and t.to == to_name:
                return t
        return None

    def transition_by_id(self, transition_id: str) -> Optional[Transition]:
        for t in self.transitions:
            if t.id == transition_id:
                return t
        return None

    def build_linear_transitions(self) -> list[Transition]:
        """Сгенерировать линейную цепочку переходов по порядку этапов.

        Условия всех переходов — пустые строки.
        """
        ordered = self.ordered_stages()
        result: list[Transition] = []
        for i in range(len(ordered) - 1):
            result.append(
                Transition(
                    id=new_id(),
                    from_stage=ordered[i].name,
                    to=ordered[i + 1].name,
                    condition="",
                )
            )
        return result


def normalize_fsm_data(data: dict) -> tuple[dict, bool]:
    """Привести сырой словарь ``fsm`` к новой схеме (transitions).

    Возвращает (словарь, изменился_ли). Если в данных ещё нет ключа
    ``transitions``, а этапы содержат поле ``condition`` (старые задачи из
    Task 13/14) — выполняется миграция: условие каждого этапа переезжает в
    переход stage[i] → stage[i+1], а из этапов поле ``condition`` удаляется.
    """
    stages = data.get("stages") or []
    if "transitions" in data:
        return data, False

    conditions = [
        (s.get("condition") or "") if isinstance(s, dict) else ""
        for s in stages
    ]
    new_stages = []
    for s in stages:
        if isinstance(s, dict):
            s = dict(s)
            s.pop("condition", None)
            new_stages.append(s)
        else:
            new_stages.append(s)

    transitions = []
    for i in range(len(new_stages) - 1):
        transitions.append(
            {
                "id": new_id(),
                "from": new_stages[i].get("name", ""),
                "to": new_stages[i + 1].get("name", ""),
                "condition": conditions[i],
            }
        )

    out = dict(data)
    out["stages"] = new_stages
    out["transitions"] = transitions
    return out, True


def format_disallowed(current: str, target: Optional[str], allowed: list[str]) -> str:
    """Текст отказа при недопустимом переходе."""
    allowed_str = ", ".join(allowed) if allowed else "нет"
    if target:
        return (
            f"Переход {current} → {target} не разрешён. "
            f"Допустимые переходы из {current}: {allowed_str}."
        )
    return (
        f"Переход из {current} не разрешён. "
        f"Допустимые переходы из {current}: {allowed_str}."
    )
