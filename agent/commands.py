"""Обработка команд (единый источник описаний команд для CLI и /help)."""
from __future__ import annotations

from dataclasses import dataclass

from .agent import Agent
from .errors import AgentError
from .fsm import format_disallowed
from .manager import AgentManager

# Единый источник описаний команд для /help.
COMMANDS: dict[str, str] = {
    "/agents": "список агентов, активный помечен.",
    "/agent <name>": "переключить активного агента.",
    "/task new <name>": "создать и активировать задачу (краткосрочная очищается).",
    "/task switch <name>": "активировать существующую задачу (краткосрочная очищается).",
    "/tasks": "список задач активного агента.",
    "/task current": "текущая активная задача.",
    "/task stages list": "показать этапы и исходящие переходы (активный помечен).",
    "/task stages set": "полностью перезаписать набор этапов (интерактивно); очищает рабочую память.",
    "/task transitions list": "показать все переходы задачи (from → to, условие).",
    "/task transitions set": "полностью перезаписать граф переходов (интерактивно).",
    "/task transitions add <from> <to> \"<condition>\"": "добавить переход.",
    "/task transitions del <id>": "удалить переход по ID.",
    "/task transitions edit <id> condition \"<текст>\"": "изменить условие перехода.",
    "/task transitions clear": "очистить все переходы (задача станет терминальной).",
    "/task state": "показать состояние: этап, ожидаемое действие, допустимые переходы, статус, пауза/завершено.",
    "/task next": "перейти по разрешённому переходу вперёд.",
    "/task back": "вернуться назад (только при явном обратном переходе).",
    "/task goto <name>": "перейти к этапу (только по разрешённому переходу).",
    "/task check": "вручную проверить условия исходящих переходов.",
    "/task pause": "поставить задачу на паузу.",
    "/task resume": "снять с паузы.",
    "/task restart": "начать задачу заново с первого этапа (краткосрочная память очищается).",
    "/profile list": "список профилей агента, активный помечен.",
    "/profile current": "показать имя активного профиля.",
    "/profile show [<name>]": "показать содержимое профиля (по умолчанию — активного).",
    "/profile new <name>": "создать профиль с пустыми обязательными полями.",
    "/profile switch <name>": "сделать профиль активным.",
    "/profile del <name>": "удалить профиль (активный становится не задан).",
    "/profile set <name> style|format|constraints <value>": "задать поле профиля.",
    "/profile unset <name> style|format|constraints": "очистить поле профиля.",
    "/profile add-note <name> <текст>": "добавить заметку в профиль.",
    "/profile del-note <name> <index>": "удалить заметку профиля по индексу.",
    "/memory show long|working|short": "показать записи слоя.",
    "/memory add long decision|knowledge <текст>": "добавить в долговременную.",
    "/memory add working <текст>": "добавить в рабочую (нужна активная задача).",
    "/memory add short <текст>": "добавить в краткосрочную.",
    "/memory del <layer> <id>": "удалить запись по ID.",
    "/memory clear short|working|long": "очистить слой.",
    "/memory files": "пути к файлам памяти, профилям и инвариантам текущего агента.",
    "/memory usage profile|long|working|short|invariant on|off": "включить/выключить слой в промте.",
    "/invariant list": "список инвариантов агента (ID, категория, текст).",
    "/invariant add <category> <текст>": "добавить инвариант (architecture|tech|stack|business|other).",
    "/invariant del <id>": "удалить инвариант по ID.",
    "/invariant edit <id> <текст>": "изменить текст инварианта.",
    "/invariant clear": "очистить все инварианты.",
    "/invariant check on|off": "включить/выключить пост-проверку LLM-валидатором.",
    "/debug on|off": "показывать записи, попавшие в промт (включая инварианты и пост-проверку).",
    "/metrics on|off": "показывать токены и время ответа.",
    "/prompt_save on|off": "спрашивать после ответа, куда сохранить (CLI-вариант A).",
    "/help": "список команд с описанием.",
    "/exit": "выход.",
}

LAYER_NAMES = {
    "invariant": "INVARIANTS",
    "profile": "PROFILE",
    "long": "LONG-TERM",
    "working": "WORKING",
    "short": "SHORT-TERM",
    "fsm": "СОСТОЯНИЕ ЗАДАЧИ",
}


@dataclass
class CommandResult:
    text: str = ""
    exit_requested: bool = False


def help_text() -> str:
    lines = ["Доступные команды:"]
    for command, description in COMMANDS.items():
        lines.append(f"  {command} — {description}")
    return "\n".join(lines)


def process_command(manager: AgentManager, raw: str) -> CommandResult:
    """Обработать строку-команду и вернуть текст ответа."""
    raw = raw.strip()
    if not raw or not raw.startswith("/"):
        return CommandResult()

    parts = raw.split()
    command = parts[0].lower()
    args = parts[1:]

    try:
        if command == "/agents":
            return CommandResult(format_agents(manager))

        if command == "/agent":
            if not args:
                raise AgentError("Укажите имя агента: /agent <name>")
            agent = manager.switch(args[0])
            return CommandResult(f"Активный агент: {agent.name}")

        if command == "/task":
            return CommandResult(handle_task(manager, args))

        if command == "/tasks":
            return CommandResult(format_tasks(manager))

        if command == "/memory":
            return CommandResult(handle_memory(manager, args))

        if command == "/profile":
            return CommandResult(handle_profile(manager, args))

        if command == "/invariant":
            return CommandResult(handle_invariant(manager, args))

        if command == "/debug":
            if len(args) != 1 or args[0] not in ("on", "off"):
                raise AgentError("Использование: /debug on|off")
            manager.active().debug_enabled = args[0] == "on"
            return CommandResult(f"Отладка: {'включена' if args[0] == 'on' else 'выключена'}.")

        if command == "/metrics":
            if len(args) != 1 or args[0] not in ("on", "off"):
                raise AgentError("Использование: /metrics on|off")
            manager.active().metrics_enabled = args[0] == "on"
            return CommandResult(f"Метрики: {'включены' if args[0] == 'on' else 'выключены'}.")

        if command == "/prompt_save":
            if len(args) != 1 or args[0] not in ("on", "off"):
                raise AgentError("Использование: /prompt_save on|off")
            manager.active().prompt_save = args[0] == "on"
            return CommandResult(
                f"Запрос сохранения после ответа: {'включён' if args[0] == 'on' else 'выключен'}."
            )

        if command == "/help":
            return CommandResult(help_text())

        if command == "/exit":
            return CommandResult("До свидания!", exit_requested=True)

        return CommandResult(
            f"Неизвестная команда: {command}. Введите /help для списка команд."
        )
    except AgentError as exc:
        return CommandResult(f"Ошибка: {exc}")


# --------------------------------------------------------------------- задачи
def handle_task(manager: AgentManager, args: list[str]) -> str:
    if not args:
        raise AgentError(
            "Использование: /task new <name> | /task switch <name> | /task current | "
            "/task stages list|set | /task transitions list|set|add|del|edit|clear | "
            "/task state | /task next | /task back | /task goto <name> | "
            "/task check | /task pause | /task resume | /task restart"
        )
    sub = args[0].lower()
    agent = manager.active()

    if sub == "new":
        if len(args) < 2:
            raise AgentError("Укажите имя: /task new <name>")
        name = " ".join(args[1:])
        agent.task_new(name)
        return f"Задача «{name}» создана и активирована. Краткосрочная память очищена."

    if sub == "switch":
        if len(args) < 2:
            raise AgentError("Укажите имя: /task switch <name>")
        name = " ".join(args[1:])
        agent.task_switch(name)
        return f"Задача «{name}» активирована. Краткосрочная память очищена."

    if sub == "current":
        current = agent.task_current()
        return f"Текущая задача: {current}" if current else "Активной задачи нет."

    if sub == "stages":
        return handle_task_stages(agent, args[1:])

    if sub == "transitions":
        return handle_task_transitions(agent, args[1:])

    if sub == "state":
        return format_task_state(agent)

    if sub == "next":
        return handle_task_next(agent)

    if sub == "back":
        return agent.fsm_back()

    if sub == "goto":
        if len(args) < 2:
            raise AgentError("Укажите имя этапа: /task goto <name>")
        return agent.fsm_move(args[1])

    if sub == "check":
        return run_transitions_check(agent, verbose=True)

    if sub == "pause":
        return agent.fsm_pause()

    if sub == "resume":
        return agent.fsm_resume()

    if sub == "restart":
        return agent.fsm_restart()

    raise AgentError("Неизвестная подкоманда /task ...")


def format_tasks(manager: AgentManager) -> str:
    agent = manager.active()
    tasks = agent.task_list()
    if not tasks:
        return "Задач у агента нет."
    lines = [f"Задачи агента «{agent.name}»:"]
    for task in tasks:
        mark = " * (активная)" if task == agent.task_current() else ""
        lines.append(f"  {task}{mark}")
    return "\n".join(lines)


# --------------------------------------------------------------------- FSM
def handle_task_stages(agent: Agent, args: list[str]) -> str:
    if not args:
        raise AgentError("Использование: /task stages list | /task stages set")
    sub = args[0].lower()
    if sub == "list":
        return format_stages_list(agent)
    if sub == "set":
        return prompt_stages_set(agent)
    raise AgentError("Неизвестная подкоманда /task stages ...")


def format_stages_list(agent: Agent) -> str:
    fsm = agent.fsm_info()
    if not fsm or not fsm.get("stages"):
        return "У задачи нет этапов. Задайте их через /task stages set."
    lines = ["Этапы задачи:"]
    current = fsm.get("current_stage")
    transitions = fsm.get("transitions", [])
    for s in fsm["stages"]:
        mark = ""
        if s["name"] == current and not fsm.get("completed"):
            mark = " * (активный)"
        lines.append(f"  {s['order']}. {s['name']}{mark} [{s['status']}]")
        lines.append(f"     описание: {s['description'] or '—'}")
        lines.append(f"     ожидаемое: {s['expected_action'] or '—'}")
        outgoing = [t for t in transitions if t["from"] == s["name"]]
        if outgoing:
            lines.append("     переходы:")
            for t in outgoing:
                lines.append(f"       → {t['to']}  (условие: {t['condition'] or '—'})")
        else:
            lines.append("     переходы: нет")
    if fsm.get("paused"):
        lines.append("Задача на паузе.")
    if fsm.get("completed"):
        lines.append("Задача завершена.")
    return "\n".join(lines)


def format_task_state(agent: Agent) -> str:
    fsm = agent.fsm_info()
    if not fsm or not fsm.get("stages"):
        return "У задачи нет этапов. Задача работает как обычный диалог с рабочей памятью."
    if fsm.get("completed"):
        return "Задача завершена (рабочая память очищена)."
    current = agent.fsm_current_stage()
    if current is None:
        return "Этапы заданы, но активный этап не определён."
    lines = ["Состояние задачи:"]
    lines.append(f"  Этап: {current['name']}")
    lines.append(f"  Описание: {current['description'] or '—'}")
    lines.append(f"  Ожидаемое действие: {current['expected_action'] or '—'}")
    lines.append(f"  Статус: {current['status']}")
    lines.append(f"  Пауза: {'да' if fsm.get('paused') else 'нет'}")
    lines.append(f"  Завершено: {'да' if fsm.get('completed') else 'нет'}")
    lines.append("Допустимые переходы из текущего этапа:")
    outgoing = [t for t in fsm.get("transitions", []) if t["from"] == current["name"]]
    if outgoing:
        for t in outgoing:
            lines.append(f"  → {t['to']}  (условие: {t['condition'] or '—'})")
    else:
        lines.append("  нет (задача завершается)")
    return "\n".join(lines)


def prompt_stages_set(agent: Agent) -> str:
    """Интерактивно задать произвольное количество этапов."""
    try:
        count_raw = input("Сколько этапов задать? ").strip()
    except (EOFError, KeyboardInterrupt):
        return "Ввод отменён."
    try:
        count = int(count_raw)
    except ValueError:
        raise AgentError("Количество этапов должно быть целым числом.") from None
    if count <= 0:
        raise AgentError("Количество этапов должно быть больше нуля.")

    stages: list[dict] = []
    for i in range(1, count + 1):
        try:
            name = input(f"  Этап {i}/{count} — имя (короткое, для команд): ").strip()
            description = input("  Описание этапа: ").strip()
            expected_action = input("  Ожидаемое действие: ").strip()
        except (EOFError, KeyboardInterrupt):
            return "Ввод отменён, этапы не изменены."
        if not name:
            raise AgentError("Имя этапа не может быть пустым.")
        stages.append(
            {
                "name": name,
                "description": description,
                "expected_action": expected_action,
            }
        )

    names = agent.fsm_stages_set(stages)
    return (
        "Набор этапов задан: "
        + ", ".join(names)
        + ". Рабочая память очищена, первый этап активен. "
        "Сгенерирован линейный граф переходов (условия пустые)."
    )


def _strip_quotes(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'"):
        return text[1:-1]
    return text


def handle_task_next(agent: Agent) -> str:
    """Переход вперёд по разрешённым переходам (с выбором, если их несколько)."""
    allowed = agent.fsm_allowed_transitions()
    if not allowed:
        current = agent.fsm_current_stage_name() or "?"
        return format_disallowed(current, None, [])
    if len(allowed) == 1:
        return agent.fsm_move(allowed[0]["to"])
    lines = ["Из текущего этапа несколько разрешённых переходов:"]
    for i, t in enumerate(allowed, 1):
        lines.append(f"  {i}. → {t['to']}  (условие: {t['condition'] or '—'})")
    try:
        choice = input("\n".join(lines) + "\nВыберите номер (или 0 для отмены): ").strip()
    except (EOFError, KeyboardInterrupt):
        return "Переход отменён."
    try:
        index = int(choice)
    except ValueError:
        return "Неверный ввод — переход отменён."
    if 1 <= index <= len(allowed):
        return agent.fsm_move(allowed[index - 1]["to"])
    return "Переход отменён."


def handle_task_transitions(agent: Agent, args: list[str]) -> str:
    if not args:
        raise AgentError(
            "Использование: /task transitions list | /task transitions set | "
            "/task transitions add <from> <to> \"<condition>\" | "
            "/task transitions del <id> | "
            "/task transitions edit <id> condition \"<текст>\" | "
            "/task transitions clear"
        )
    sub = args[0].lower()

    if sub == "list":
        return format_transitions_list(agent)

    if sub == "set":
        return prompt_transitions_set(agent)

    if sub == "add":
        if len(args) < 3:
            raise AgentError(
                "Использование: /task transitions add <from> <to> \"<condition>\""
            )
        frm = args[1]
        to = args[2]
        condition = _strip_quotes(" ".join(args[3:]))
        transition = agent.fsm_transition_add(frm, to, condition)
        return (
            f"Переход добавлен: {transition['from']} → {transition['to']} "
            f"(id {transition['id']}, условие: {transition['condition'] or '—'})."
        )

    if sub == "del":
        if len(args) < 2:
            raise AgentError("Укажите ID: /task transitions del <id>")
        removed = agent.fsm_transition_del(args[1])
        return (
            f"Переход {args[1]} удалён."
            if removed
            else f"Переход {args[1]} не найден."
        )

    if sub == "edit":
        if len(args) < 4 or args[2].lower() != "condition":
            raise AgentError(
                "Использование: /task transitions edit <id> condition \"<текст>\""
            )
        text = _strip_quotes(" ".join(args[3:]))
        transition = agent.fsm_transition_edit(args[1], text)
        return f"Условие перехода {transition['id']} обновлено: {transition['condition']}."

    if sub == "clear":
        return agent.fsm_transitions_clear()

    raise AgentError("Неизвестная подкоманда /task transitions ...")


def format_transitions_list(agent: Agent) -> str:
    fsm = agent.fsm_info()
    if not fsm or not fsm.get("stages"):
        return "У задачи нет этапов. Задайте их через /task stages set."
    transitions = fsm.get("transitions", [])
    if not transitions:
        return "Переходов нет (задача терминальна с текущего этапа)."
    lines = ["Переходы задачи:"]
    for t in transitions:
        lines.append(
            f"  {t['id']}: {t['from']} → {t['to']}  (условие: {t['condition'] or '—'})"
        )
    return "\n".join(lines)


def prompt_transitions_set(agent: Agent) -> str:
    """Интерактивно полностью перезаписать граф переходов."""
    try:
        count_raw = input("Сколько переходов задать? ").strip()
    except (EOFError, KeyboardInterrupt):
        return "Ввод отменён."
    try:
        count = int(count_raw)
    except ValueError:
        raise AgentError("Количество переходов должно быть целым числом.") from None
    if count < 0:
        raise AgentError("Количество переходов не может быть отрицательным.")

    transitions: list[dict] = []
    for i in range(1, count + 1):
        try:
            frm = input(f"  Переход {i}/{count} — из этапа (from): ").strip()
            to = input("  В этап (to): ").strip()
            condition = input("  Условие (естественный язык, можно пусто): ").strip()
        except (EOFError, KeyboardInterrupt):
            return "Ввод отменён, переходы не изменены."
        transitions.append({"from": frm, "to": to, "condition": condition})

    result = agent.fsm_transitions_set(transitions)
    return f"Граф переходов задан: {len(result)} переход(ов)."


def run_transitions_check(agent: Agent, verbose: bool = False) -> str:
    """Проверить условия исходящих переходов (общая для авто и /task check)."""
    try:
        result = agent.check_transitions()
    except AgentError as exc:
        return f"Ошибка проверки условий: {exc}"
    if result is None:
        return (
            "Проверка недоступна (нет этапов, нет последнего обмена, задача на паузе "
            "или завершена)."
            if verbose
            else ""
        )
    if result.get("terminal"):
        return result.get("message", "Задача завершена.")

    triggered = result.get("triggered", [])
    if not triggered:
        return "Условия переходов ещё не выполнены." if verbose else ""

    if len(triggered) == 1:
        t = triggered[0]
        try:
            answer = input(
                f"Условие перехода {t['from']} → {t['to']} выполнено. Перейти? [y/n] "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "Переход отменён."
        if answer in ("y", "yes", "д", "да"):
            moved = agent.fsm_move(t["to"])
            terminal = agent.fsm_finish_if_terminal()
            return f"{moved} {terminal}" if terminal else moved
        return "Остаёмся на текущем этапе."

    lines = ["Сработали следующие переходы:"]
    for i, t in enumerate(triggered, 1):
        lines.append(
            f"  {i}. {t['from']} → {t['to']}  (условие: {t['condition'] or '—'})"
        )
    try:
        choice = input("\n".join(lines) + "\nВыберите номер (или 0 для отмены): ").strip()
    except (EOFError, KeyboardInterrupt):
        return "Переход отменён."
    try:
        index = int(choice)
    except ValueError:
        return "Неверный ввод — остаёмся на текущем этапе."
    if 1 <= index <= len(triggered):
        moved = agent.fsm_move(triggered[index - 1]["to"])
        terminal = agent.fsm_finish_if_terminal()
        return f"{moved} {terminal}" if terminal else moved
    return "Переход отменён."


# --------------------------------------------------------------------- память
def handle_memory(manager: AgentManager, args: list[str]) -> str:
    if not args:
        raise AgentError(
            "Использование: /memory show|add|del|clear|files|usage ..."
        )
    sub = args[0].lower()
    agent = manager.active()

    if sub == "show":
        if len(args) < 2:
            raise AgentError("Укажите слой: /memory show long|working|short")
        return format_memory_show(agent, args[1].lower())

    if sub == "add":
        return handle_memory_add(agent, args[1:])

    if sub == "del":
        if len(args) < 3:
            raise AgentError("Использование: /memory del <layer> <id>")
        layer = args[1].lower()
        record_id = args[2]
        removed = agent.memory_del(layer, record_id)
        return (
            f"Запись {record_id} удалена."
            if removed
            else f"Запись {record_id} не найдена."
        )

    if sub == "clear":
        if len(args) < 2:
            raise AgentError("Укажите слой: /memory clear short|working|long")
        layer = args[1].lower()
        agent.memory_clear(layer)
        return f"Слой {layer} очищен."

    if sub == "files":
        return format_memory_files(agent)

    if sub == "usage":
        if len(args) < 3:
            raise AgentError(
                "Использование: /memory usage profile|long|working|short on|off"
            )
        layer = args[1].lower()
        flag = args[2].lower()
        value = agent.memory_usage(layer, flag)
        state = "включён" if value else "выключен"
        return f"Слой {layer} в промте: {state}."

    raise AgentError("Неизвестная подкоманда /memory ...")


def handle_memory_add(agent: Agent, args: list[str]) -> str:
    if not args:
        raise AgentError(
            "Использование: /memory add long <type> <текст> | "
            "/memory add working <текст> | /memory add short <текст>"
        )
    layer = args[0].lower()

    if layer == "long":
        if len(args) < 3:
            raise AgentError(
                "Использование: /memory add long decision|knowledge <текст>"
            )
        record_type = args[1].lower()
        content = " ".join(args[2:])
        record = agent.memory_add("long", content, record_type=record_type)
        return f"Запись добавлена в long ({record_type}): {record['id']}"

    if layer == "working":
        if len(args) < 2:
            raise AgentError("Использование: /memory add working <текст>")
        content = " ".join(args[1:])
        record = agent.memory_add("working", content)
        return f"Запись добавлена в working: {record['id']}"

    if layer == "short":
        if len(args) < 2:
            raise AgentError("Использование: /memory add short <текст>")
        content = " ".join(args[1:])
        record = agent.memory_add("short", content)
        return f"Запись добавлена в short: {record['id']}"

    raise AgentError("Неизвестный слой для /memory add ...")


# --------------------------------------------------------------------- профили
def handle_profile(manager: AgentManager, args: list[str]) -> str:
    if not args:
        raise AgentError(
            "Использование: /profile list|current|show|new|switch|del|set|unset|"
            "add-note|del-note ..."
        )
    sub = args[0].lower()
    agent = manager.active()

    if sub == "list":
        return format_profile_list(agent)

    if sub == "current":
        return format_profile_current(agent)

    if sub == "show":
        name = args[1] if len(args) >= 2 else None
        return format_profile_show(agent, name)

    if sub == "new":
        if len(args) < 2:
            raise AgentError("Укажите имя: /profile new <name>")
        agent.profile_new(args[1])
        return f"Профиль «{args[1]}» создан."

    if sub == "switch":
        if len(args) < 2:
            raise AgentError("Укажите имя: /profile switch <name>")
        agent.profile_switch(args[1])
        return f"Активный профиль: {args[1]}."

    if sub == "del":
        if len(args) < 2:
            raise AgentError("Укажите имя: /profile del <name>")
        agent.profile_del(args[1])
        return f"Профиль «{args[1]}» удалён."

    if sub == "set":
        if len(args) < 4:
            raise AgentError("Использование: /profile set <name> <field> <value>")
        name = args[1]
        field = args[2].lower()
        value = " ".join(args[3:])
        agent.profile_set(name, field, value)
        return f"Профиль «{name}»: поле {field} обновлено."

    if sub == "unset":
        if len(args) < 3:
            raise AgentError("Использование: /profile unset <name> <field>")
        name = args[1]
        field = args[2].lower()
        agent.profile_unset(name, field)
        return f"Профиль «{name}»: поле {field} очищено."

    if sub == "add-note":
        if len(args) < 3:
            raise AgentError("Использование: /profile add-note <name> <текст>")
        name = args[1]
        text = " ".join(args[2:])
        agent.profile_add_note(name, text)
        return f"Заметка добавлена в профиль «{name}»."

    if sub == "del-note":
        if len(args) < 3:
            raise AgentError("Использование: /profile del-note <name> <index>")
        name = args[1]
        try:
            index = int(args[2])
        except ValueError:
            raise AgentError("Индекс заметки должен быть целым числом.") from None
        agent.profile_del_note(name, index)
        return f"Заметка удалена из профиля «{name}»."

    raise AgentError("Неизвестная подкоманда /profile ...")


# --------------------------------------------------------------------- инварианты
def handle_invariant(manager: AgentManager, args: list[str]) -> str:
    if not args:
        raise AgentError(
            "Использование: /invariant list | /invariant add <category> <текст> | "
            "/invariant del <id> | /invariant edit <id> <текст> | "
            "/invariant clear | /invariant check on|off"
        )
    sub = args[0].lower()
    agent = manager.active()

    if sub == "list":
        return format_invariant_list(agent)

    if sub == "add":
        if len(args) < 3:
            raise AgentError(
                "Использование: /invariant add <category> <текст> "
                "(category: architecture|tech|stack|business|other)"
            )
        category = args[1]
        text = " ".join(args[2:])
        invariant = agent.invariant_add(category, text)
        return (
            f"Инвариант добавлен: {invariant['id']} "
            f"[{invariant['category']}] {invariant['text']}"
        )

    if sub == "del":
        if len(args) < 2:
            raise AgentError("Укажите ID: /invariant del <id>")
        removed = agent.invariant_del(args[1])
        return (
            f"Инвариант {args[1]} удалён."
            if removed
            else f"Инвариант {args[1]} не найден."
        )

    if sub == "edit":
        if len(args) < 3:
            raise AgentError("Использование: /invariant edit <id> <текст>")
        invariant = agent.invariant_edit(args[1], " ".join(args[2:]))
        return f"Инвариант {invariant['id']} обновлён: {invariant['text']}"

    if sub == "clear":
        agent.invariant_clear()
        return "Все инварианты очищены."

    if sub == "check":
        if len(args) < 2:
            raise AgentError("Использование: /invariant check on|off")
        value = agent.invariant_check_set(args[1].lower())
        state = "включена" if value else "выключена"
        return f"Пост-проверка инвариантов: {state}."

    raise AgentError("Неизвестная подкоманда /invariant ...")


def format_invariant_list(agent: Agent) -> str:
    invariants = agent.invariant_list()
    if not invariants:
        return "Инвариантов нет."
    lines = ["Инварианты агента:"]
    for inv in invariants:
        lines.append(
            f"  {inv['id']} [{inv.get('category', 'other')}] {inv.get('text', '')}"
        )
    return "\n".join(lines)


def format_profile_list(agent: Agent) -> str:
    names = agent.profile_list()
    if not names:
        return "Профилей нет."
    active = agent.profile_current()
    lines = ["Профили агента:"]
    for name in names:
        mark = " * (активный)" if name == active else ""
        lines.append(f"  {name}{mark}")
    return "\n".join(lines)


def format_profile_current(agent: Agent) -> str:
    active = agent.profile_current()
    return f"Активный профиль: {active}" if active else "Активный профиль не задан."


def format_profile_show(agent: Agent, name: str | None = None) -> str:
    profile = agent.profile_show(name)
    lines = [f"Профиль «{profile.get('name')}»:"]
    lines.append(f"  style: {profile.get('style', '')}")
    lines.append(f"  format: {profile.get('format', '')}")
    lines.append(f"  constraints: {profile.get('constraints', '')}")
    lines.append(f"  created_at: {profile.get('created_at', '')}")
    lines.append(f"  updated_at: {profile.get('updated_at', '')}")
    notes = profile.get("notes", [])
    if notes:
        lines.append("  notes:")
        for index, note in enumerate(notes):
            lines.append(f"    [{index}] {note}")
    else:
        lines.append("  notes: (нет)")
    return "\n".join(lines)


# ------------------------------------------------------------------ форматирование
def format_agents(manager: AgentManager) -> str:
    lines = ["Агенты:"]
    for name in manager.list_names():
        mark = " * (активный)" if name == manager.active_name else ""
        lines.append(f"  {name}{mark}")
    return "\n".join(lines)


def format_memory_show(agent: Agent, layer: str) -> str:
    if layer == "long":
        data = agent.memory_show("long")
        records = data.get("records", [])
        if not records:
            return "Долговременная память пуста."
        lines = ["Долговременная память (long):"]
        for r in records:
            lines.append(f"  [{r['type']}] {r['id']} — {r['content']} ({r['ts']})")
        return "\n".join(lines)

    if layer == "working":
        data = agent.memory_show("working")
        records = data.get("records", [])
        if not records:
            return f"Рабочая память (задача «{data['task']}») пуста."
        lines = [f"Рабочая память (задача «{data['task']}»):"]
        for r in records:
            lines.append(f"  {r['id']} — {r['content']} ({r['ts']})")
        return "\n".join(lines)

    if layer == "short":
        data = agent.memory_show("short")
        messages = data.get("messages", [])
        if not messages:
            return "Краткосрочная память пуста."
        lines = [f"Краткосрочная память (задача «{data.get('task')}»):"]
        for m in messages:
            role = "Пользователь" if m["role"] == "user" else "Ассистент"
            lines.append(f"  {m['id']} [{role}] {m['content']} ({m['ts']})")
        return "\n".join(lines)

    raise AgentError("Укажите слой: long | working | short")


def format_memory_files(agent: Agent) -> str:
    files = agent.memory_files()
    lines = ["Файлы памяти агента:"]
    lines.append(f"  invariants: {files['invariants']}")
    lines.append(f"  long-term : {files['long']}")
    lines.append(f"  short-term: {files['short']}")
    if files["working"]:
        lines.append(f"  working   : {files['working']}")
    else:
        lines.append("  working   : — (нет активной задачи)")
    lines.append(f"  profiles  : {files['profiles_dir']}")
    if files["active_profile"]:
        lines.append(f"  активный профиль: {files['active_profile']}")
    else:
        lines.append("  активный профиль: — (нет активного профиля)")
    return "\n".join(lines)


def format_debug(agent: Agent) -> str:
    if not agent.last_debug:
        return "Отладочный вывод недоступен (ещё не было запросов)."
    lines = ["Записи, отправленные в промт:"]
    for section in agent.last_debug:
        layer = section["layer"]
        if layer == "fsm":
            lines.append("\n" + section.get("text", ""))
            continue
        lines.append(f"\n[{LAYER_NAMES[layer]}]")
        if layer == "invariant":
            for inv in section.get("invariants", []):
                lines.append(f"  {inv.get('id')} — {inv.get('text', '')}")
        elif layer == "profile":
            lines.append(f"  Профиль: {section.get('name')}")
            fields = section.get("fields", {})
            lines.append(f"  style: {fields.get('style', '')}")
            lines.append(f"  format: {fields.get('format', '')}")
            lines.append(f"  constraints: {fields.get('constraints', '')}")
            notes = section.get("notes", [])
            if notes:
                lines.append("  notes:")
                for note in notes:
                    lines.append(f"    - {note}")
        elif layer == "short":
            for m in section.get("messages", []):
                lines.append(f"  ({m['role']}) {m['content']}")
        else:
            for r in section.get("records", []):
                prefix = f"[{r.get('type')}] " if layer == "long" else ""
                lines.append(f"  {prefix}{r['content']}")
    if agent.last_invariant_check is not None:
        check = agent.last_invariant_check
        if check.get("status") == "VIOLATION":
            lines.append(
                f"\nПост-проверка инвариантов: VIOLATION: {check.get('invariant_id', '?')}"
            )
        else:
            lines.append("\nПост-проверка инвариантов: OK")
    if agent.last_transition_check is not None:
        tc = agent.last_transition_check
        if tc.get("terminal"):
            lines.append("\nПроверка переходов: терминальный этап — задача завершена.")
        else:
            checked = tc.get("checked", [])
            if not checked:
                lines.append("\nПроверка переходов: нет переходов с условиями для проверки.")
            else:
                lines.append("\nПроверка переходов:")
                for c in checked:
                    verdict = "да" if c["result"] else "нет"
                    lines.append(
                        f"  {c['from']} → {c['to']}  (условие: {c['condition'] or '—'}): {verdict}"
                    )
            triggered = tc.get("triggered", [])
            if triggered:
                lines.append("Сработавшие переходы:")
                for t in triggered:
                    lines.append(f"  {t['from']} → {t['to']}")
    return "\n".join(lines)


def format_metrics(agent: Agent) -> str:
    if not agent.last_metrics:
        return "Метрики недоступны (ещё не было запросов)."
    m = agent.last_metrics
    lines = ["Метрики последнего ответа:"]
    lines.append(f"  Время: {m['elapsed']} с")
    lines.append(
        "  Токены (prompt/completion/total): "
        f"{m['prompt_tokens']} / {m['completion_tokens']} / {m['total_tokens']}"
    )
    if m.get("model"):
        lines.append(f"  Модель: {m['model']}")
    return "\n".join(lines)



