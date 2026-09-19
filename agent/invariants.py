"""InvariantStore — работа с инвариантами агента.

Инварианты — жёсткие ограничения, которые агент не имеет права нарушать:
выбранная архитектура, принятые технические решения, ограничения по стеку,
бизнес-правила. Это отдельный слой, более приоритетный, чем профиль.

Хранение:
- один файл на агента: memory/<agent>/invariants.json.

Привязка — к агенту (не к задаче, не глобально). Инварианты действуют всегда:
и при активной задаче, и без неё, и при задаче на паузе, и при завершённой задаче.

Идентификаторы — UUID и не переиспользуются после удаления.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from .errors import MemoryError
from .memory import new_id, now_iso

CATEGORIES = ("architecture", "tech", "stack", "business", "other")


class InvariantStore:
    """Загрузка/сохранение и CRUD-операции над инвариантами агента."""

    def __init__(self, memory_dir: str | Path) -> None:
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    @property
    def invariants_path(self) -> Path:
        return self.memory_dir / "invariants.json"

    # ------------------------------------------------------- ввод-вывод JSON
    def _load(self) -> dict:
        if not self.invariants_path.exists():
            return {"invariants": []}
        try:
            data = json.loads(self.invariants_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise MemoryError(
                f"Не удалось прочитать файл инвариантов {self.invariants_path}: {exc}"
            ) from exc
        if not isinstance(data, dict) or not isinstance(data.get("invariants"), list):
            raise MemoryError(
                f"Некорректная структура файла инвариантов {self.invariants_path}."
            )
        return data

    def _save(self, data: dict) -> None:
        try:
            self.invariants_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            raise MemoryError(
                f"Не удалось записать файл инвариантов {self.invariants_path}: {exc}"
            ) from exc

    # ------------------------------------------------------------------- CRUD
    def list(self) -> list[dict]:
        """Вернуть список инвариантов (как в файле)."""
        return list(self._load().get("invariants", []))

    def get(self, invariant_id: str) -> Optional[dict]:
        for invariant in self.list():
            if invariant.get("id") == invariant_id:
                return invariant
        return None

    def add(self, category: str, text: str) -> dict:
        category = (category or "").strip().lower()
        if category not in CATEGORIES:
            raise MemoryError(
                "Категория должна быть: architecture | tech | stack | business | other."
            )
        text = (text or "").strip()
        if not text:
            raise MemoryError("Текст инварианта не может быть пустым.")
        data = self._load()
        invariant = {
            "id": new_id(),
            "category": category,
            "text": text,
            "ts": now_iso(),
        }
        data.setdefault("invariants", []).append(invariant)
        self._save(data)
        return invariant

    def delete(self, invariant_id: str) -> bool:
        data = self._load()
        invariants = data.get("invariants", [])
        for index, invariant in enumerate(invariants):
            if invariant.get("id") == invariant_id:
                invariants.pop(index)
                self._save(data)
                return True
        return False

    def edit(self, invariant_id: str, text: str) -> dict:
        text = (text or "").strip()
        if not text:
            raise MemoryError("Текст инварианта не может быть пустым.")
        data = self._load()
        for invariant in data.get("invariants", []):
            if invariant.get("id") == invariant_id:
                invariant["text"] = text
                invariant["ts"] = now_iso()
                self._save(data)
                return invariant
        raise MemoryError(f"Инвариант {invariant_id} не найден.")

    def clear(self) -> None:
        self._save({"invariants": []})
