from __future__ import annotations

from typing import Any


SCALAR_SLOTS = (
    "category",
    "material",
    "color",
    "size",
    "style",
    "brand",
    "budget",
)
LIST_SLOTS = ("feature", "use_case", "other")
ALLOWED_SLOTS = frozenset((*SCALAR_SLOTS, *LIST_SLOTS))
ASK_ORDER = (
    "category",
    "color",
    "material",
    "size",
    "style",
    "brand",
    "budget",
)


def _normalize_name(name: str) -> str | None:
    key = str(name or "").strip().lower()
    return key if key in ALLOWED_SLOTS else None


def _coerce_value(name: str, value: object) -> Any:
    if value is None:
        return None
    if name == "budget":
        if isinstance(value, bool):
            raise ValueError("budget must be a number")
        if isinstance(value, (int, float)):
            return int(value)
        text = str(value).replace(",", "").strip()
        digits = "".join(ch if ch.isdigit() else " " for ch in text)
        parts = [part for part in digits.split() if part]
        if not parts:
            raise ValueError(f"could not parse budget from {value!r}")
        return int(parts[-1])
    text = str(value).strip()
    if not text:
        raise ValueError("value must not be empty")
    return text


class SessionState:
    """Per-conversation shopping slots. Tools mutate this object; it never wipes unspecified keys."""

    def __init__(self, session_id: str, user_profile: dict | None = None) -> None:
        self.session_id = session_id
        self.user_profile = dict(user_profile or {})
        self.slots: dict[str, Any] = {name: None for name in SCALAR_SLOTS}
        self.slots.update({name: [] for name in LIST_SLOTS})
        self.unspecified: set[str] = set()
        self.history: list[dict[str, Any]] = []
        self.last_assistant_message: str | None = None

    def _record(self, turn: int | None, action: str, name: str, old: Any, new: Any) -> None:
        self.history.append(
            {
                "turn": turn,
                "action": action,
                "name": name,
                "old": old,
                "new": new,
            }
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "slots": {key: value for key, value in self.slots.items() if value not in (None, [])},
            "unspecified": sorted(self.unspecified),
        }

    def get_state(self) -> dict[str, Any]:
        data = self.snapshot()
        data["history"] = self.history[-8:]
        return data

    def set_slot(self, name: str, value: object, turn: int | None = None) -> dict[str, Any]:
        key = _normalize_name(name)
        if key is None:
            return {"ok": False, "error": f"unknown slot {name!r}", **self.snapshot()}
        try:
            coerced = _coerce_value(key, value)
        except ValueError as exc:
            return {"ok": False, "error": str(exc), **self.snapshot()}

        current = self.slots[key]
        if key in LIST_SLOTS:
            old = list(current)
            if coerced not in current:
                current.append(coerced)
            self.unspecified.discard(key)
            self._record(turn, "set", key, old, list(current))
            return {"ok": True, **self.snapshot()}

        if current not in (None, "") and current != coerced:
            return {
                "ok": False,
                "error": f"{key} is already {current!r}; call update_slot to rewrite",
                **self.snapshot(),
            }
        old = current
        self.slots[key] = coerced
        self.unspecified.discard(key)
        self._record(turn, "set", key, old, coerced)
        return {"ok": True, **self.snapshot()}

    def update_slot(self, name: str, value: object, turn: int | None = None) -> dict[str, Any]:
        key = _normalize_name(name)
        if key is None:
            return {"ok": False, "error": f"unknown slot {name!r}", **self.snapshot()}
        try:
            coerced = _coerce_value(key, value)
        except ValueError as exc:
            return {"ok": False, "error": str(exc), **self.snapshot()}

        if key in LIST_SLOTS:
            old = list(self.slots[key])
            self.slots[key] = [coerced]
            self.unspecified.discard(key)
            self._record(turn, "update", key, old, list(self.slots[key]))
            return {"ok": True, **self.snapshot()}

        old = self.slots[key]
        self.slots[key] = coerced
        self.unspecified.discard(key)
        self._record(turn, "update", key, old, coerced)
        return {"ok": True, **self.snapshot()}

    def clear_slot(self, name: str, value: object | None = None, turn: int | None = None) -> dict[str, Any]:
        key = _normalize_name(name)
        if key is None:
            return {"ok": False, "error": f"unknown slot {name!r}", **self.snapshot()}

        if key in LIST_SLOTS:
            old = list(self.slots[key])
            if value is None:
                self.slots[key] = []
            else:
                try:
                    coerced = _coerce_value(key, value)
                except ValueError as exc:
                    return {"ok": False, "error": str(exc), **self.snapshot()}
                self.slots[key] = [item for item in self.slots[key] if item != coerced]
            self._record(turn, "clear", key, old, list(self.slots[key]))
            return {"ok": True, **self.snapshot()}

        old = self.slots[key]
        self.slots[key] = None
        self._record(turn, "clear", key, old, None)
        return {"ok": True, **self.snapshot()}

    def reset_slots(self, keep: list[str] | None = None, turn: int | None = None) -> dict[str, Any]:
        retain = {_normalize_name(item) for item in (keep or [])}
        retain.discard(None)
        old = dict(self.slots)
        for key in SCALAR_SLOTS:
            if key not in retain:
                self.slots[key] = None
        for key in LIST_SLOTS:
            if key not in retain:
                self.slots[key] = []
        self.unspecified -= {key for key in self.unspecified if key not in retain}
        self._record(turn, "reset", "*", old, dict(self.slots))
        return {"ok": True, **self.snapshot()}

    def mark_unspecified(self, name: str, turn: int | None = None) -> dict[str, Any]:
        key = _normalize_name(name)
        if key is None:
            return {"ok": False, "error": f"unknown slot {name!r}", **self.snapshot()}
        self.clear_slot(key, turn=turn)
        self.unspecified.add(key)
        self._record(turn, "unspecified", key, None, None)
        return {"ok": True, **self.snapshot()}

    def next_ask_attribute(self) -> str | None:
        for name in ASK_ORDER:
            if name in self.unspecified:
                continue
            value = self.slots.get(name)
            if value in (None, "", []):
                return name
        return None

    def query_terms(self) -> list[str]:
        terms: list[str] = []
        for name in (*SCALAR_SLOTS, *LIST_SLOTS):
            if name == "budget":
                continue
            value = self.slots.get(name)
            if value in (None, "", []):
                continue
            if isinstance(value, list):
                terms.extend(str(item) for item in value)
            else:
                terms.append(str(value))
        return terms

    def budget_limit(self) -> int | None:
        value = self.slots.get("budget")
        return int(value) if isinstance(value, int) else None
