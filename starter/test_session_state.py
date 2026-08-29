from __future__ import annotations

import unittest
from types import SimpleNamespace

from starter.session_state import SessionState
from starter.state_tracker import dispatch, track


class SessionStateTests(unittest.TestCase):
    def test_information_accumulation(self) -> None:
        session = SessionState("s1")
        session.set_slot("category", "jacket", turn=1)
        session.set_slot("style", "winter", turn=1)
        session.set_slot("color", "red", turn=2)
        session.set_slot("budget", "under $100", turn=2)
        self.assertEqual(
            session.snapshot()["slots"],
            {"category": "jacket", "style": "winter", "color": "red", "budget": 100},
        )

    def test_intent_override_rewrite_and_add(self) -> None:
        session = SessionState("s1")
        session.set_slot("category", "jacket")
        session.set_slot("style", "winter")
        session.set_slot("color", "red")
        session.set_slot("budget", 100)
        session.update_slot("color", "blue")
        session.set_slot("material", "leather")
        self.assertEqual(
            session.snapshot()["slots"],
            {
                "category": "jacket",
                "style": "winter",
                "color": "blue",
                "material": "leather",
                "budget": 100,
            },
        )

    def test_set_slot_refuses_scalar_conflict(self) -> None:
        session = SessionState("s1")
        session.set_slot("color", "red")
        result = session.set_slot("color", "blue")
        self.assertFalse(result["ok"])
        self.assertEqual(session.slots["color"], "red")
        session.update_slot("color", "blue")
        self.assertEqual(session.slots["color"], "blue")

    def test_clear_slot_erases_only_named_constraint(self) -> None:
        session = SessionState("s1")
        session.set_slot("color", "red")
        session.set_slot("budget", 100)
        session.clear_slot("color")
        self.assertIsNone(session.slots["color"])
        self.assertEqual(session.slots["budget"], 100)

    def test_category_pivot_resets_stale_filters(self) -> None:
        session = SessionState("s1")
        session.set_slot("category", "jacket")
        session.set_slot("color", "blue")
        session.set_slot("material", "leather")
        session.set_slot("budget", 100)
        session.reset_slots()
        session.set_slot("category", "boots")
        self.assertEqual(session.snapshot()["slots"], {"category": "boots"})

    def test_dispatch_tools(self) -> None:
        session = SessionState("s1")
        dispatch(session, "set_slot", {"name": "color", "value": "red"}, turn=1)
        dispatch(session, "update_slot", {"name": "color", "value": "blue"}, turn=2)
        self.assertEqual(session.slots["color"], "blue")


class FakeCompletions:
    def __init__(self, script: list[object]) -> None:
        self.script = list(script)

    def __call__(self, **kwargs: object) -> object:
        return self.script.pop(0)


def _message(tool_calls: list[object] | None = None, content: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=tool_calls, content=content))],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
    )


def _call(call_id: str, name: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class TrackerLoopTests(unittest.TestCase):
    def test_mock_tool_loop_accumulates_then_overrides(self) -> None:
        session = SessionState("s1")
        first = _message(
            [
                _call("1", "set_slot", '{"name":"category","value":"jacket"}'),
                _call("2", "set_slot", '{"name":"style","value":"winter"}'),
            ]
        )
        second = _message(content="Noted a winter jacket.")
        create = FakeCompletions([first, second])
        usage = track(session, "I want a winter jacket.", 1, create=create)
        self.assertEqual(session.slots["category"], "jacket")
        self.assertEqual(session.slots["style"], "winter")
        self.assertEqual(usage["prompt_tokens"], 6)
        self.assertEqual(usage["completion_tokens"], 4)

        override = _message(
            [
                _call("3", "update_slot", '{"name":"color","value":"blue"}'),
                _call("4", "set_slot", '{"name":"material","value":"leather"}'),
            ]
        )
        done = _message(content="Showing blue leather jackets.")
        create = FakeCompletions([override, done])
        session.set_slot("color", "red")
        session.set_slot("budget", 100)
        track(session, "Forget red, show blue leather jackets.", 3, create=create)
        self.assertEqual(
            session.snapshot()["slots"],
            {
                "category": "jacket",
                "style": "winter",
                "color": "blue",
                "material": "leather",
                "budget": 100,
            },
        )


if __name__ == "__main__":
    unittest.main()
