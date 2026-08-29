from __future__ import annotations

import json
import os
from typing import Any, Callable

from starter.session_state import ALLOWED_SLOTS, SessionState

SLOT_ENUM = sorted(ALLOWED_SLOTS)

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_state",
            "description": "Read the current shopping slot state and recent mutation history.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_slot",
            "description": (
                "Accumulate a new constraint. Fails if a scalar slot already has a different value; "
                "use update_slot to rewrite."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "enum": SLOT_ENUM},
                    "value": {"type": ["string", "integer"]},
                },
                "required": ["name", "value"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_slot",
            "description": "Intent override: overwrite a slot (for example color red -> blue).",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "enum": SLOT_ENUM},
                    "value": {"type": ["string", "integer"]},
                },
                "required": ["name", "value"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "clear_slot",
            "description": "Erase a slot. For list slots, pass value to remove one item only.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "enum": SLOT_ENUM},
                    "value": {"type": ["string", "integer", "null"]},
                },
                "required": ["name"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reset_slots",
            "description": (
                "Full search pivot (new product type). Clears all slots except those listed in keep."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keep": {
                        "type": "array",
                        "items": {"type": "string", "enum": SLOT_ENUM},
                    }
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mark_unspecified",
            "description": "User has no preference for this attribute; do not ask for it again.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string", "enum": SLOT_ENUM}},
                "required": ["name"],
                "additionalProperties": False,
            },
        },
    },
]

SYSTEM_PROMPT = """You are a shopping slot tracker for one conversation.
Call tools to update SessionState. Do not invent constraints the user did not state.
Rules:
- New information that does not conflict: set_slot (accumulation). Prior slots must remain.
- Contradiction, replacement, or "instead": update_slot (rewrite) or clear_slot then set_slot.
- "Forget X" / "ignore my earlier preference": clear_slot or update_slot for only the obsolete keys.
- Completely different product type: reset_slots, then set_slot for the new search.
- "I don't have a preference for {attribute}": mark_unspecified.
- budget is a numeric max price (under $100 -> 100).
Never dump a full JSON state as the final answer; tools are the source of truth.
After tools have been applied, reply with a short confirmation only."""


def _parse_arguments(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def dispatch(session: SessionState, name: str, arguments: dict[str, Any], turn: int) -> dict[str, Any]:
    if name == "get_state":
        return session.get_state()
    if name == "set_slot":
        return session.set_slot(arguments.get("name", ""), arguments.get("value"), turn=turn)
    if name == "update_slot":
        return session.update_slot(arguments.get("name", ""), arguments.get("value"), turn=turn)
    if name == "clear_slot":
        return session.clear_slot(arguments.get("name", ""), arguments.get("value"), turn=turn)
    if name == "reset_slots":
        keep = arguments.get("keep")
        return session.reset_slots(keep if isinstance(keep, list) else None, turn=turn)
    if name == "mark_unspecified":
        return session.mark_unspecified(arguments.get("name", ""), turn=turn)
    return {"ok": False, "error": f"unknown tool {name}"}


def _usage_from(payload: Any) -> tuple[int, int]:
    usage = getattr(payload, "usage", None)
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0) if usage is not None else 0
    completion = int(getattr(usage, "completion_tokens", 0) or 0) if usage is not None else 0
    return max(0, prompt), max(0, completion)


def track(
    session: SessionState,
    user_message: str,
    turn: int,
    *,
    client: Any | None = None,
    create: Callable[..., Any] | None = None,
    model: str | None = None,
    max_rounds: int = 4,
) -> dict[str, int]:
    """Run the Groq/OpenAI tool loop against session. Returns token usage."""
    prompt_tokens = 0
    completion_tokens = 0
    completer = create
    if completer is None:
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        print(api_key if api_key else "No API KEY FOUND", user_message)
        if not api_key:
            return {"prompt_tokens": 0, "completion_tokens": 0, "Reason": "No API KEY FOUND"}
        if client is None:
            try:
                from openai import OpenAI
            except ImportError:
                return {"prompt_tokens": 0, "completion_tokens": 0}
            client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
        completer = client.chat.completions.create

    chosen_model = model or os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    snapshot = json.dumps(session.snapshot(), ensure_ascii=True)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"Current slots: {snapshot}"},
    ]
    if session.last_assistant_message:
        messages.append({"role": "assistant", "content": session.last_assistant_message})
    messages.append({"role": "user", "content": user_message})

    for _ in range(max_rounds):
        print("messages", messages)
        response = completer(
            model=chosen_model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        p, c = _usage_from(response)
        prompt_tokens += p
        completion_tokens += c
        choice = response.choices[0].message
        tool_calls = getattr(choice, "tool_calls", None) or []
        if not tool_calls:
            text = getattr(choice, "content", None)
            if isinstance(text, str) and text.strip():
                session.last_assistant_message = text.strip()
                print("\n=== ASSISTANT OUTPUT ===")
                print(text.strip())
                print("========================\n")
            break
        messages.append(
            {
                "role": "assistant",
                "content": getattr(choice, "content", None) or "",
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments or "{}",
                        },
                    }
                    for call in tool_calls
                ],
            }
        )
        for call in tool_calls:
            print("\n=== TOOL CALL ===")
            print("Tool:", call.function.name)
            print("Arguments:", call.function.arguments)
            print("=================\n")

            result = dispatch(
                session,
                call.function.name,
                _parse_arguments(call.function.arguments),
                turn
            )

            print("=== TOOL RESULT ===")
            print(result)
            print("==================\n")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result, ensure_ascii=True),
                }
            )

    print("\n=== RAW RESPONSE ===")
    print(response)
    print("====================\n")


    return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
