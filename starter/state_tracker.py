from __future__ import annotations

import json
import os
import re
from typing import Any, Callable

from starter.ClassifyIntent import get_shared_classifier
from starter.session_state import ALLOWED_SLOTS, LIST_SLOTS, SessionState
from dotenv import load_dotenv

load_dotenv()

_CLASSIFIER = get_shared_classifier()
HIGH_CONFIDENCE = 0.7
AUTO_SLOTS = frozenset(
    {
        "category", "material", "color", "size", "style", "brand", "budget",
        "feature", "use_case", "other",
    }
)
TOKEN_RE = re.compile(r"[a-z0-9$]+", re.IGNORECASE)
RESIDUAL_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "need", "get", "find", "show", "like", "id", "im", "under", "over", "than",
    "less", "more", "below", "above", "between", "bucks", "dollars", "usd",
    "size", "sz", "wanting", "looking", "please", "can", "could", "a", "am",
    # Ignore prompt boilerplate that carries no constraint content and should
    # not trigger an unnecessary LLM call.
    "don", "dont", "doesn", "doesnt", "have", "additional", "preference",
    "preferences", "key", "requirement", "requirements", "matters", "what",
    "there", "no", "none", "any", "sure", "just", "well", "so", "also",
    "actually", "ignore", "earlier", "instead", "rather", "prefer",
    # Slot names describe the requested attribute; they are not residual text.
    "category", "color", "material", "size", "style", "brand", "budget",
    "feature", "use_case", "other", "use", "case",
}
MESSY_RE = re.compile(
    r"\b(not|without|don't|dont|do not|doesn't|doesnt|remove|forget|instead|"
    r"no longer|no preference|don't care|dont care|anything|either)\b",
    re.IGNORECASE,
)
PIVOT_RE = re.compile(
    r"\b(actually looking for|switch to|something else)\b",
    re.IGNORECASE,
)
ATTRIBUTE_CHANGE_RE = re.compile(
    r"\b(change|changed|changing|update|updated|replace|replaced|instead|"
    r"ignore|forget|remove|switch|different|no longer|rather than)\b",
    re.IGNORECASE,
)
MAX_SIMPLE_TOKENS = 14

SLOT_ENUM = sorted(ALLOWED_SLOTS)

# These slots accept descriptive text. Budget and size are excluded because
# numeric or short-token values are valid for them.
TEXTUAL_SLOTS = frozenset(
    {"category", "material", "color", "style", "brand", "feature", "use_case", "other"}
)
NUMERIC_ONLY_RE = re.compile(r"^\$?\d+(\.\d+)?$")

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
                "use update_slot to rewrite. Always pass the actual descriptive text the user used "
                "(e.g. 'alloy', 'stainless steel') -- never a bare number or placeholder id, even for "
                "budget (pass it as a numeric string like '100')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "enum": SLOT_ENUM},
                    "value": {"type": "string"},
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
            "description": (
                "Intent override: overwrite a slot (for example color red -> blue). Always pass the "
                "actual descriptive text the user used -- never a bare number or placeholder id."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "enum": SLOT_ENUM},
                    "value": {"type": "string"},
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
                    "value": {"type": ["string", "null"]},
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
When this tool loop is called, parse the complete latest user message into
atomic requirement components before using tools. The classifier output is
only a hint; do not limit parsing to its low-confidence results. Do not invent
constraints the user did not state.
Workflow:
1. Read the entire latest user message.
2. Split it into each product/category, attribute, and preference component.
3. Map every component to the best supported slot and apply all valid components.
4. Use the user's own descriptive wording for slot values.
Rules:
- Do not re-set slots listed as already applied unless the user is clearly correcting them.
- New information that does not conflict: set_slot (accumulation). Prior slots must remain.
- Contradiction, replacement, or "instead": update_slot (rewrite) or clear_slot then set_slot.
- "Forget X" / "ignore my earlier preference": clear_slot or update_slot for only the obsolete keys.
- Completely different product type: reset_slots, then set_slot for the new search.
- "I don't have a preference for {attribute}": mark_unspecified.
- Unmapped classifier "others" (gender, fit, pattern, condition) may belong in style, use_case, or other.
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


def _values_for_strip(value: Any) -> list[str]:
    items = value if isinstance(value, list) else [value]
    phrases: list[str] = []
    for item in items:
        text = str(item).strip().lower()
        if not text:
            continue
        phrases.append(text)
        phrases.extend(TOKEN_RE.findall(text))
        digits = re.findall(r"\d+", text)
        phrases.extend(digits)
    return phrases


def _residual_intent(user_message: str, scored: dict[str, dict[str, Any]]) -> bool:
    remaining = user_message.lower()
    phrases: list[str] = []
    for entry in scored.values():
        phrases.extend(_values_for_strip(entry.get("value")))
    phrases = sorted(set(phrases), key=len, reverse=True)
    for phrase in phrases:
        if not phrase:
            continue
        remaining = re.sub(rf"\b{re.escape(phrase)}\b", " ", remaining)
        remaining = remaining.replace(phrase, " ")
    leftover = [
        token
        for token in TOKEN_RE.findall(remaining)
        if len(token) > 1 and token.lower() not in RESIDUAL_STOPWORDS and not token.isdigit()
    ]
    return bool(leftover)


def _is_first_turn(session: SessionState, turn: int) -> bool:
    """True when this is the first user turn of a session."""
    return turn == 1 or not session.history


def _low_confidence_slots(
    scored: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Return only classifier results that need LLM/tool interpretation."""
    return {
        name: entry
        for name, entry in scored.items()
        if float(entry.get("confidence") or 0.0) < HIGH_CONFIDENCE
    }


def _needs_llm(user_message: str, scored: dict[str, dict[str, Any]]) -> bool:
    """Use the LLM only for long messages or explicit attribute changes.

    Short, ordinary requests are handled by the classifier and their
    unmatched words are retained as free-form retrieval terms.  ``scored``
    remains in the signature so the routing function can be called without
    changing the existing tracker interface.
    """
    del scored
    token_count = len(TOKEN_RE.findall(user_message))
    return token_count > MAX_SIMPLE_TOKENS or bool(
        ATTRIBUTE_CHANGE_RE.search(user_message)
    )


def _remember_freeform_terms(
    session: SessionState,
    user_message: str,
    scored: dict[str, dict[str, Any]],
) -> None:
    """Keep useful unmatched terms for later retrieval without duplicating
    the existing structured-slot extraction."""
    remaining = user_message.lower()
    phrases: list[str] = []
    for entry in scored.values():
        phrases.extend(_values_for_strip(entry.get("value")))
    for phrase in sorted(set(phrases), key=len, reverse=True):
        remaining = re.sub(rf"\b{re.escape(phrase)}\b", " ", remaining)
        remaining = remaining.replace(phrase, " ")

    for token in TOKEN_RE.findall(remaining):
        token = token.lower()
        if (
            len(token) > 1
            and not token.isdigit()
            and token not in RESIDUAL_STOPWORDS
            and token not in session.freeform_terms
        ):
            session.freeform_terms.append(token)

    # Keep the retrieval query bounded as the conversation grows.
    del session.freeform_terms[:-60]


def _apply_high_confidence(
    session: SessionState,
    scored: dict[str, dict[str, Any]],
    turn: int,
) -> list[str]:
    """Apply high-confidence classifier results locally, without the LLM."""
    applied: list[str] = []

    for name, entry in scored.items():
        if name not in AUTO_SLOTS:
            continue
        if float(entry.get("confidence") or 0) < HIGH_CONFIDENCE:
            continue

        value = entry.get("value")
        values = value if name in LIST_SLOTS and isinstance(value, list) else [value]

        for item in values:
            current = session.slots.get(name)

            if name in LIST_SLOTS or current in (None, "", []):
                result = dispatch(
                    session,
                    "set_slot",
                    {"name": name, "value": item},
                    turn,
                )
            elif str(current).lower() == str(item).lower() or (
                name == "budget"
                and isinstance(current, int)
                and str(current) in str(item)
            ):
                applied.append(name)
                continue
            else:
                # A high-confidence conflict can be resolved locally.
                result = dispatch(
                    session,
                    "update_slot",
                    {"name": name, "value": item},
                    turn,
                )

            if result.get("ok"):
                applied.append(name)

    return list(dict.fromkeys(applied))


def _rejection_reason(slot_name: str, value: Any, user_message: str) -> str | None:
    """Sanity-check an LLM-proposed set_slot/update_slot value before it is trusted.

    Guards against two observed tool-calling failure modes on the free-text
    slots (material, feature, category, ...): the model emitting a bare
    digit/placeholder id instead of the real text (e.g. "81" instead of
    "alloy"), and the model inventing a value with no basis in what the user
    actually said this turn. budget/size are exempt since numeric values are
    legitimate there.
    """
    if slot_name not in TEXTUAL_SLOTS:
        return None

    items = value if isinstance(value, list) else [value]
    haystack = user_message.lower()

    for item in items:
        if isinstance(item, bool):
            continue
        if isinstance(item, (int, float)):
            return f"{slot_name} must be descriptive text, not a bare number ({item!r})"

        text = str(item).strip()
        if not text:
            continue
        if NUMERIC_ONLY_RE.match(text):
            return f"{slot_name} must be descriptive text, not a bare number ({text!r})"

        words = [w for w in TOKEN_RE.findall(text.lower()) if len(w) > 2]
        if words:
            hits = sum(1 for word in words if word in haystack)
            if hits / len(words) < 0.5:
                return (
                    f"{slot_name} value {text!r} doesn't appear in the user's message; "
                    "use the user's own wording"
                )
    return None


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
    """Classifier-first state tracking with a simple LLM complexity gate.

    The rule-based IntentClassifier always runs first. Any slot it extracted
    at >= HIGH_CONFIDENCE is applied locally via _apply_high_confidence --
    no LLM call. For a long or explicit-change message, the complete user
    message is sent to the LLM, which decomposes it into atomic components
    and applies each component to the appropriate slot.
    """

    scored = _CLASSIFIER.extract_constraints_scored(user_message)
    low_confidence = _low_confidence_slots(scored)
    applied = _apply_high_confidence(session, scored, turn)
    _remember_freeform_terms(session, user_message, scored)
    pending_attribute = getattr(session, "pending_attribute", None)

    # Short statements without explicit change language stay local.
    print(user_message)
    if not _needs_llm(user_message, scored):
        print("Not Using LLM")
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "llm_called": False,
            "classifier": scored,
            "low_confidence": {},
            "applied": applied,
        }
    print("Using LLM")
    # Otherwise, send the complete complex/change message through the tool loop.
    prompt_tokens = 0
    completion_tokens = 0
    completer = create
    if completer is None:
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not api_key:
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "llm_called": False,
                "classifier": scored,
                "low_confidence": low_confidence,
                "applied": applied,
                "reason": "No API KEY FOUND",
            }
        if client is None:
            try:
                from openai import OpenAI
            except ImportError:
                return {"prompt_tokens": 0, "completion_tokens": 0}
            client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
        completer = client.chat.completions.create

    chosen_model = model or os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    snapshot = json.dumps(session.snapshot(), ensure_ascii=True)
    classifier_json = json.dumps(scored, ensure_ascii=True)
    applied_json = json.dumps(applied, ensure_ascii=True)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"Current slots: {snapshot}"},
        {
            "role": "system",
            "content": (
                "Decompose the full latest user message before acting. "
                f"Classifier hints (value/confidence/source; not complete): {classifier_json}. "
                f"Already applied locally: {applied_json}. "
                "Apply every valid component from the message, including components already hinted "
                "by the classifier when they are not yet present in state."
            ),
        },
    ]
    if pending_attribute:
        messages.append(
            {
                "role": "system",
                "content": (
                    f"You most recently asked the user about '{pending_attribute}'. "
                    "If the message doesn't clearly state a value for a different slot, "
                    "treat it as the user's answer to that attribute: call set_slot for "
                    f"'{pending_attribute}' if they gave a value (use their own wording), "
                    f"or mark_unspecified for '{pending_attribute}' if they said they have "
                    "no preference. Don't drop a real answer just because it doesn't match "
                    "a known vocabulary term."
                ),
            }
        )
    if session.last_assistant_message:
        messages.append({"role": "assistant", "content": session.last_assistant_message})
    messages.append({"role": "user", "content": user_message})

    for _ in range(max_rounds):
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

            call_args = _parse_arguments(call.function.arguments)

            rejection = None
            if call.function.name in ("set_slot", "update_slot"):
                rejection = _rejection_reason(
                    call_args.get("name", ""), call_args.get("value"), user_message
                )

            if rejection:
                result = {"ok": False, "error": rejection, **session.snapshot()}
            else:
                result = dispatch(session, call.function.name, call_args, turn)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result, ensure_ascii=True),
                }
            )

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "llm_called": True,
        "classifier": scored,
        "low_confidence": low_confidence,
        "applied": applied,
    }
