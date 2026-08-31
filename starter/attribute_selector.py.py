"""
Room for improvement
--------------------
Attribute definitions; what should be classified under style, use-case, feature, etc.?

Price bucketing; what thresholds for budget?


Inputs
------
known attributes:   dict of slot -> value for attributes already pinned down.
                    Keys and values are NOT assumed to be pre-normalized -- they
                    may come from upstream extraction/UI code with inconsistent
                    casing or phrasing, e.g. {"Color": "blue", "Size": "Large"}
                    instead of {"color": "blue", "size": "L"}. `normalize_known`
                    (called automatically by `choose_attribute`) handles this.
candidates:         up to ~300 catalog product dicts (ideally the pool that
                    already satisfies `known`, e.g. output of the retrieval
                    stage; the selection logic itself works on any list of
                    product dicts).

Output
--------------------------------------------------------------------------
Attribute:          Provide agent with attribute that results in maximum information gain
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Optional

from starter.ClassifyIntent import get_shared_classifier

_CLASSIFIER = get_shared_classifier()

SLOT_ATTRIBUTES = [
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case",
]

# Dataset profile tags are broader than the structured slots exposed by the
# agent. Keep the mapping here so profile-aware prioritization has one clear
# normalization point. The order of preference_tags is preserved later and is
# treated as the user's preference priority.
_PROFILE_TAG_TO_SLOT = {
    "category": "category",
    "material": "material",
    "color": "color",
    "size": "size",
    "fit": "style",
    "style": "style",
    "brand": "brand",
    "price": "budget",
    "budget": "budget",
    "comfort": "feature",
    "durability": "feature",
    "warmth": "feature",
    "weather": "feature",
    "performance": "feature",
    "occasion": "use_case",
    "use_case": "use_case",
}

# Normalize known slot names and values before comparing them.

# Alternate spellings and synonyms for slot names. Keys are normalized before
# lookup by _normalize_key.
_KEY_SYNONYMS = {
    "category": "category", "cat": "category", "type": "category",
    "product_type": "category",
    "material": "material", "materials": "material", "fabric": "material",
    "color": "color", "colour": "color", "colors": "color", "colours": "color",
    "size": "size", "sz": "size",
    "style": "style", "fit": "style",
    "brand": "brand", "make": "brand",
    "budget": "budget", "price": "budget", "price_range": "budget",
    "feature": "feature", "features": "feature",
    "use_case": "use_case", "usecase": "use_case", "use": "use_case",
    "occasion": "use_case",
}

# Map common size spellings onto a canonical scale.
_SIZE_NORMALIZE = {
    "xxs": "XXS", "extra extra small": "XXS",
    "xs": "XS", "extra small": "XS", "x-small": "XS", "x small": "XS",
    "s": "S", "small": "S",
    "m": "M", "med": "M", "medium": "M",
    "l": "L", "large": "L",
    "xl": "XL", "x-large": "XL", "x large": "XL", "extra large": "XL",
    "xxl": "XXL", "xx-large": "XXL", "xx large": "XXL", "2xl": "XXL",
    "xxxl": "XXXL", "xxx-large": "XXXL", "3xl": "XXXL",
}


#Normalization and cleaning process (below) was made under assumption that sessionState data is messy
#Certain processes are not necessary

# Normalize raw key into canonical slot name.
def _normalize_key(raw_key: str) -> Optional[str]:
    cleaned = re.sub(r"[\s\-]+", "_", raw_key.strip().lower())
    if cleaned in _KEY_SYNONYMS:
        return _KEY_SYNONYMS[cleaned]
    if cleaned in SLOT_ATTRIBUTES or cleaned == "other":
        return cleaned
    return None  # unrecognized attribute name; caller ignores rather than guesses


# Normalize raw size string value.
def _normalize_size_value(raw_value: str) -> str:
    cleaned = str(raw_value).strip().lower()
    if cleaned in _SIZE_NORMALIZE:
        return _SIZE_NORMALIZE[cleaned]
    # Numeric sizes (such as shoe or children's sizes) are kept as written.
    return re.sub(r"\s+", " ", cleaned).upper() if len(cleaned) <= 4 else cleaned


# Normalize keys and values in the known dict.
def normalize_known(known: dict[str, Any]) -> dict[str, str]:
    """
    Map a possibly messy `known` dict (inconsistent key casing/spelling,
    unnormalized values like "Large" vs "L" vs "large") onto canonical
    slot names with normalized values. Unrecognized keys and empty
    values are dropped.
    """
    normalized: dict[str, str] = {}
    for raw_key, raw_value in known.items():
        if not raw_value:  # covers None, "", [], 0, False
            continue
        slot = _normalize_key(str(raw_key))
        if slot is None:
            continue

        if isinstance(raw_value, list):
            value = ", ".join(str(item).strip() for item in raw_value if str(item).strip())
            if not value:
                continue
            normalized[slot] = value.lower()
            continue

        if slot == "budget" and isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool):
            normalized[slot] = f"${int(raw_value)}"
            continue

        value = str(raw_value).strip()
        if not value:
            continue
        normalized[slot] = _normalize_size_value(value) if slot == "size" else value.lower()
    return normalized


# Extract possible values for each unknown slot from the candidate pool.

# The classifier exposes style and use-case vocabulary through related lists.
_SLOT_VOCAB: dict[str, list] = {
    "category": [_CLASSIFIER.categories],
    "material": [_CLASSIFIER.materials],
    "color": [_CLASSIFIER.colors],
    "brand": [_CLASSIFIER.brands],
    "style": [_CLASSIFIER.fits, _CLASSIFIER.patterns],
    "use_case": [_CLASSIFIER.style],
    "feature": [_CLASSIFIER.features],
}

_SIZE_TOKEN_RE = re.compile(
    r"\b(xxs|xs|s|m|l|xl|xxl|xxxl|small|medium|large|x-large|xx-large)\b",
    re.IGNORECASE,
)

# Cache flattened product text and extracted values across turns.
_TEXT_CACHE: dict[str, str] = {}
_VALUE_CACHE: dict[tuple[str, str], list[str]] = {}


# Return candidate product lookup key.
def _product_key(product: dict) -> str:
    return str(product.get("parent_asin") or id(product))


# Bucket ceilings for the budget slot in USD; the final bucket is open-ended.
_PRICE_EDGES = [15, 30, 50, 100]
_PRICE_RE = re.compile(r"(\d+(?:\.\d+)?)")

# Parse a catalog price into a float when possible.
def _parse_price(raw: Any) -> Optional[float]:
    """Best-effort float parse. Catalog prices show up as None, a float,
    or strings like 'from 21.30' or '—'."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    match = _PRICE_RE.search(str(raw))
    return float(match.group(1)) if match else None

# Convert a price into the corresponding budget bucket.
def _price_bucket(price: float) -> str:
    for edge in _PRICE_EDGES:
        if price < edge:
            return f"under ${edge}"
    return f"${_PRICE_EDGES[-1]}+"


# Flatten product fields into a lowercase string.
def _product_text(product: dict) -> str:
    """Flatten a catalog product's text fields into one lowercase blob."""
    key = _product_key(product)
    cached = _TEXT_CACHE.get(key)
    if cached is not None:
        return cached

    parts: list[str] = []
    for field in ("title", "features", "description", "categories", "store"):
        value = product.get(field)
        if isinstance(value, list):
            parts.extend(str(v) for v in value)
        elif value is not None:
            parts.append(str(value))
    details = product.get("details")
    if isinstance(details, dict):
        parts.extend(f"{k} {v}" for k, v in details.items())
    result = " ".join(parts).lower()
    _TEXT_CACHE[key] = result
    return result

# Extract values exposed by a product for a given slot.
def _values_for_slot(slot: str, product: dict, text: str) -> list[str]:
    """Every distinct (normalized) value this product exposes for `slot`."""
    key = (_product_key(product), slot)
    cached = _VALUE_CACHE.get(key)
    if cached is not None:
        return cached

    if slot == "budget":
        price = _parse_price(product.get("price"))
        result = [] if price is None else [_price_bucket(price)]
        _VALUE_CACHE[key] = result
        return result
    if slot == "size":
        tokens = {m.group(1).lower() for m in _SIZE_TOKEN_RE.finditer(text)}
        result = sorted({_normalize_size_value(t) for t in tokens})
        _VALUE_CACHE[key] = result
        return result
    found: set[str] = set()
    for vocab in _SLOT_VOCAB.get(slot, []):
        for word in vocab:
            # Substring checks keep repeated candidate scans inexpensive.
            # These values only guide question selection; final ranking uses
            # the agent's more precise matcher.
            if word in text:
                found.add(word)
    result = sorted(found)
    _VALUE_CACHE[key] = result
    return result


# Score unknown slots by how effectively they split the candidate pool.

# Pick diverse values from value counts to show in prompt.
def _pick_diverse_values(counts: Counter, k: int = 4) -> list[str]:
    """
    Choose up to `k` values to surface in the clarification prompt.
    """
    ranked = [v for v, _ in counts.most_common()]
    if len(ranked) <= k:
        return ranked

    picks = [ranked[0]]
    tail = ranked[1:]
    remaining_slots = k - 1
    if remaining_slots <= 0:
        return picks

    step = len(tail) / remaining_slots
    seen = set(picks)
    for i in range(remaining_slots):
        idx = min(int(i * step), len(tail) - 1)
        candidate = tail[idx]
        if candidate in seen:
            candidate = next((alt for alt in tail[idx:] if alt not in seen), candidate)
        seen.add(candidate)
        picks.append(candidate)
    return picks


@dataclass
class SlotSuggestion:
    attribute: str
    top_values: list[str]
    coverage: float             # fraction of candidates exposing a value
    expected_remaining: float   # lower = better narrowing
    candidate_count: int


# Compute split score and return SlotSuggestion for a slot.
def _score_slot(slot: str, candidates: list[dict]) -> Optional[SlotSuggestion]:
    n = len(candidates)
    if n == 0:
        return None
    counts: Counter[str] = Counter()
    covered = 0
    for product in candidates:
        text = _product_text(product)
        values = _values_for_slot(slot, product, text)
        if values:
            covered += 1
            counts.update(values)
    if not counts:
        return None  # nothing in the pool exposes this attribute at all
    total_votes = sum(counts.values())
    # Use the full distribution for scoring; displayed options only affect the
    # prompt and must not change which attribute wins.
    expected_remaining = sum(c * c for c in counts.values()) / total_votes
    return SlotSuggestion(
        attribute=slot,
        top_values=_pick_diverse_values(counts, k=4),
        coverage=covered / n,
        expected_remaining=expected_remaining,
        candidate_count=n,
    )


# Convert a SessionState snapshot into the `known` shape used by this module.
def known_from_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """
    Adapter for SessionState.snapshot()'s exact return shape (see
    session_state.py):
        {"slots": {...currently filled slots...}, "unspecified": [...]}
    """
    known = dict(snapshot.get("slots") or {})
    for name in snapshot.get("unspecified") or []:
        known.setdefault(name, "no preference")
    return known


# Extract priority slots based on user profile preference tags.
def _profile_priority(user_profile: dict[str, Any] | None) -> list[str]:
    """Map ordered dataset preference tags to unique selector slots."""
    if not user_profile:
        return []

    raw_tags = user_profile.get("preference_tags", [])
    if isinstance(raw_tags, str):
        raw_tags = [raw_tags]
    if not isinstance(raw_tags, (list, tuple, set)):
        return []

    priority: list[str] = []
    for raw_tag in raw_tags:
        tag = re.sub(r"[\s\-]+", "_", str(raw_tag).strip().lower())
        slot = _PROFILE_TAG_TO_SLOT.get(tag)
        if slot and slot not in priority:
            priority.append(slot)
    return priority

# Return the most useful unknown attribute.
def choose_attribute(
    known: dict[str, Any],
    candidates: list[dict],
    min_coverage: float = 0.15,
    asked: set[str] | None = None,
    user_profile: dict[str, Any] | None = None,
) -> Optional[SlotSuggestion]:
    """
    Pick the unknown slot whose values best split `candidates`.

    `asked`: used to check if attribute has been asked

    Returns a synthetic attribute="other" SlotSuggestion (never None) when
    every slot is already known/asked, or when none of the remaining
    unknown slots have a detectable value anywhere in `candidates`.
    """
    known_norm = normalize_known(known)
    already_asked = asked or set()
    unknown_slots = [
        s for s in SLOT_ATTRIBUTES
        if not known_norm.get(s) and s not in already_asked
    ]

    scored = [s for s in (_score_slot(slot, candidates) for slot in unknown_slots) if s]
    qualified = [s for s in scored if s.coverage >= min_coverage]
    pool = qualified or scored
    if not pool:
        # If no measurable slot remains, return a generic prompt so callers
        # can continue without a special case.
        return SlotSuggestion(
            attribute="other",
            top_values=[],
            coverage=0.0,
            expected_remaining=float(len(candidates)),
            candidate_count=len(candidates),
        )

    # Profile tags are an explicit personalization signal. Prefer the first
    # mapped tag that is still unknown/unused and has enough candidate
    # coverage; otherwise retain the original narrowing heuristic below.
    qualified_by_slot = {s.attribute: s for s in qualified}
    for preferred_slot in _profile_priority(user_profile):
        preferred = qualified_by_slot.get(preferred_slot)
        if preferred is not None:
            return preferred

    # Prefer stronger narrowing, then prefer attributes represented more often.
    pool.sort(key=lambda s: (s.expected_remaining, -s.coverage))
    return pool[0]


_PROMPT_TEMPLATES = {
    "category": "What type of {noun} are you looking for — {options}, or something else?",
    "material": "Do you have a material preference — {options}, or something else?",
    "color": "What color are you thinking — {options}, or another color?",
    "size": "What size do you need?",
    "style": "What style fits best — {options}, or something else?",
    "brand": "Any brand you'd like — {options}, or no preference?",
    "budget": "What's your budget?",
    "feature": "Any must-have features — {options}, or something else?",
    "use_case": "What's it for — {options}, or something else?",
    "other": "Is there anything else about what you're looking for that I should know?",
}

# Render a natural-language prompt for the selected attribute.
def generate_prompt(suggestion: SlotSuggestion, noun: str = "item") -> str:
    options = ", ".join(suggestion.top_values[:3])
    template = _PROMPT_TEMPLATES.get(suggestion.attribute, "Any preference for {options}?")
    return template.format(options=options, noun=noun)


# Generate top-level proactive guidance payload.
def propose_clarification(
    known: dict[str, Any],
    candidates: list[dict],
    asked: set[str] | None = None,
) -> dict:
    """
    Top-level entry point for one turn of proactive guidance.

    Returns: {"ask_attribute": <slot or None>, "message": <prompt text or None>}
    -- shaped to drop straight into the turn_response contract.
    """
    suggestion = choose_attribute(known, candidates, asked=asked)
    if suggestion is None:
        return {"ask_attribute": None, "message": None}
    known_norm = normalize_known(known)
    noun = known_norm.get("category") or "item"
    return {
        "ask_attribute": suggestion.attribute,
        "message": generate_prompt(suggestion, noun=noun),
    }