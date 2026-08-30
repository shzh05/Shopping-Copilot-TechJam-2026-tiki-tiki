"""
(Handwritten by me)
Proactive Guidance: pick the single most useful unknown attribute to ask
about, given what we already know and the current candidate pool.

Fits into pillar II of the challenge ("Trigger an immediate retrieval
cutoff when facing Over-Generality ... to actively generate structured,
proactive clarification prompts that guide user convergence").

--------------------------------------------------------------------------
Inputs
--------------------------------------------------------------------------
known:      dict of slot -> value for attributes already pinned down.
            Keys and values are NOT assumed to be pre-normalized -- they
            may come from upstream extraction/UI code with inconsistent
            casing or phrasing, e.g. {"Color": "blue", "Size": "Large"}
            instead of {"color": "blue", "size": "L"}. `normalize_known`
            (called automatically by `choose_attribute`) handles this.
candidates: up to ~300 catalog product dicts (ideally the pool that
            already satisfies `known`, e.g. output of the retrieval
            stage; the selection logic itself works on any list of
            product dicts).

Slots handled (must match agent_api_contract.json's ask_attribute enum,
minus "other"):
    category, material, color, size, style, brand, budget, feature, use_case

--------------------------------------------------------------------------
Approach
--------------------------------------------------------------------------
1. Normalize `known`'s keys (case/spelling variants -> canonical slot
   names) and values (lowercased; sizes mapped onto a single S/M/L/XL...
   scale) so a slot that's already pinned down isn't mistaken for unknown.
2. For every remaining unknown slot, scan each candidate's text (title,
   features, description, categories, details, store) and pull out every
   value that slot could take, reusing ClassifyIntent.IntentClassifier's
   vocab so extraction stays consistent with the rest of the pipeline.
   `budget` is handled separately via numeric price bucketing; `size` via
   a regex over common size tokens, normalized the same way as `known`.
3. For each slot, build a histogram of values across the pool and score it
   by the *expected number of candidates left* if we ask the question and
   the user's answer lands on one of the observed values with probability
   proportional to how common that value already is in the pool:

        E[remaining] = sum(count_v ** 2) / total_votes

   This is minimized by splits that are both even (no single dominant
   value) and fine-grained (many distinct values) -- exactly the property
   we want from a clarifying question. Slots where almost nothing in the
   pool exposes a value (low `coverage`) are penalized/skipped, since
   asking about them is likely to produce "no preference" dead ends.
4. The slot with the lowest expected remaining count wins. We surface its
   most common concrete values (up to 3) in a natural-language question.

This is a heuristic, not true expected information gain over the *true*
hidden user preference -- we don't know that distribution, so we use the
pool's own distribution as the best available proxy.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Optional

from ClassifyIntent import IntentClassifier

_CLASSIFIER = IntentClassifier()

SLOT_ATTRIBUTES = [
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case",
]

# --------------------------------------------------------------------------
# Normalization of `known` (and, for size, of extracted candidate values)
# --------------------------------------------------------------------------

# Alternate spellings/casings/synonyms for each slot's key. Keys here are
# already lowercased/underscored before lookup (see _normalize_key).
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

# Every common way a size might be written, mapped onto one canonical scale.
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


def _normalize_key(raw_key: str) -> Optional[str]:
    cleaned = re.sub(r"[\s\-]+", "_", raw_key.strip().lower())
    if cleaned in _KEY_SYNONYMS:
        return _KEY_SYNONYMS[cleaned]
    if cleaned in SLOT_ATTRIBUTES or cleaned == "other":
        return cleaned
    return None  # unrecognized attribute name; caller ignores rather than guesses


def _normalize_size_value(raw_value: str) -> str:
    cleaned = str(raw_value).strip().lower()
    if cleaned in _SIZE_NORMALIZE:
        return _SIZE_NORMALIZE[cleaned]
    # numeric sizes (shoes, kids, etc.) -- keep as-is, just tidy whitespace
    return re.sub(r"\s+", " ", cleaned).upper() if len(cleaned) <= 4 else cleaned


def normalize_known(known: dict[str, Any]) -> dict[str, str]:
    """
    Map a possibly messy `known` dict (inconsistent key casing/spelling,
    unnormalized values like "Large" vs "L" vs "large") onto canonical
    slot names with normalized values. Unrecognized keys and empty
    values are dropped.

    Compatible with SessionState.slots (session_state.py) as-is:
    - Falsy values (None, "", [], 0) are treated as "not yet known",
      matching SCALAR_SLOTS defaulting to None and LIST_SLOTS
      defaulting to [] there.
    - List-valued slots (feature/use_case/other) are joined into a
      readable comma-separated string rather than stringified as a
      raw Python list.
    - A numeric budget (SessionState coerces budget to an int max
      price) is rendered as "$<amount>".
    - The "other" slot isn't one of the 9 attributes this module can
      ask about or extract from the catalog, so it's accepted as a
      known-slot signal (a non-empty "other" won't be mistaken for
      unknown) but never becomes something `choose_attribute` suggests.
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


# --------------------------------------------------------------------------
# Candidate-pool value extraction
# --------------------------------------------------------------------------

# `style` and `use_case` don't exist as single lists in IntentClassifier,
# so each combines two related lists.
_SLOT_VOCAB: dict[str, list] = {
    "category": [_CLASSIFIER.categories],
    "material": [_CLASSIFIER.materials],
    "color": [_CLASSIFIER.colors],
    "brand": [_CLASSIFIER.brands],
    "style": [_CLASSIFIER.fits, _CLASSIFIER.patterns],
    "use_case": [_CLASSIFIER.occasions],
    "feature": [_CLASSIFIER.features],
}

_SIZE_TOKEN_RE = re.compile(
    r"\b(xxs|xs|s|m|l|xl|xxl|xxxl|small|medium|large|x-large|xx-large)\b",
    re.IGNORECASE,
)

# Bucket ceilings in USD; the final bucket is open-ended ("$100+").
_PRICE_EDGES = [15, 30, 50, 100]
_PRICE_RE = re.compile(r"(\d+(?:\.\d+)?)")


def _parse_price(raw: Any) -> Optional[float]:
    """Best-effort float parse. Catalog prices show up as None, a float,
    or strings like 'from 21.30' or '—'."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    match = _PRICE_RE.search(str(raw))
    return float(match.group(1)) if match else None


def _price_bucket(price: float) -> str:
    for edge in _PRICE_EDGES:
        if price < edge:
            return f"under ${edge}"
    return f"${_PRICE_EDGES[-1]}+"


def _product_text(product: dict) -> str:
    """Flatten a catalog product's text fields into one lowercase blob."""
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
    return " ".join(parts).lower()


def _values_for_slot(slot: str, product: dict, text: str) -> list[str]:
    """Every distinct (normalized) value this product exposes for `slot`."""
    if slot == "budget":
        price = _parse_price(product.get("price"))
        return [] if price is None else [_price_bucket(price)]
    if slot == "size":
        tokens = {m.group(1).lower() for m in _SIZE_TOKEN_RE.finditer(text)}
        return sorted({_normalize_size_value(t) for t in tokens})
    found: set[str] = set()
    for vocab in _SLOT_VOCAB.get(slot, []):
        for word in vocab:
            if re.search(rf"\b{re.escape(word)}\b", text):
                found.add(word)
    return sorted(found)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def _pick_diverse_values(counts: Counter, k: int = 4) -> list[str]:
    """
    Choose up to `k` values to surface in the clarification prompt.

    Pure `most_common(k)` tends to hand back near-duplicates of the
    catalog's own dominant values (e.g. "solid, printed, floral" might
    all be minority patterns dwarfed by "solid"), which quietly steers
    the user toward whatever's already common rather than letting them
    say what they actually want. To reduce that anchoring effect, we
    always keep the single most common value (it's the most likely
    match, so it's still useful to show), then spread the remaining
    slots evenly across the *rest* of the distribution -- including
    less-common values -- rather than taking the next-most-common ones
    in a row. This is deterministic (no randomness) but surfaces more
    of the real variety present in the pool.
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
    # Note: narrowing power (expected_remaining) is still computed from the
    # FULL value distribution, not just the values we choose to show --
    # diversifying the displayed options never affects which attribute wins.
    expected_remaining = sum(c * c for c in counts.values()) / total_votes
    return SlotSuggestion(
        attribute=slot,
        top_values=_pick_diverse_values(counts, k=4),
        coverage=covered / n,
        expected_remaining=expected_remaining,
        candidate_count=n,
    )


def known_from_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """
    Adapter for SessionState.snapshot()'s exact return shape (see
    session_state.py):
        {"slots": {...currently filled slots...}, "unspecified": [...]}

    `unspecified` entries (the user explicitly said "no preference") are
    folded into the returned dict as known too, so `choose_attribute`
    doesn't mistake "asked and declined" for "never asked" and suggest
    asking about it again. `normalize_known` handles the rest (casing,
    list-slot joining, numeric budget, etc.) once this is passed in.
    """
    known = dict(snapshot.get("slots") or {})
    for name in snapshot.get("unspecified") or []:
        known.setdefault(name, "no preference")
    return known


def choose_attribute(
    known: dict[str, Any],
    candidates: list[dict],
    min_coverage: float = 0.15,
) -> Optional[SlotSuggestion]:
    """
    Pick the unknown slot whose values best split `candidates`.

    `known` may use inconsistent key casing/spelling and unnormalized
    values (e.g. {"Color": "blue", "Size": "Large"}); it is normalized
    internally via `normalize_known` before anything is compared.
    Slots where fewer than `min_coverage` of candidates expose a value are
    deprioritized (but used as a fallback if nothing else qualifies, so we
    still ask *something* rather than nothing).
    """
    known_norm = normalize_known(known)
    unknown_slots = [s for s in SLOT_ATTRIBUTES if not known_norm.get(s)]

    scored = [s for s in (_score_slot(slot, candidates) for slot in unknown_slots) if s]
    qualified = [s for s in scored if s.coverage >= min_coverage]
    pool = qualified or scored
    if not pool:
        return None

    # Lower expected_remaining wins; break ties by preferring higher coverage
    # (a question more of the pool can actually answer).
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
}


def generate_prompt(suggestion: SlotSuggestion, noun: str = "item") -> str:
    options = ", ".join(suggestion.top_values[:3])
    template = _PROMPT_TEMPLATES.get(suggestion.attribute, "Any preference for {options}?")
    return template.format(options=options, noun=noun)


def propose_clarification(known: dict[str, Any], candidates: list[dict]) -> dict:
    """
    Top-level entry point for one turn of proactive guidance.

    Returns: {"ask_attribute": <slot or None>, "message": <prompt text or None>}
    -- shaped to drop straight into the turn_response contract.
    """
    suggestion = choose_attribute(known, candidates)
    if suggestion is None:
        return {"ask_attribute": None, "message": None}
    known_norm = normalize_known(known)
    noun = known_norm.get("category") or "item"
    return {
        "ask_attribute": suggestion.attribute,
        "message": generate_prompt(suggestion, noun=noun),
    }
