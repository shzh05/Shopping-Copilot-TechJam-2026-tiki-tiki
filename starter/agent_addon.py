from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from starter.session_state import SessionState
from starter.state_tracker import track
from starter.attribute_selector import choose_attribute, known_from_snapshot


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


class Agent:
    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, SessionState] = {}
        self._prices: dict[str, float | None] = {}
        self._products: dict[str, dict] = {}
        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                parent_asin = str(product["parent_asin"])
                self._products[parent_asin] = product
                price = product.get("price")
                self._prices[parent_asin] = float(price) if isinstance(price, (int, float)) else None
                batch.append(
                    (
                        parent_asin,
                        _text(product.get("title")),
                        _text(product.get("categories")),
                        _text(product.get("features")),
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = SessionState(session_id, user_profile)

    # Function that uses bm25 to search and sort items
    def _bm25_search(self, terms: list[str], limit: int) -> list[str]:
        unique_terms = list(dict.fromkeys(terms))[:40]

        if not unique_terms:
            return []

        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        rows = self.connection.execute(
        """
        SELECT parent_asin
        FROM products
        WHERE products MATCH ?
        ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?
        """,
        (expression, limit)
        ).fetchall()

        return [str(row[0]) for row in rows]

    # Function to add Reciprocal Rank Fusion
    def _add_rrf(self, scores: dict[str, float], results: list[str], weight: float = 1.0, k: int = 60) -> None:
        for rank, asin in enumerate(results, start=1):
            scores[asin] = scores.get(asin, 0.0) + (weight / (k + rank))

    # Function that scores based on matches of keywords between query and product description
    def _match_score(self, value: object, text: str) -> float:
        if value in (None, "", []):
            return 0.0

        values = value if isinstance(value, list) else [value]
        text_terms = set(_terms(text))
        score = 0.0

        for item in values:
            query_terms = set(_terms(str(item)))

            if not query_terms:
                continue

            overlap = len(query_terms & text_terms) / len(query_terms)
            score += overlap

        return score

    def _product_fields(self, product: dict) -> dict[str, str]:
        return {
            "title": _text(product.get("title")).lower(),
            "categories": _text(product.get("categories")).lower(),
            "features": _text(product.get("features")).lower(),
            "details": _text(product.get("details")).lower(),
            "store": _text(product.get("store")).lower(),
            "description": _text(product.get("description")).lower(),
        }

    # Generates score for each attribute of the product that best matches the current state
    def _constraint_score(self, product: dict, session: SessionState) -> float:
        fields = self._product_fields(product)
        slots = session.slots

        score = 0.0

        # CATEGORY
        category = slots.get("category")
        if category:
            score += 0.15 * self._match_score(
                category,
                fields["title"],
            )

            score += 0.15 * self._match_score(
                category,
                fields["categories"],
            )

        # COLOR
        color = slots.get("color")
        if color:
            score += 0.05 * self._match_score(
                color,
                " ".join(fields.values()),
            )

        # MATERIAL
        material = slots.get("material")
        if material:
            score += 0.1 * self._match_score(
                material,
                fields["title"] + " " + fields["features"] + " " + fields["details"] + " " + fields["description"]
            )

        # BRAND
        brand = slots.get("brand")
        if brand:
            score += 0.1 * self._match_score(
                brand,
                fields["store"] + " " + fields["title"],
            )

        # SIZE
        size = slots.get("size")
        if size:
            score += 0.05 * self._match_score(
                size,
                fields["details"] + " " + fields["features"],
            )

        # STYLE
        style = slots.get("style")
        if style:
            score += 0.05 * self._match_score(
                style,
                fields["title"]
                + " "
                + fields["features"]
                + " "
                + fields["details"],
            )

        # FEATURES
        features = slots.get("feature")
        if features:
            score += 0.05 * self._match_score(
                features,
                fields["features"]
                + " "
                + fields["details"]
                + " "
                + fields["description"],
            )

        # USE CASE
        use_case = slots.get("use_case")
        if use_case:
            score += 0.05 * self._match_score(
                use_case,
                fields["title"]
                + " "
                + fields["features"]
                + " "
                + fields["description"],
            )

        return score

    # Shared ranking logic, factored out of _search so respond() can pull a
    # larger candidate pool (for attribute_selector) without duplicating the
    # retrieval/scoring code or running it twice.
    def _ranked_asins(self, session: SessionState, user_message: str) -> list[str]:
        # 1. Build queries
        slot_terms = _terms(" ".join(session.query_terms()))
        message_terms = _terms(user_message)
        all_terms = list(dict.fromkeys(slot_terms + message_terms))[:40]

        # 2. Candidate retrieval
        candidate_limit = 300  # adjusted from 200 to 300
        all_results = self._bm25_search(all_terms, limit=candidate_limit)
        slot_results = self._bm25_search(slot_terms, limit=candidate_limit)

        # 3. Fuse retrieval routes
        scores: dict[str, float] = {}
        self._add_rrf(scores, all_results, weight=1.0)
        self._add_rrf(scores, slot_results, weight=1.3) # Fine tune the weight

        # 4. Filter + Re-rank
        budget = session.budget_limit()
        ranked = []

        for asin, retrieval_score in scores.items():
            product = self._products[asin]
            price = self._prices.get(asin)

            if (
                budget is not None
                and price is not None
                and price > budget
            ):
                continue

            constraint_score = self._constraint_score(product, session)
            final_score = 20.0 * retrieval_score + constraint_score # Fine tune the 20.0

            ranked.append((final_score, asin))

        # 5. Sort highest score first
        ranked.sort(reverse=True)

        return [asin for _, asin in ranked]

    def _search(self, session: SessionState, user_message: str, top_k: int) -> list[dict]:
        ranked_asins = self._ranked_asins(session, user_message)

        return [
            {"parent_asin": asin}
            for asin in ranked_asins[:top_k]
        ]

    # Feeds the current session state (slots the user has already given)
    # together with the top 300 candidates from _search's ranking into
    # attribute_selector.choose_attribute to figure out the single most
    # useful attribute to ask the user about next. Attributes already put
    # to the user this session (session.asked) are excluded, and whatever
    # gets chosen here is recorded into session.asked before it's returned
    # -- so a "no preference"/ignored answer doesn't cause the same
    # question to get re-asked turn after turn.
    def _select_ask_attribute(self, session: SessionState, user_message: str) -> str | None:
        ranked_asins = self._ranked_asins(session, user_message)
        candidate_pool = [self._products[asin] for asin in ranked_asins[:300]]

        known = known_from_snapshot(session.snapshot())
        suggestion = choose_attribute(known, candidate_pool, asked=session.asked)

        if suggestion is None:
            return None

        session.asked.add(suggestion.attribute)
        return suggestion.attribute

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        session = self._sessions.get(session_id)
        if session is None:
            raise RuntimeError("reset must be called before respond")
        
        print(user_message, turn)
        usage = track(session, user_message, turn)

        print(session.slots)

        recommendations = self._search(session, user_message, top_k)
        ask_attribute = self._select_ask_attribute(session, user_message)
        session.pending_attribute = ask_attribute

        print(ask_attribute)
        message = session.last_assistant_message or "Here are the closest matches I found."
        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
            },
        }