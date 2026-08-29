from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from starter.session_state import SessionState
from starter.state_tracker import track


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
    """Conversational shopping agent: Groq slot tracker plus BM25 retrieval."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, SessionState] = {}
        self._prices: dict[str, float | None] = {}
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

    def _search(self, terms: list[str], top_k: int, budget: int | None) -> list[dict]:
        unique_terms = list(dict.fromkeys(terms))[:40]
        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        if not expression:
            return []
        fetch_k = top_k * 5 if budget is not None else top_k
        rows = self.connection.execute(
            "SELECT parent_asin FROM products WHERE products MATCH ? "
            "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
            (expression, fetch_k),
        ).fetchall()
        recommendations: list[dict] = []
        for (parent_asin,) in rows:
            asin = str(parent_asin)
            price = self._prices.get(asin)
            if budget is not None and price is not None and price > budget:
                continue
            recommendations.append({"parent_asin": asin})
            if len(recommendations) >= top_k:
                break
        return recommendations

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
        usage = track(session, user_message, turn)
        slot_terms = _terms(" ".join(session.query_terms()))
        message_terms = _terms(user_message)
        recommendations = self._search(slot_terms + message_terms, top_k, session.budget_limit())
        message = session.last_assistant_message or "Here are the closest matches I found."
        return {
            "message": message,
            "ask_attribute": session.next_ask_attribute(),
            "recommendations": recommendations,
            "usage": usage,
        }
