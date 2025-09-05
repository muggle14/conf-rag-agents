"""
Session service facade that wraps the Cosmos-backed session store.
"""

from typing import Optional

from memory.cosmos_session import SessionStore


class SessionService:
    """Facade for session memory operations."""

    def __init__(self, store: Optional[SessionStore] = None):
        self.store = store or SessionStore()

    def remember_clarification(
        self, session_id: str, question: str, *, user_id: Optional[str] = None
    ) -> None:
        self.store.remember_clarification(session_id, question, user_id)

    def remember_accept(
        self, session_id: str, doc_ids: list[str], *, user_id: Optional[str] = None
    ) -> None:
        self.store.remember_accept(session_id, doc_ids, user_id)

    def remember_reject(
        self,
        session_id: str,
        doc_ids: list[str],
        reason: Optional[str] = None,
        *,
        user_id: Optional[str] = None,
    ) -> None:
        self.store.remember_reject(session_id, doc_ids, reason, user_id)

    def add_query(
        self,
        session_id: str,
        query: str,
        response: str,
        *,
        user_id: Optional[str] = None,
    ) -> None:
        self.store.add_query(session_id, query, response, user_id)
