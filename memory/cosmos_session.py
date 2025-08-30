# memory/cosmos_session.py
"""
Cosmos DB Session Store for durable per-session memory.
Includes user-level authentication and authorization.
"""

import datetime
import hashlib
import logging
import os
from typing import Any, Dict, List, Optional

from azure.cosmos import CosmosClient, PartitionKey, exceptions
from azure.identity import DefaultAzureCredential

# Configuration
# Use the SQL API database for sessions
DB_NAME = os.getenv(
    "COSMOS_SQL_DATABASE", os.getenv("COSMOS_SESSION_DATABASE", "rag-sessions")
)
CONTAINER = os.getenv(
    "COSMOS_SQL_CONTAINER", os.getenv("COSMOS_SESSION_CONTAINER", "sessions")
)
PK = PartitionKey(path="/session_id", kind="Hash")

# Logging
logger = logging.getLogger(__name__)


class SessionStore:
    """
    Durable session storage with user-level authentication.
    Stores conversation history, accepted documents, and user preferences.
    """

    def __init__(self, use_managed_identity: bool = False):
        """
        Initialize Cosmos DB session store.

        Args:
            use_managed_identity: Use Azure Managed Identity instead of connection keys
        """
        try:
            if use_managed_identity:
                # Use managed identity for production
                credential = DefaultAzureCredential()
                url = (
                    os.getenv("COSMOS_ENDPOINT", "")
                    .replace("/gremlin", "")
                    .replace(":443/", "")
                )
                if not url.startswith("https://"):
                    url = f"https://{url}"
                self.client = CosmosClient(url, credential=credential)
            else:
                # Use connection string or key for development
                # Use the SQL API endpoint directly
                endpoint = os.getenv("COSMOS_SQL_ENDPOINT", os.getenv("COSMOS_URL", ""))
                key = os.getenv("COSMOS_SQL_KEY", os.getenv("COSMOS_KEY", ""))

                if not endpoint:
                    raise ValueError(
                        "COSMOS_SQL_ENDPOINT or COSMOS_URL environment variable not set"
                    )
                if not key:
                    raise ValueError(
                        "COSMOS_SQL_KEY or COSMOS_KEY environment variable not set"
                    )

                self.client = CosmosClient(endpoint, credential=key)

            # Create database and container if not exists
            self.db = self.client.create_database_if_not_exists(DB_NAME)
            self.ct = self.db.create_container_if_not_exists(
                id=CONTAINER,
                partition_key=PK,
                # Don't specify throughput for serverless accounts
            )
            logger.info(f"Connected to Cosmos DB: {DB_NAME}/{CONTAINER}")

        except Exception as e:
            logger.error(f"Failed to initialize Cosmos DB session store: {e}")
            raise

    def _create_session_key(
        self, session_id: str, user_id: Optional[str] = None
    ) -> str:
        """
        Create a composite session key that includes user ID for security.

        Args:
            session_id: The session identifier
            user_id: The user identifier for access control

        Returns:
            Composite session key
        """
        if user_id:
            # Create hash of user_id + session_id for security
            combined = f"{user_id}:{session_id}"
            return hashlib.sha256(combined.encode()).hexdigest()[:32]
        return session_id

    def get(self, session_id: str, user_id: Optional[str] = None) -> dict:
        """
        Retrieve session data with user authentication.

        Args:
            session_id: Session identifier
            user_id: User identifier for access control

        Returns:
            Session data dictionary
        """
        try:
            composite_key = self._create_session_key(session_id, user_id)
            item = self.ct.read_item(item=composite_key, partition_key=session_id)

            # Verify user access
            if user_id and item.get("user_id") != user_id:
                logger.warning(
                    f"Unauthorized access attempt: user {user_id} to session {session_id}"
                )
                return self._create_default_session(session_id, user_id)

            return item

        except exceptions.CosmosResourceNotFoundError:
            return self._create_default_session(session_id, user_id)
        except Exception as e:
            logger.error(f"Error reading session {session_id}: {e}")
            return self._create_default_session(session_id, user_id)

    def _create_default_session(
        self, session_id: str, user_id: Optional[str] = None
    ) -> dict:
        """Create a default session object."""
        composite_key = self._create_session_key(session_id, user_id)
        return {
            "id": composite_key,
            "session_id": session_id,
            "user_id": user_id,
            "history": [],
            "accepted_docs": [],
            "rejected_docs": [],
            "metadata": {
                "created_at": datetime.datetime.utcnow().isoformat() + "Z",
                "query_count": 0,
                "last_activity": None,
            },
            "user_preferences": {
                "preferred_sources": [],
                "expertise_level": "intermediate",
                "response_style": "concise",
            },
        }

    def upsert(self, session: dict, user_id: Optional[str] = None):
        """
        Update or insert session data with user authentication.

        Args:
            session: Session data dictionary
            user_id: User identifier for access control
        """
        try:
            # Validate user access
            if user_id and session.get("user_id") and session["user_id"] != user_id:
                logger.error(
                    f"User {user_id} cannot modify session owned by {session['user_id']}"
                )
                raise PermissionError(
                    f"User {user_id} not authorized to modify this session"
                )

            # Update timestamps
            session["updated_at"] = datetime.datetime.utcnow().isoformat() + "Z"
            session["metadata"] = session.get("metadata", {})
            session["metadata"]["last_activity"] = session["updated_at"]

            # Ensure user_id is set
            if user_id and not session.get("user_id"):
                session["user_id"] = user_id

            # Update composite key if needed
            if user_id:
                session["id"] = self._create_session_key(session["session_id"], user_id)

            self.ct.upsert_item(session)
            logger.info(f"Updated session {session['session_id']} for user {user_id}")

        except Exception as e:
            logger.error(f"Error upserting session: {e}")
            raise

    def remember_clarification(
        self, session_id: str, clarification: str, user_id: Optional[str] = None
    ):
        """
        Remember a clarification in the session history.

        Args:
            session_id: Session identifier
            clarification: Clarification text to remember
            user_id: User identifier for access control
        """
        s = self.get(session_id, user_id)
        s["history"].append(
            {
                "type": "clarification",
                "text": clarification,
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            }
        )
        self.upsert(s, user_id)

    def remember_accept(
        self, session_id: str, doc_ids: List[str], user_id: Optional[str] = None
    ):
        """
        Remember accepted documents in the session.

        Args:
            session_id: Session identifier
            doc_ids: List of document IDs that were accepted
            user_id: User identifier for access control
        """
        s = self.get(session_id, user_id)
        s["accepted_docs"] = list(set((s.get("accepted_docs") or []) + doc_ids))
        s["history"].append(
            {
                "type": "accepted_docs",
                "doc_ids": doc_ids,
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            }
        )
        self.upsert(s, user_id)

    def remember_reject(
        self,
        session_id: str,
        doc_ids: List[str],
        reason: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        """
        Remember rejected documents in the session.

        Args:
            session_id: Session identifier
            doc_ids: List of document IDs that were rejected
            reason: Optional reason for rejection
            user_id: User identifier for access control
        """
        s = self.get(session_id, user_id)
        s["rejected_docs"] = list(set((s.get("rejected_docs") or []) + doc_ids))
        s["history"].append(
            {
                "type": "rejected_docs",
                "doc_ids": doc_ids,
                "reason": reason,
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            }
        )
        self.upsert(s, user_id)

    def add_thinking_step(
        self, session_id: str, step: Dict[str, Any], user_id: Optional[str] = None
    ):
        """
        Add a thinking process step to the session.

        Args:
            session_id: Session identifier
            step: Thinking step containing agent, action, reasoning, result
            user_id: User identifier for access control
        """
        s = self.get(session_id, user_id)

        # Initialize thinking_steps list if not present
        if "thinking_steps" not in s:
            s["thinking_steps"] = []

        thinking_step = {
            "timestamp": step.get("timestamp", datetime.datetime.utcnow().timestamp()),
            "agent": step.get("agent", "system"),
            "action": step.get("action", ""),
            "reasoning": step.get("reasoning", ""),
            "result": step.get("result"),
        }

        s["thinking_steps"].append(thinking_step)
        self.upsert(s, user_id)

    def get_thinking_steps(
        self, session_id: str, user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all thinking steps for a session.

        Args:
            session_id: Session identifier
            user_id: User identifier for access control

        Returns:
            List of thinking steps
        """
        s = self.get(session_id, user_id)
        return s.get("thinking_steps", [])

    def add_messages(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        user_id: Optional[str] = None,
    ):
        """
        Add multiple messages to conversation history.

        Args:
            session_id: Session identifier
            messages: List of message dictionaries with role, content, timestamp
            user_id: User identifier for access control
        """
        s = self.get(session_id, user_id)

        for msg in messages:
            event = {
                "type": "message",
                "timestamp": msg.get(
                    "timestamp", datetime.datetime.utcnow().timestamp()
                ),
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
                "metadata": msg.get("metadata", {}),
            }
            s["history"].append(event)

        self.upsert(s, user_id)

    def add_query(
        self, session_id: str, query: str, response: str, user_id: Optional[str] = None
    ):
        """
        Add a query-response pair to the session history.

        Args:
            session_id: Session identifier
            query: User query
            response: System response
            user_id: User identifier for access control
        """
        s = self.get(session_id, user_id)
        s["history"].append(
            {
                "type": "qa_pair",
                "query": query,
                "response": response,
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            }
        )

        # Update query count
        s["metadata"]["query_count"] = s["metadata"].get("query_count", 0) + 1

        self.upsert(s, user_id)

    def get_conversation_history(
        self, session_id: str, user_id: Optional[str] = None, limit: int = 10
    ) -> List[Dict]:
        """
        Get recent conversation history for context.

        Args:
            session_id: Session identifier
            user_id: User identifier for access control
            limit: Maximum number of history items to return

        Returns:
            List of recent conversation history items
        """
        s = self.get(session_id, user_id)
        history = s.get("history", [])

        # Filter to only QA pairs for conversation context
        qa_pairs = [h for h in history if h.get("type") == "qa_pair"]

        return qa_pairs[-limit:] if qa_pairs else []

    def update_user_preferences(
        self,
        session_id: str,
        preferences: Dict[str, Any],
        user_id: Optional[str] = None,
    ):
        """
        Update user preferences for the session.

        Args:
            session_id: Session identifier
            preferences: Dictionary of user preferences
            user_id: User identifier for access control
        """
        s = self.get(session_id, user_id)
        s["user_preferences"] = {**s.get("user_preferences", {}), **preferences}
        s["history"].append(
            {
                "type": "preferences_update",
                "preferences": preferences,
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            }
        )
        self.upsert(s, user_id)

    def get_session_stats(
        self, session_id: str, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get statistics about the session.

        Args:
            session_id: Session identifier
            user_id: User identifier for access control

        Returns:
            Dictionary with session statistics
        """
        s = self.get(session_id, user_id)

        history_types = {}
        for item in s.get("history", []):
            item_type = item.get("type", "unknown")
            history_types[item_type] = history_types.get(item_type, 0) + 1

        return {
            "session_id": session_id,
            "user_id": s.get("user_id"),
            "total_queries": s.get("metadata", {}).get("query_count", 0),
            "accepted_docs": len(s.get("accepted_docs", [])),
            "rejected_docs": len(s.get("rejected_docs", [])),
            "history_breakdown": history_types,
            "created_at": s.get("metadata", {}).get("created_at"),
            "last_activity": s.get("metadata", {}).get("last_activity"),
        }

    def delete_session(self, session_id: str, user_id: Optional[str] = None):
        """
        Delete a session (with user authentication).

        Args:
            session_id: Session identifier
            user_id: User identifier for access control
        """
        try:
            # Verify user owns the session
            s = self.get(session_id, user_id)
            if user_id and s.get("user_id") != user_id:
                raise PermissionError(
                    f"User {user_id} not authorized to delete this session"
                )

            composite_key = self._create_session_key(session_id, user_id)
            self.ct.delete_item(item=composite_key, partition_key=session_id)
            logger.info(f"Deleted session {session_id} for user {user_id}")

        except Exception as e:
            logger.error(f"Error deleting session: {e}")
            raise


# Example usage
if __name__ == "__main__":
    # Example with user authentication
    store = SessionStore()

    # User "alice" creates a session
    session_id = "test-session-123"
    user_id = "alice@example.com"

    # Remember some interactions
    store.add_query(
        session_id,
        "What is Confluence?",
        "Confluence is a collaboration tool...",
        user_id,
    )
    store.remember_accept(session_id, ["doc1", "doc2"], user_id)
    store.remember_clarification(session_id, "Looking for pricing information", user_id)

    # Get session stats
    stats = store.get_session_stats(session_id, user_id)
    print(f"Session stats: {stats}")

    # Get conversation history
    history = store.get_conversation_history(session_id, user_id)
    print(f"Conversation history: {history}")
