"""
conversation.py
===============
Lightweight CRUD helpers for the Cosmos DB *conversations* container.

Data model (one document per message)
------------------------------------
{
    "id":              "<uuid4>",              # PK
    "conversation_id": "<same for thread>",    # string
    "role":            "user" | "assistant",
    "content":         "<message text>",
    "ts":              "2025-07-27T12:34:56Z", # ISO-UTC
    "deleted":         false                   # soft-delete flag
}

Functions exposed
-----------------
append_msg(convo_id, role, content)          → None
fetch_conversation(convo_id)                 → List[dict]
soft_delete_conversation(convo_id)           → None
list_conversations(limit=20, include_deleted=False) → List[str]
"""

from __future__ import annotations

import datetime
import logging
import os
import uuid
from typing import Any, Dict, List

from azure.cosmos import CosmosClient, exceptions

# --------------------------------------------------------------------------- #
#  Config & Cosmos client (singleton via module-level creation)               #
# --------------------------------------------------------------------------- #
log = logging.getLogger("confluence-qa.conversation")

_COSMOS_ENDPOINT = os.getenv("COSMOS_DB_ENDPOINT")
_COSMOS_KEY = os.getenv("COSMOS_DB_KEY")
_DB_NAME = os.getenv("COSMOS_DB_DATABASE_NAME", "ConfluenceQA")
_CONTAINER_NAME = os.getenv("COSMOS_DB_CONVERSATIONS_CONTAINER", "conversations")

if not (_COSMOS_ENDPOINT and _COSMOS_KEY):
    raise RuntimeError("COSMOS_DB_ENDPOINT / COSMOS_DB_KEY env-vars are required.")

_cosmos = CosmosClient(_COSMOS_ENDPOINT, _COSMOS_KEY)
_db = _cosmos.get_database_client(_DB_NAME)
_container = _db.get_container_client(_CONTAINER_NAME)


# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #
def _iso_now() -> str:
    return datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"


# --------------------------------------------------------------------------- #
#  Public API                                                                 #
# --------------------------------------------------------------------------- #
def append_msg(convo_id: str, *, role: str, content: str) -> None:
    """
    Append a message to a conversation (soft-real-time; no partition hot-spot
    because id = new uuid each time).
    """
    doc = {
        "id": str(uuid.uuid4()),
        "conversation_id": convo_id,
        "role": role,
        "content": content,
        "ts": _iso_now(),
        "deleted": False,
    }
    try:
        _container.upsert_item(doc)
    except exceptions.CosmosHttpResponseError as exc:
        log.error("Cosmos upsert failed: %s", exc)
        raise


def fetch_conversation(convo_id: str) -> List[Dict[str, Any]]:
    """
    Return the full (non-deleted) message list ordered by timestamp.
    """
    query = (
        "SELECT * FROM c "
        "WHERE c.conversation_id = @cid AND c.deleted = false "
        "ORDER BY c.ts"
    )
    items = _container.query_items(
        query=query,
        parameters=[{"name": "@cid", "value": convo_id}],
        enable_cross_partition_query=True,
    )
    return list(items)


def soft_delete_conversation(convo_id: str) -> None:
    """
    Mark all docs in a conversation as deleted.  This is idempotent.
    """
    docs = fetch_conversation(convo_id)
    for d in docs:
        d["deleted"] = True
        try:
            _container.upsert_item(d)
        except exceptions.CosmosHttpResponseError as exc:
            log.warning("Soft-delete failed for %s: %s", d["id"], exc)


def list_conversations(*, limit: int = 20, include_deleted: bool = False) -> List[str]:
    """
    Return up to `limit` distinct conversation ids (most recent first).
    """
    where = "" if include_deleted else "WHERE c.deleted = false"
    query = (
        f"SELECT DISTINCT c.conversation_id, MAX(c.ts) AS last "
        f"FROM c {where} GROUP BY c.conversation_id "
        f"ORDER BY last DESC OFFSET 0 LIMIT {limit}"
    )
    items = _container.query_items(query=query, enable_cross_partition_query=True)
    return [d["conversation_id"] for d in items]
