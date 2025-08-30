"""
graph_feedback.py
=================
Persist reasoning breadcrumbs back into Cosmos DB (Gremlin API).

• Each question text is hashed → fixed-length vertex id.
• When an Autogen agent spawns a sub-question, we      ⤵
     └─► ensure both vertices exist (label='question')
     └─► upsert an edge (default label = 'DependsOn')

The module offers:

    upsert_edge(parent_text, child_text, etype="DependsOn", weight=1.0)

…plus a convenience bulk function:

    upsert_edges([(src, dst), …], etype="DependsOn")

All calls are **fire-and-forget**; exceptions are logged but won’t break
the agent’s reply path.
"""

from __future__ import annotations

import asyncio
import datetime
import functools
import hashlib
import logging
import os
from collections.abc import Iterable
from typing import Tuple

from gremlin_python.driver import client, serializer
from gremlin_python.process.traversal import T

# --------------------------------------------------------------------------- #
#  Env + logger                                                               #
# --------------------------------------------------------------------------- #
log = logging.getLogger("confluence-qa.graph-feedback")

GREMLIN_ENDPOINT = os.getenv("COSMOS_GRAPH_DB_ENDPOINT")
GREMLIN_KEY = os.getenv("COSMOS_GRAPH_DB_KEY")
GRAPH_DB = os.getenv("COSMOS_GRAPH_DB_DATABASE")
GRAPH_COLLECTION = os.getenv("COSMOS_GRAPH_DB_COLLECTION")
GRAPH_PORT = 443


# --------------------------------------------------------------------------- #
#  Client singleton                                                           #
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=1)
def _client() -> client.Client:
    if not (GREMLIN_ENDPOINT and GREMLIN_KEY):
        raise RuntimeError("Cosmos Gremlin endpoint/key env-vars not set.")
    return client.Client(
        f"wss://{GREMLIN_ENDPOINT}:{GRAPH_PORT}/",
        "g",
        username=f"/dbs/{GRAPH_DB}/colls/{GRAPH_COLLECTION}",
        password=GREMLIN_KEY,
        message_serializer=serializer.GraphSONSerializersV2d0(),
    )


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #
_HASH_LEN = 20


def _hash(text: str) -> str:
    """MD5 → hex → truncate so vertex id ≤ 50 chars (Cosmos limit)."""
    return hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()[:_HASH_LEN]


def _now_iso() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


# --------------------------------------------------------------------------- #
#  Gremlin templates (parameterised) - Refined for Cosmos DB schema           #
# --------------------------------------------------------------------------- #
_UPSERT_VERTEX = """
g.V(qid).fold().
  coalesce(unfold(),
           addV('question')
             .property(T.id, qid)
             .property('pageId', qid)           # Required partition key for Cosmos DB
             .property('text', qtxt)
             .property('created_at', ctime)
             .property('updated_at', ctime)
             .property('label', 'question')
             .property('version', 1))
"""

_ADD_EDGE = """
.coalesce(
  __.outE(etype).where(inV().hasId(dst_id)),
  addE(etype)
    .to(V(dst_id))
    .property('created_at', ctime)
    .property('weight', weight)
    .property('label', etype)
)
"""


# --------------------------------------------------------------------------- #
#  Public API                                                                  #
# --------------------------------------------------------------------------- #
def upsert_edge(
    src_q: str,
    dst_q: str,
    *,
    etype: str = "DependsOn",
    weight: float = 1.0,
    trace_id: str = None,
) -> None:
    """
    Ensure vertices for `src_q` and `dst_q`, then add (or upsert) an edge
    with label `etype`. Safe to call many times; Gremlin idempotently reuses
    the existing edge if present.
    """
    from tracing.autogen_tracer import log as trace_log
    from tracing.autogen_tracer import new_trace_id

    # Generate trace_id if not provided
    if not trace_id:
        trace_id = new_trace_id()

    # Log feedback start
    trace_log(
        "feedback_start",
        trace_id,
        source_question=src_q,
        target_question=dst_q,
        edge_type=etype,
        weight=weight,
    )

    sid, did = _hash(src_q), _hash(dst_q)
    g = _client()

    try:
        g.submitAsync(
            _UPSERT_VERTEX.replace("qid", "sid")  # first vertex
            + _UPSERT_VERTEX.replace("qid", "did")  # second vertex
            + f"V(sid){_ADD_EDGE}",  # edge
            bindings=dict(
                sid=sid,
                did=did,
                qtxt_src=src_q,
                qtxt_dst=dst_q,
                ctime=_now_iso(),
                etype=etype,
                weight=weight,
                T=T,  # traversal token
            ),
        ).result()  # .result() waits for completion
        log.debug("Upserted edge %s: %s ➜ %s", etype, sid, did)

        # Log feedback complete
        trace_log(
            "feedback_complete",
            trace_id,
            source_id=sid,
            target_id=did,
            edge_type=etype,
            success=True,
        )
    except Exception as exc:  # broad: Gremlin wraps many error types
        log.warning("Failed to upsert edge (%s ➜ %s): %s", sid, did, exc, exc_info=True)

        # Log feedback error
        trace_log(
            "feedback_error", trace_id, source_id=sid, target_id=did, error=str(exc)
        )


def upsert_edges(pairs: Iterable[Tuple[str, str]], *, etype: str = "DependsOn") -> None:
    """
    Bulk convenience wrapper.  Falls back to sequential submission when the
    driver’s async-batch is not available.
    """
    loop = asyncio.get_event_loop()
    tasks = [loop.run_in_executor(None, upsert_edge, s, d, etype) for s, d in pairs]
    loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
