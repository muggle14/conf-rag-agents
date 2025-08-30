"""
graph_lookup.py
===============
Fetch neighbouring pages from Cosmos DB (Gremlin API).

Usage
-----
from agents.autogen_tools.graph_lookup import get_neighbors
docs = get_neighbors("123456", edge_types=("ParentOf","LinksTo"), k=5)
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List

from gremlin_python.driver import client, serializer

log = logging.getLogger("confluence-qa.graph-lookup")

# ---------------- Environment ----------------
ENDPOINT = os.getenv("COSMOS_GRAPH_DB_ENDPOINT")
KEY = os.getenv("COSMOS_GRAPH_DB_KEY")
DB = os.getenv("COSMOS_GRAPH_DB_DATABASE", "confluence-graph")
COLL = os.getenv("COSMOS_GRAPH_DB_COLLECTION", "page-relationships")

# ---------------- Client (singleton) ---------
_GR = client.Client(
    f"wss://{ENDPOINT}:443/",
    "g",
    username=f"/dbs/{DB}/colls/{COLL}",
    password=KEY,
    message_serializer=serializer.GraphSONSerializersV2d0(),
)


# ---------------- Public helper --------------
def get_neighbors(
    page_id: str,
    edge_types: tuple[str, ...] = ("ParentOf", "LinksTo", "References"),
    *,
    k: int = 5,
    fields: tuple[str, ...] = ("id", "title", "url"),
    trace_id: str = None,
) -> List[Dict]:
    """
    Parameters
    ----------
    page_id    : str – vertex id of the page whose neighbourhood we want
    edge_types : tuple[str] – which edge labels count as 'neighbour'
    k          : int – max neighbours to return
    fields     : tuple[str] – properties to project
    trace_id   : str – optional trace ID for telemetry

    Returns
    -------
    List[dict] – at most k dictionaries with requested fields
    """
    from tracing.autogen_tracer import log, new_trace_id

    # Generate trace_id if not provided
    if not trace_id:
        trace_id = new_trace_id()

    # Log graph operation start
    log(
        "graph_start",
        trace_id,
        page_id=page_id,
        edge_types=edge_types,
        max_neighbors=k,
        operation="get_neighbors",
    )

    edge_labels = ",".join(f"'{e}'" for e in edge_types)
    proj = ",".join(f"'{f}'" for f in fields)

    gremlin = (
        f"g.V('{page_id}')"
        f".bothE({edge_labels}).otherV()"
        f".limit({k})"
        f".project({proj})" + "".join([f".by(values('{f}'))" for f in fields])
    )

    try:
        res = _GR.submitAsync(gremlin).result().all().result()
        neighbors = [{f: v for f, v in zip(fields, row)} for row in res]

        # Log graph neighbors found
        log(
            "graph_neighbors",
            trace_id,
            page_id=page_id,
            neighbor_count=len(neighbors),
            neighbor_ids=[n.get("id") for n in neighbors if "id" in n],
        )

        return neighbors
    except Exception as exc:
        log.warning("Gremlin query failed: %s", exc)

        # Log graph error
        log("graph_error", trace_id, page_id=page_id, error=str(exc))

        return []


# How to use both tools side by side
# --------------------------------------------------------------------------- #
# from agents.autogen_tools.graph_lookup   import get_neighbors  # READ
# from agents.autogen_tools.graph_feedback import upsert_edge    # WRITE

# # ---- Router enriches context ----
# neigh_docs = get_neighbors(main_doc["id"], k=5)

# # ---- Agent spawns sub-question ----
# upsert_edge(parent_q, child_q)
# --------------------------------------------------------------------------- #

# | Purpose                                                                 | Module                        | Direction      | Typical call-site                |
# |-------------------------------------------------------------------------|-------------------------------|---------------|----------------------------------|
# | Look up existing neighbours in the graph so the agent can enrich its context | `graph_lookup.py` (or `neighbor_lookup.py`) | READ          | Router / retrieval pipeline      |
# | Feed back new logical links (DependsOn, etc.) that the agent discovers while reasoning | `graph_feedback.py`           | WRITE / UPSERT | Agent callback `on_new_subquestion()` |
