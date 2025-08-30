"""
neighbor_from_doc.py
--------------------
Utilities for extracting a page’s neighbours from the metadata already
embedded in an Azure AI Search hit.

A “neighbour” = parent, children, or siblings that reside in the *same*
Confluence space / index.  We rely on the WebApiSkill output fields:

    parent_page_id : str | None
    child_ids      : List[str]
    siblings       : List[str]
    graph_centrality_score : float  (optional)

Functions
~~~~~~~~~
extract_neighbor_ids(hit, k=5, rank=False) -> List[str]
fetch_neighbor_docs(search_tool, hit, k=5, rank=False) -> List[dict]
"""

from __future__ import annotations

from typing import Any, Dict, List


# --------------------------------------------------------------------------- #
#  Core: return list of neighbour IDs                                          #
# --------------------------------------------------------------------------- #
def extract_neighbor_ids(
    hit: Dict[str, Any], *, k: int = 5, rank: bool = False
) -> List[str]:
    """
    Parameters
    ----------
    hit   : Azure Search document (dict)
    k     : maximum total neighbours to return
    rank  : if True, sort children + siblings by their `graph_centrality_score`
            (parent always stays first when present)

    Returns
    -------
    List[str] – at most `k` unique doc IDs
    """
    parent_id = hit.get("parent_page_id")
    child_ids = list(hit.get("children_ids", []))
    sibling_ids = list(hit.get("adjacent_ids", []))

    # Note: Ranking by centrality requires fetching neighbor docs first
    # This function just returns IDs in order: parent, children, siblings
    # For ranking, use fetch_neighbor_docs() with rank=True instead

    ordered: List[str] = []
    if parent_id:
        ordered.append(parent_id)
    ordered.extend(child_ids)
    ordered.extend(sibling_ids)

    # ----- dedupe while preserving order -----
    uniq: List[str] = []
    seen = set()
    for _id in ordered:
        if _id and _id not in seen:
            uniq.append(_id)
            seen.add(_id)
        if len(uniq) >= k:
            break
    return uniq


# --------------------------------------------------------------------------- #
#  Convenience: return the neighbour *documents* directly                     #
# --------------------------------------------------------------------------- #
def fetch_neighbor_docs(
    search_tool, hit: Dict[str, Any], *, k: int = 5, rank: bool = False
) -> List[Dict[str, Any]]:
    """
    Parameters
    ----------
    search_tool : AzureAISearchTool instance (must expose .by_ids(ids: List[str]))
    hit         : the main Azure Search document (dict)
    k, rank     : forwarded to `extract_neighbor_ids`

    Returns
    -------
    List[dict] – the neighbour search documents (may be < k if some IDs missing)
    """
    ids = extract_neighbor_ids(hit, k=k, rank=False)  # Always get IDs first
    docs = search_tool.by_ids(ids)

    # ----- optional ranking by centrality (higher first) -----
    if rank and docs:
        # Sort by graph_centrality_score (higher scores first)
        docs.sort(key=lambda doc: doc.get("graph_centrality_score", 0), reverse=True)

        # Keep parent first if present
        parent_id = hit.get("parent_page_id")
        if parent_id:
            parent_docs = [doc for doc in docs if doc.get("id") == parent_id]
            other_docs = [doc for doc in docs if doc.get("id") != parent_id]
            docs = parent_docs + other_docs

    return docs[:k]  # Return at most k documents


# Usage example in the router.py
# --------------------------------------------------------------
# from autogen_tools.azure_search import AzureAISearchTool
# from autogen_tools.neighbor_from_doc import (
#     extract_neighbor_ids, fetch_neighbor_docs)

# search = AzureAISearchTool()
# main_hit = search.raw_results(question, top=1)[0]

# Get IDs only
# ids = extract_neighbor_ids(main_hit, k=5, rank=True)

# Or fetch full docs in one call
# neigh_docs = fetch_neighbor_docs(search, main_hit, k=5, rank=True)
# --------------------------------------------------------------
