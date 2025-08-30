"""
graph_normalize.py
==================
Normalize GraphSON valueMap(true) responses from Cosmos DB Gremlin API
into flat dictionaries for easier consumption.

GraphSON often returns properties as lists even for single values.
This module provides utilities to unwrap and normalize these responses.
"""

from typing import Any, Dict, List


def _val(x):
    """
    Unwrap GraphSON value lists to extract single values.
    GraphSON valueMap(true) often yields lists; unwrap singletons.

    Args:
        x: A value that might be a single-element list

    Returns:
        The unwrapped value if it's a single-element list, otherwise the original value
    """
    if isinstance(x, list) and len(x) == 1:
        return x[0]
    return x


def normalize_vertex(vmap: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a GraphSON vertex representation into a flat dictionary.

    Args:
        vmap: A GraphSON vertex map with potentially nested/listed properties

    Returns:
        A flat dictionary with normalized properties
    """
    return {
        "id": _val(vmap.get("id")) or _val(vmap.get("page_id")),
        "page_id": _val(vmap.get("page_id")),
        "title": _val(vmap.get("title")),
        "url": _val(vmap.get("url")),
        "space": _val(vmap.get("space")),
        "path": _val(vmap.get("path")),
        "depth": _val(vmap.get("depth")),
    }


def normalize_list(lst: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize a list of GraphSON vertex representations.

    Args:
        lst: A list of GraphSON vertex maps

    Returns:
        A list of normalized flat dictionaries
    """
    return [normalize_vertex(x) for x in (lst or []) if isinstance(x, dict)]
