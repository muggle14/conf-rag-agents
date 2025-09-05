"""
graph_tool.py
=============
Enhanced graph operations for fetching neighbors and tree structures
from Cosmos DB Gremlin API with normalized responses.

This module provides:
- neighbors(): Get 1-hop parents, children, and siblings
- tree(): Generate a compact hierarchical tree structure
"""

import os
import re
from typing import Any, Dict, Optional

from gremlin_python.driver import client, serializer

from .graph_normalize import normalize_list, normalize_vertex


class GraphTool:
    """
    Enhanced graph operations for Confluence page relationships.
    Provides normalized neighbor lookup and tree generation.
    """

    def __init__(self, url: str = None, username: str = None, password: str = None):
        """
        Initialize GraphTool with Cosmos DB Gremlin connection.

        Args:
            url: Gremlin endpoint URL (defaults to environment variable)
            username: Database path for auth (defaults to environment variable)
            password: Access key (defaults to environment variable)
        """

        def _first_non_empty(*values: str) -> Optional[str]:
            for v in values:
                if v:
                    return v
            return None

        def _extract_host(value: Optional[str]) -> Optional[str]:
            if not value:
                return None
            v = value.strip()
            if re.match(r"^[a-zA-Z]+://", v):
                from urllib.parse import urlparse

                parsed = urlparse(v)
                return parsed.hostname
            v = v.split("/")[0]
            v = v.split(":")[0]
            return v

        # Use provided values or fall back to environment variables (new or legacy)
        if not url:
            endpoint_raw = _first_non_empty(
                os.getenv("COSMOS_GRAPH_DB_ENDPOINT"),
                os.getenv("COSMOS_DB_ENDPOINT"),
            )
            host = _extract_host(endpoint_raw)
            if not host:
                raise ValueError(
                    "Cosmos Gremlin endpoint not configured (COSMOS_GRAPH_DB_ENDPOINT or COSMOS_DB_ENDPOINT)"
                )
            url = f"wss://{host}:443/"

        if not username:
            db = _first_non_empty(
                os.getenv("COSMOS_GRAPH_DB_DATABASE"),
                os.getenv("COSMOS_DB_DATABASE"),
                "confluence-graph",
            )
            coll = _first_non_empty(
                os.getenv("COSMOS_GRAPH_DB_COLLECTION"),
                os.getenv("COSMOS_DB_CONTAINER"),
                "page-relationships",
            )
            username = f"/dbs/{db}/colls/{coll}"

        if not password:
            password = _first_non_empty(
                os.getenv("COSMOS_GRAPH_DB_KEY"), os.getenv("COSMOS_DB_KEY")
            )

        self.client = client.Client(
            url,
            "g",
            username=username,
            password=password,
            message_serializer=serializer.GraphSONSerializersV2d0(),
        )

    def neighbors(self, page_id: str, trace_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch 1-hop neighbors (parents, children, siblings) for a given page.

        Args:
            page_id: The ID of the page to get neighbors for
            trace_id: Optional trace ID for telemetry

        Returns:
            Dictionary with 'parents', 'children', and 'siblings' lists,
            each containing normalized page dictionaries
        """
        # Import tracing utilities if needed
        if trace_id:
            try:
                from tracing.logger import log

                log("graph_start", trace_id, page_id=page_id)
            except ImportError:
                pass

        # Gremlin query to get parents, children, and siblings in one traversal
        query = f"""
        g.V().has('page','page_id','{page_id}')
         .project('parents','children','siblings')
           .by(__.in('PARENT_OF').valueMap(true).fold())
           .by(__.out('PARENT_OF').valueMap(true).fold())
           .by(__.in('PARENT_OF').out('PARENT_OF').where(values('page_id').is(neq('{page_id}'))).valueMap(true).fold())
        """

        try:
            # Execute query
            result = self.client.submit(query).all().result()

            if result:
                res = result[0]
                output = {
                    "parents": normalize_list(res.get("parents", [])),
                    "children": normalize_list(res.get("children", [])),
                    "siblings": normalize_list(res.get("siblings", [])),
                }
            else:
                output = {"parents": [], "children": [], "siblings": []}

            # Log results if tracing
            if trace_id:
                try:
                    from tracing.logger import log

                    log(
                        "graph_neighbors",
                        trace_id,
                        parents=len(output["parents"]),
                        children=len(output["children"]),
                        siblings=len(output["siblings"]),
                    )
                except ImportError:
                    pass

            return output

        except Exception as e:
            # Log error if tracing
            if trace_id:
                try:
                    from tracing.logger import log

                    log("graph_error", trace_id, error=str(e))
                except ImportError:
                    pass

            # Return empty structure on error
            return {"parents": [], "children": [], "siblings": []}

    def tree(
        self, page_id: str, depth: int = 2, trace_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a hierarchical tree structure starting from a page.

        Args:
            page_id: The root page ID
            depth: How many levels deep to traverse (default: 2)
            trace_id: Optional trace ID for telemetry

        Returns:
            Nested dictionary with 'id', 'title', 'url', and 'children' keys
        """
        # Import tracing utilities if needed
        if trace_id:
            try:
                from tracing.logger import log

                log("graph_tree_start", trace_id, page_id=page_id, depth=depth)
            except ImportError:
                pass

        # Gremlin query to build a tree structure
        query = f"""
        g.V().has('page','page_id','{page_id}')
         .repeat(__.out('PARENT_OF')).times({depth})
         .tree().by(valueMap(true))
        """

        try:
            # Execute query
            result = self.client.submit(query).all().result()

            if not result:
                # Return minimal structure if no results
                return {"id": page_id, "title": "Unknown", "url": None, "children": []}

            raw = result[0]

            # Recursive function to collapse the Gremlin tree structure
            def collapse(node):
                """
                Convert Gremlin tree format to our simplified format.
                The Gremlin tree returns nested dicts where keys are vertex maps.
                """
                if not node:
                    return None

                # Extract the single key-value pair (vertex -> subtree)
                items = list(node.items())
                if not items:
                    return None

                vertex_map, subtree = items[0]

                # Normalize the vertex
                root = normalize_vertex(vertex_map)

                # Process children recursively
                children = []
                if subtree and isinstance(subtree, dict):
                    for child_vertex, child_subtree in subtree.items():
                        child_node = collapse({child_vertex: child_subtree})
                        if child_node:
                            children.append(child_node)

                return {
                    "id": root["page_id"] or root["id"],
                    "title": root["title"] or "Untitled",
                    "url": root["url"],
                    "children": children,
                }

            tree_result = collapse(raw)

            # Log results if tracing
            if trace_id:
                try:
                    from tracing.logger import log

                    # Count total nodes in tree
                    def count_nodes(node):
                        if not node:
                            return 0
                        return 1 + sum(
                            count_nodes(child) for child in node.get("children", [])
                        )

                    log(
                        "graph_tree",
                        trace_id,
                        depth=depth,
                        total_nodes=count_nodes(tree_result),
                    )
                except ImportError:
                    pass

            return tree_result

        except Exception as e:
            # Log error if tracing
            if trace_id:
                try:
                    from tracing.logger import log

                    log("graph_tree_error", trace_id, error=str(e))
                except ImportError:
                    pass

            # Return minimal structure on error
            return {
                "id": page_id,
                "title": "Error loading tree",
                "url": None,
                "children": [],
            }
