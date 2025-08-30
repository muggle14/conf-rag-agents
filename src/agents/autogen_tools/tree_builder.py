# tree_builder.py
"""
Tree builder utility for creating page hierarchy trees from Gremlin graph data.
Uses graph_normalize to clean GraphSON responses.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Set

from agents.autogen_tools.graph_normalize import normalize_vertex

logger = logging.getLogger(__name__)


class TreeBuilder:
    """Build and render page hierarchy trees from graph data."""

    def __init__(self, gremlin_client):
        """
        Initialize tree builder.

        Args:
            gremlin_client: Gremlin client for graph queries
        """
        self.gremlin_client = gremlin_client

    async def build_page_trees(self, page_ids: Set[str]) -> List[Dict[str, Any]]:
        """
        Build page hierarchy trees for a set of page IDs.

        Args:
            page_ids: Set of page IDs to build trees for

        Returns:
            List of rendered tree structures
        """
        trees = []
        processed_roots = set()

        for page_id in page_ids:
            # Get ancestry path
            ancestry_query = f"""
            g.V('{page_id}')
              .repeat(out('ParentOf')).emit()
              .path()
              .by(valueMap(true))
            """

            try:
                result = await asyncio.to_thread(
                    self.gremlin_client.submit, ancestry_query
                )
                paths = result.all().result()

                if paths and len(paths[0]) > 0:
                    # Normalize the path vertices
                    normalized_path = [normalize_vertex(v) for v in paths[0]]

                    # Root is the last element in the ancestry path
                    root_vertex = normalized_path[-1] if normalized_path else None

                    if root_vertex and root_vertex["id"] not in processed_roots:
                        processed_roots.add(root_vertex["id"])
                        tree = await self.build_complete_tree(
                            root_vertex["id"], answer_page_ids={page_id}
                        )
                        trees.append(tree)

            except Exception as e:
                logger.error(f"Error building tree for {page_id}: {e}")
                continue

        return trees

    async def build_complete_tree(
        self, root_id: str, answer_page_ids: Optional[Set[str]] = None
    ) -> Dict[str, Any]:
        """
        Build complete tree from root node.

        Args:
            root_id: Root page ID
            answer_page_ids: Set of page IDs that contain answers

        Returns:
            Tree structure dictionary
        """
        answer_page_ids = answer_page_ids or set()

        # Get all descendants using tree()
        query = f"""
        g.V('{root_id}')
          .repeat(__.in('ParentOf')).emit()
          .tree()
          .by(valueMap(true))
        """

        try:
            result = await asyncio.to_thread(self.gremlin_client.submit, query)
            tree_data = result.all().result()

            if not tree_data or len(tree_data) == 0:
                # Fallback for single node
                return self._create_single_node_tree(root_id, answer_page_ids)

            # Convert GraphSON tree to our structure
            return self._convert_tree(tree_data[0], answer_page_ids)

        except Exception as e:
            logger.error(f"Error building complete tree for {root_id}: {e}")
            return self._create_single_node_tree(root_id, answer_page_ids)

    def _convert_tree(
        self, tree_data: Dict, answer_page_ids: Set[str]
    ) -> Dict[str, Any]:
        """
        Convert GraphSON tree to our tree structure.

        Args:
            tree_data: GraphSON tree data
            answer_page_ids: Set of page IDs that contain answers

        Returns:
            Normalized tree structure
        """

        def convert_node(node_data, vertex_data=None):
            # Handle the root level which has vertex data as key
            if vertex_data is None:
                # Root level - extract first key as vertex data
                if isinstance(node_data, dict) and len(node_data) > 0:
                    vertex_data = list(node_data.keys())[0]
                    children_data = node_data[vertex_data]
                else:
                    return None
            else:
                children_data = node_data

            # Normalize vertex
            normalized = (
                normalize_vertex(vertex_data)
                if isinstance(vertex_data, dict)
                else {"id": vertex_data}
            )
            page_id = normalized.get("id") or normalized.get("page_id")

            # Build node
            node = {
                "page_id": page_id,
                "title": normalized.get("title", "Untitled"),
                "url": normalized.get("url", f"/wiki/pages/{page_id}"),
                "is_answer_source": page_id in answer_page_ids,
                "children": [],
            }

            # Process children recursively
            if isinstance(children_data, dict):
                for child_vertex, child_subtree in children_data.items():
                    child_node = convert_node(child_subtree, child_vertex)
                    if child_node:
                        node["children"].append(child_node)

            return node

        return convert_node(tree_data)

    def _create_single_node_tree(
        self, page_id: str, answer_page_ids: Set[str]
    ) -> Dict[str, Any]:
        """Create a single-node tree for fallback."""
        return {
            "page_id": page_id,
            "title": page_id,
            "url": f"/wiki/pages/{page_id}",
            "is_answer_source": page_id in answer_page_ids,
            "children": [],
        }

    def render_tree_markdown(self, tree: Dict[str, Any], level: int = 0) -> str:
        """
        Render tree as markdown with indentation.

        Args:
            tree: Tree structure
            level: Current indentation level

        Returns:
            Markdown string representation
        """
        indent = "  " * level

        # Highlight if this node contains answer
        if tree.get("is_answer_source"):
            line = (
                f"{indent}- **[{tree['title']}]({tree['url']})** ⭐ *(contains answer)*"
            )
        else:
            line = f"{indent}- [{tree['title']}]({tree['url']})"

        lines = [line]

        # Render children
        for child in tree.get("children", []):
            lines.append(self.render_tree_markdown(child, level + 1))

        return "\n".join(lines)

    def tree_contains_answer(self, tree: Dict[str, Any]) -> bool:
        """
        Check if tree contains any answer sources.

        Args:
            tree: Tree structure

        Returns:
            True if tree contains answer sources
        """
        if tree.get("is_answer_source"):
            return True

        for child in tree.get("children", []):
            if self.tree_contains_answer(child):
                return True

        return False

    async def get_breadcrumb(self, page_id: str) -> List[str]:
        """
        Get breadcrumb path for a page.

        Args:
            page_id: Page ID

        Returns:
            List of titles from root to page
        """
        query = f"""
        g.V('{page_id}')
          .repeat(out('ParentOf')).emit()
          .values('title')
        """

        try:
            result = await asyncio.to_thread(self.gremlin_client.submit, query)
            titles = result.all().result()
            return list(reversed(titles))
        except Exception as e:
            logger.error(f"Error getting breadcrumb for {page_id}: {e}")
            return [page_id]
