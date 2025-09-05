"""
Graph service facade that abstracts Cosmos Gremlin graph operations.
"""

from typing import Any, Dict, Optional

from src.agents.autogen_tools.graph_tool import GraphTool


class GraphService:
    """Facade for graph neighbor and tree queries."""

    def __init__(self, tool: Optional[GraphTool] = None):
        self.tool = tool or GraphTool()

    def neighbors(
        self, page_id: str, *, trace_id: Optional[str] = None
    ) -> Dict[str, Any]:
        return self.tool.neighbors(page_id, trace_id)

    def tree(
        self, page_id: str, *, depth: int = 2, trace_id: Optional[str] = None
    ) -> Dict[str, Any]:
        return self.tool.tree(page_id, depth=depth, trace_id=trace_id)
