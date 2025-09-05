"""
Search service facade that abstracts Azure Search tool usage.
"""

from typing import Optional

from src.agents.autogen_tools.search_tool import AzureSearchTool


class SearchService:
    """Facade for search operations."""

    def __init__(self, tool: Optional[AzureSearchTool] = None):
        self.tool = tool or AzureSearchTool()

    def search(
        self,
        query: str,
        *,
        k: int = 10,
        space: Optional[str] = None,
        trace_id: Optional[str] = None,
        enable_agent_rerank: bool = False,
        mode: str = "simple",
    ) -> list[dict]:
        return self.tool.search(
            query,
            k=k,
            space=space,
            trace_id=trace_id,
            enable_agent_rerank=enable_agent_rerank,
            mode=mode,
        )
