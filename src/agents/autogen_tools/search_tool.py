"""
Enhanced Azure Search Tool with optional agent-based reranking.
"""

from typing import Dict, List, Optional

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

from tracing import autogen_tracer as trace_logger

from .agents.reranker import agent_rerank
from .config.env import (
    AGENT_RERANK_ENABLED,
    AGENT_RERANK_MAXK,
    AZ_SEARCH_ENDPOINT,
    AZ_SEARCH_INDEX,
    AZ_SEARCH_KEY,
    AZ_SEARCH_SEM_CFG,
)


class AzureSearchTool:
    """
    Azure AI Search tool with optional GPT-4o reranking capability.
    """

    def __init__(
        self,
        endpoint: str = AZ_SEARCH_ENDPOINT,
        index: str = AZ_SEARCH_INDEX,
        key: str = AZ_SEARCH_KEY,
        semantic_cfg: str = AZ_SEARCH_SEM_CFG,
    ):
        """
        Initialize Azure Search client.

        Args:
            endpoint: Azure Search endpoint URL
            index: Index name
            key: Azure Search API key
            semantic_cfg: Optional semantic configuration name
        """
        if not endpoint or not key or not index:
            raise ValueError("Azure Search credentials not configured")

        self.client = SearchClient(
            endpoint=endpoint, index_name=index, credential=AzureKeyCredential(key)
        )
        self.semantic_cfg = semantic_cfg
        self.index_name = index

    def search(
        self,
        query: str,
        k: int = 10,
        space: Optional[str] = None,
        trace_id: Optional[str] = None,
        enable_agent_rerank: Optional[bool] = None,
        mode: str = "simple",
    ) -> List[Dict]:
        """
        Search with optional agent-based reranking.

        Args:
            query: Search query string
            k: Number of results to return
            space: Optional Confluence space filter
            trace_id: Optional trace ID for logging
            enable_agent_rerank: Override global rerank setting
            mode: Search mode - "simple", "semantic", or "vector"

        Returns:
            List of search hits, optionally reranked
        """
        with trace_logger.trace_tool(
            "azure_search",
            query=query,
            k=k,
            space=space,
            mode=mode,
            index_name=self.index_name,
        ) as (span, auto_trace_id):
            # Use provided trace_id or auto-generated one
            tid = trace_id or auto_trace_id

            # Build filter if space specified
            filter_expr = f"space eq '{space}'" if space else None

            # Configure search based on mode
            # Build select fields - only include fields that exist in the index
            # Start with basic fields that should always exist
            select_fields = ["id", "content", "title", "url"]

            # Add optional fields - these may or may not exist
            optional_fields = [
                "space",
                "path",
                "parent_page_id",
                "children_ids",
                "adjacent_ids",
                "graph_centrality_score",
            ]

            search_kwargs = {"search_text": query, "top": k, "filter": filter_expr}

            # Only add select if we have specific fields
            # If select is not specified, all fields are returned
            # This avoids the error when fields don't exist
            # search_kwargs["select"] = select_fields + optional_fields

            # Add semantic configuration if available and requested
            if mode == "semantic" and self.semantic_cfg:
                search_kwargs.update(
                    {
                        "query_type": "semantic",
                        "semantic_configuration_name": self.semantic_cfg,
                        "query_language": "en-us",
                        "query_speller": "lexicon",
                        "query_answer": "extractive|count-3",
                    }
                )

            # Perform search
            results = self.client.search(**search_kwargs)

            # Convert to list of dicts
            hits = []
            for r in results:
                # Build hit dictionary with available fields
                hit = {
                    "id": r.get("id", ""),
                    "title": r.get("title", "No Title"),
                    "url": r.get("url", ""),
                    "score": r.get("@search.score", 0),
                    "snippet": self._get_snippet(r),
                }

                # Add optional fields if they exist
                if "space" in r:
                    hit["space"] = r["space"]
                if "path" in r:
                    hit["path"] = r["path"]
                if "parent_page_id" in r:
                    hit["parent_page_id"] = r["parent_page_id"]
                if "children_ids" in r:
                    hit["children_ids"] = r["children_ids"]
                if "adjacent_ids" in r:
                    hit["adjacent_ids"] = r["adjacent_ids"]
                if "graph_centrality_score" in r:
                    hit["graph_centrality_score"] = r["graph_centrality_score"]

                # Add semantic answer if available
                if mode == "semantic" and "@search.captions" in r:
                    captions = r["@search.captions"]
                    if captions:
                        hit["semantic_snippet"] = captions[0].text

                hits.append(hit)

            span.set_attribute("search.results_raw", len(hits))
            span.add_event(
                "search_completed",
                {
                    "result_count": len(hits),
                    "top_score": hits[0]["score"] if hits else 0,
                },
            )

            # Log raw results
            if tid:
                trace_logger.log(
                    "search_results_raw",
                    tid,
                    count=len(hits),
                    ids=[h["id"] for h in hits],
                    scores=[h["score"] for h in hits],
                )

            # Optional agent re-rank
            use_rerank = (
                AGENT_RERANK_ENABLED
                if enable_agent_rerank is None
                else enable_agent_rerank
            )

            if use_rerank and hits:
                before_ids = [h["id"] for h in hits]

                # Rerank using GPT-4o
                hits = agent_rerank(
                    query, hits, maxk=min(AGENT_RERANK_MAXK, k), trace_id=tid
                )

                after_ids = [h["id"] for h in hits]

                # Log reranking details
                if tid:
                    trace_logger.log(
                        "agent_rerank",
                        tid,
                        before=before_ids[:AGENT_RERANK_MAXK],
                        after=after_ids[:AGENT_RERANK_MAXK],
                        agent_scores=[
                            h.get("agent_score", None) for h in hits[:AGENT_RERANK_MAXK]
                        ],
                    )

                span.set_attribute("search.reranked", True)
                span.add_event(
                    "rerank_completed",
                    {
                        "reranked_count": min(AGENT_RERANK_MAXK, len(hits)),
                        "order_changed": before_ids != after_ids,
                    },
                )

            return hits

    def _get_snippet(self, result: Dict, max_length: int = 500) -> str:
        """
        Extract snippet from search result.

        Args:
            result: Search result dictionary
            max_length: Maximum snippet length

        Returns:
            Snippet string
        """
        # Try to get semantic snippet first
        if "@search.captions" in result:
            captions = result["@search.captions"]
            if captions:
                return captions[0].text[:max_length]

        # Fall back to content
        content = result.get("content", "")
        if isinstance(content, str):
            return content[:max_length].replace("\n", " ")

        return ""

    def search_by_ids(self, ids: List[str]) -> List[Dict]:
        """
        Retrieve documents by their IDs.

        Args:
            ids: List of document IDs

        Returns:
            List of documents
        """
        docs = []
        for doc_id in ids:
            try:
                doc = self.client.get_document(key=doc_id)
                docs.append(doc)
            except Exception:
                continue
        return docs


# Convenience function for quick searches
def quick_search(
    query: str, k: int = 5, rerank: bool = False, mode: str = "simple"
) -> List[Dict]:
    """
    Quick search function for testing.

    Args:
        query: Search query
        k: Number of results
        rerank: Whether to enable reranking
        mode: Search mode

    Returns:
        Search results
    """
    tool = AzureSearchTool()
    return tool.search(query, k=k, enable_agent_rerank=rerank, mode=mode)


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "architecture overview"

    print("🔍 Azure Search with Optional Reranking")
    print("=" * 80)
    print(f"Query: '{query}'")
    print(f"Reranking: {AGENT_RERANK_ENABLED}")
    print("=" * 80)

    # Test without reranking
    print("\n📊 Results WITHOUT reranking:")
    print("-" * 40)
    results_no_rerank = quick_search(query, k=5, rerank=False)

    for i, hit in enumerate(results_no_rerank, 1):
        print(f"{i}. {hit['title']} (score: {hit['score']:.4f})")

    # Test with reranking
    print("\n📊 Results WITH reranking:")
    print("-" * 40)
    results_rerank = quick_search(query, k=5, rerank=True)

    for i, hit in enumerate(results_rerank, 1):
        agent_score = hit.get("agent_score", "N/A")
        print(f"{i}. {hit['title']} (agent_score: {agent_score})")

    # Compare top results
    print("\n🔄 Comparison:")
    print("-" * 40)

    if results_no_rerank and results_rerank:
        if results_no_rerank[0]["id"] != results_rerank[0]["id"]:
            print("✅ Reranking changed the top result!")
            print(f"  Before: {results_no_rerank[0]['title']}")
            print(f"  After:  {results_rerank[0]['title']}")
        else:
            print("ℹ️ Top result remained the same after reranking")
