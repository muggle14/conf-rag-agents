from __future__ import annotations

import functools
import os
from typing import Dict, List

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from dotenv import load_dotenv
from openai import AzureOpenAI

# Import AutoGen-compatible OpenTelemetry tracer
from tracing import autogen_tracer as trace_logger

# Load environment variables from .env file
load_dotenv()


# -------------------- constants --------------------
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "confluence-graph-embeddings-v2")
EMBED_DEPLOY = os.getenv("AOAI_EMBED_DEPLOY")
VECTOR_FIELD = "content_vector"
TEXT_FIELD = "content"


# -------------------- class ------------------------
class AzureAISearchTool:
    """
    Hybrid (vector + semantic + keyword) search tool usable by Autogen agents.
    """

    def __init__(
        self,
        endpoint: str = os.getenv("AZURE_SEARCH_ENDPOINT"),
        key: str = os.getenv("AZURE_SEARCH_KEY"),
    ):
        # Initialize clients only if credentials are available
        self._client = None
        self._openai_client = None
        self._embed_cached = None

        # Initialize Azure OpenAI client for embeddings first
        openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        openai_key = os.getenv("AZURE_OPENAI_API_KEY")

        if openai_endpoint and openai_key:
            try:
                self._openai_client = AzureOpenAI(
                    api_key=openai_key,
                    azure_endpoint=openai_endpoint,
                    api_version="2024-02-01",
                )
                # Create cached embedding function
                self._embed_cached = functools.lru_cache(maxsize=4096)(
                    self._embed_query
                )
                print("✅ Azure OpenAI client initialized successfully")
            except Exception as e:
                print(f"Warning: Failed to initialize Azure OpenAI client: {e}")
        else:
            print("Warning: Azure OpenAI credentials not found")

        # Initialize Azure Search client
        if endpoint and key:
            try:
                self._client = SearchClient(
                    endpoint=endpoint,
                    index_name=INDEX_NAME,
                    credential=AzureKeyCredential(key),
                )
                print("✅ Azure Search client initialized successfully")
            except Exception as e:
                print(f"Warning: Failed to initialize Azure Search client: {e}")
                # Try to initialize without any optional parameters
                try:
                    from azure.search.documents import SearchClient as SearchClientV2

                    self._client = SearchClientV2(
                        endpoint, INDEX_NAME, AzureKeyCredential(key)
                    )
                    print("✅ Azure Search client initialized with fallback method")
                except Exception as e2:
                    print(f"Warning: Fallback initialization also failed: {e2}")
        else:
            print("Warning: Azure Search credentials not found")

    # ---------- core --------------------------------
    def __call__(self, query: str, k: int = 8, mode: str = "vector") -> str:
        if not self._client:
            return "❌ Azure Search client not initialized"
        hits = self.raw_results(query, top=k, mode=mode)
        return "\n\n".join(self._format(hit) for hit in hits)

    # ---------- helpers -----------------------------
    def _embed_query(self, query: str) -> List[float]:
        """Generate embeddings for a query using Azure OpenAI."""
        response = self._openai_client.embeddings.create(
            input=query, model=EMBED_DEPLOY or "text-embedding-ada-002"
        )
        return response.data[0].embedding

    def raw_results(
        self, query: str, *, top: int = 3, mode: str = "vector"
    ) -> List[Dict]:
        """
        Get search results with configurable search mode.

        Args:
            query: Search query string
            top: Number of results to return
            mode: Search mode - "vector" (default), "keyword", or "hybrid"
        """
        # Use OpenTelemetry span for tracing
        with trace_logger.trace_tool(
            "azure_search",
            query=query,
            top=top,
            search_mode=mode,
            index_name=INDEX_NAME,
        ) as (span, trace_id):
            if mode == "vector":
                # Vector-only search
                if not self._embed_cached:
                    raise ValueError(
                        "Vector search requires Azure OpenAI client for embeddings"
                    )
                q_vec = self._embed_cached(query)
                results = list(
                    self._client.search(
                        search_text=None,
                        vector_queries=[
                            {
                                "kind": "vector",
                                "vector": q_vec,
                                "fields": VECTOR_FIELD,
                                "k": top,
                            }
                        ],
                        top=top,
                        select=[
                            "id",
                            TEXT_FIELD,
                            "title",
                            "url",
                            "parent_page_id",
                            "children_ids",
                            "adjacent_ids",
                            "graph_centrality_score",
                        ],
                    )
                )

                # Add result count to span
                span.set_attribute("search.result_count", len(results))
                span.add_event("search_completed", {"result_count": len(results)})

                return results
            elif mode == "keyword":
                # Keyword-only search
                results = list(
                    self._client.search(
                        search_text=query,
                        top=top,
                        select=[
                            "id",
                            TEXT_FIELD,
                            "title",
                            "url",
                            "parent_page_id",
                            "children_ids",
                            "adjacent_ids",
                            "graph_centrality_score",
                        ],
                    )
                )

                # Add result count to span
                span.set_attribute("search.result_count", len(results))
                span.add_event("search_completed", {"result_count": len(results)})

                return results
            elif mode == "hybrid":
                # Hybrid search (vector + keyword)
                q_vec = self._embed_cached(query)
                results = list(
                    self._client.search(
                        search_text=query,
                        vector_queries=[
                            {
                                "kind": "vector",
                                "vector": q_vec,
                                "fields": VECTOR_FIELD,
                                "k": top,
                            }
                        ],
                        top=top,
                        select=[
                            "id",
                            TEXT_FIELD,
                            "title",
                            "url",
                            "parent_page_id",
                            "children_ids",
                            "adjacent_ids",
                            "graph_centrality_score",
                        ],
                    )
                )

                # Add result count to span
                span.set_attribute("search.result_count", len(results))
                span.add_event("search_completed", {"result_count": len(results)})

                return results
            else:
                raise ValueError(
                    f"Invalid search mode: {mode}. Use 'vector', 'keyword', or 'hybrid'"
                )

    def by_ids(self, ids: List[str]) -> List[Dict]:
        docs = []
        for _id in ids:
            try:
                docs.append(self._client.get_document(key=_id))
            except Exception:
                continue
        return docs

    # ---------- private -----------------------------
    @staticmethod
    def _format(hit: Dict) -> str:
        title = hit.get("title", "No Title")
        doc_id = hit.get("id", "No ID")
        content = hit.get(TEXT_FIELD, "No content available")
        preview = content[:500].replace("\n", " ")
        return f"**{title}** (id:{doc_id})\n{preview}…"


if __name__ == "__main__":
    import sys

    # Default query
    default_query = "pricing details"

    # Get query from command line or use default
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else default_query

    print("🔍 Azure AI Search - Vector Search Test")
    print("=" * 80)
    print(f"Query: '{query}'")
    print("Mode: Vector Search")
    print("=" * 80)

    # Initialize the search tool
    search_tool = AzureAISearchTool()

    try:
        # Get formatted results
        print("\n📝 Formatted Results:")
        print("-" * 80)
        results = search_tool(query, k=5, mode="vector")
        print(results)
        print("-" * 80)

        # Get raw results for detailed analysis
        print("\n📊 Detailed Results with Metadata:")
        raw_results = search_tool.raw_results(query, top=5, mode="vector")

        if not raw_results:
            print("No results found for this query.")
        else:
            for i, result in enumerate(raw_results, 1):
                print(f"\n--- Result {i} ---")
                print(f"Title: {result.get('title', 'No Title')}")
                print(f"Score: {result.get('@search.score', 0):.4f}")
                print(f"URL: {result.get('url', 'No URL')}")
                print(
                    f"Graph Centrality: {result.get('graph_centrality_score', 0):.4f}"
                )

                # Graph relationships
                parent_id = result.get("parent_page_id")
                children_ids = result.get("children_ids", [])
                adjacent_ids = result.get("adjacent_ids", [])

                print(f"Parent Page: {parent_id if parent_id else 'None'}")
                print(f"Children: {len(children_ids)} pages")
                print(f"Adjacent: {len(adjacent_ids)} pages")

                # Show content preview
                content = result.get("content", "No content")
                preview = content[:250].replace("\n", " ")
                print(f"Content Preview: {preview}...")

        # Summary statistics
        print("\n📈 Summary:")
        print(f"Total results: {len(raw_results)}")
        if raw_results:
            avg_score = sum(r.get("@search.score", 0) for r in raw_results) / len(
                raw_results
            )
            print(f"Average score: {avg_score:.4f}")

    except Exception as e:
        print(f"\n❌ Vector search failed: {e}")
        print("\nPlease ensure:")
        print("1. Azure OpenAI credentials are set in .env")
        print("2. Embedding deployment name is correct")
        print("3. Search index has vector fields configured")
        print("\nUsage: python azure_search.py [query]")
        print("Example: python azure_search.py pricing details")

# Usage examples:
# source .venv/bin/activate && python src/agents/autogen_tools/azure_search.py
# source .venv/bin/activate && python src/agents/autogen_tools/azure_search.py pricing details
# source .venv/bin/activate && python src/agents/autogen_tools/azure_search.py security incident
