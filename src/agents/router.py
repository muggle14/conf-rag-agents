"""
confluence_qa.agents.router
---------------------------
• Defines the Autogen message-router rule  "search-with-neighbours".
• Instantiates the shared AzureAISearchTool.
• Provides build_agent() → returns a QAAgent pre-wired with logging + graph feedback.
"""

from __future__ import annotations

import logging
import os

# Message class is not available in current autogen version
# We'll use a simple dict instead
from typing import List

from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import (
    ChatCompletionClient,
    ModelCapabilities,
    ModelInfo,
    RequestUsage,
)
from openai import OpenAI

# ---------- local imports ----------
from src.agents.autogen_tools.azure_search import AzureAISearchTool
from src.agents.autogen_tools.graph_feedback import upsert_edge
from src.agents.autogen_tools.neighbor_from_doc import (
    extract_neighbor_ids,
)
from src.utils.logging import log_step

# ---------- logger ----------
log = logging.getLogger("confluence-qa.router")

# ---------- search tool (singleton) ----------
_search = AzureAISearchTool()


# ---------- router rule ----------
def search_with_neighbours(query: str, trace_id: str = None) -> str:
    """
    1. Run hybrid search against Azure AI Search.
    2. Pull neighbour docs via their IDs already stored in the main hit.
    3. Return a system-message containing main hit + neighbour snippets.
    """
    from tracing.autogen_tracer import log, new_trace_id

    # Generate trace_id if not provided
    if not trace_id:
        trace_id = new_trace_id()

    # Log search start
    log("search_start", trace_id, query=query, operation="search_with_neighbours")

    if not _search._client:
        log("search_error", trace_id, error="Search service not available")
        return "🛈 Search service not available."

    search_results: List[dict] = _search.raw_results(query, top=1)
    if not search_results:
        log("search_results", trace_id, count=0, found=False)
        return "🛈 No documents found."

    main_document = search_results[0]
    neighbor_page_ids = extract_neighbor_ids(main_document)

    # Log graph neighbors operation
    log(
        "graph_neighbors",
        trace_id,
        main_doc_id=main_document.get("id"),
        neighbor_ids=neighbor_page_ids,
        neighbor_count=len(neighbor_page_ids),
    )

    neighbor_documents = _search.by_ids(neighbor_page_ids)

    # ---------- craft context ----------
    context_parts = [
        "# Main document\n",
        main_document["content"][:800],
        "\n\n# Neighbours",
    ] + [f"- **{doc['title']}**\n{doc['content'][:400]}" for doc in neighbor_documents]

    log.info(f"Router: {len(neighbor_documents)} neighbours added")

    # Log search results with details
    log(
        "search_results",
        trace_id,
        count=1,
        main_doc_id=main_document.get("id"),
        neighbor_count=len(neighbor_documents),
        scores=[main_document.get("@search.score") or main_document.get("score")],
    )

    return "\n\n".join(context_parts)


# ---------- custom QA agent ----------
class QAAgent(AssistantAgent):
    """
    • Logs every tool call start/end to Cosmos DB.
    • Writes `DependsOn` edges when it spawns sub-questions.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def log_tool_call(self, tool_name, args, result):
        log_step(self.name, f"call:{tool_name}", f"args={args}", str(result)[:500])

    def log_subquestion(self, parent_q, child_q):
        try:
            upsert_edge(parent_q, child_q)
        except Exception as e:
            print(f"Warning: Failed to upsert edge: {e}")
        log_step(self.name, "spawn-subQ", parent_q, child_q)


# ---------- Minimal OpenAI Client Wrapper ----------
class MinimalOpenAIClient(ChatCompletionClient):
    """Minimal wrapper to satisfy autogen's ChatCompletionClient interface"""

    def __init__(self):
        # Create OpenAI client
        if os.getenv("AZURE_OPENAI_ENDPOINT"):
            # For Azure OpenAI
            self._client = OpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                base_url=f"{os.getenv('AZURE_OPENAI_ENDPOINT')}/openai/deployments/{os.getenv('AZURE_OPENAI_DEPLOYMENT', 'gpt-4')}",
                default_headers={"api-key": os.getenv("AZURE_OPENAI_API_KEY")},
            )
            self._model = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
        else:
            # Standard OpenAI
            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self._model = os.getenv("OPENAI_MODEL", "gpt-4")

    async def create(self, messages, **kwargs):
        """Create a chat completion"""
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                formatted_messages.append(msg)
            else:
                formatted_messages.append({"role": "user", "content": str(msg)})

        return self._client.chat.completions.create(
            model=self._model, messages=formatted_messages, **kwargs
        )

    def create_stream(self, messages, **kwargs):
        """Create a streaming chat completion"""
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                formatted_messages.append(msg)
            else:
                formatted_messages.append({"role": "user", "content": str(msg)})

        return self._client.chat.completions.create(
            model=self._model, messages=formatted_messages, stream=True, **kwargs
        )

    @property
    def capabilities(self):
        return ModelCapabilities(vision=False, function_calling=True, json_output=True)

    @property
    def model_info(self):
        return ModelInfo(
            vision=False,
            function_calling=True,
            json_output=True,
            family="openai",
            name=self._model,
        )

    def actual_usage(self):
        return RequestUsage(prompt_tokens=0, completion_tokens=0)

    def total_usage(self):
        return RequestUsage(prompt_tokens=0, completion_tokens=0)

    def count_tokens(self, messages, **kwargs):
        return 0  # Simple stub

    def remaining_tokens(self, messages, **kwargs):
        return 100000  # Simple stub

    async def close(self):
        pass  # No cleanup needed


# ---------- factory ----------
def build_agent() -> QAAgent:
    """
    Returns a ready-to-use QAAgent instance.
    Import and reuse this across Azure Function endpoints to avoid cold-start spin-up.
    """
    model_client = MinimalOpenAIClient()

    return QAAgent(
        name="QA_Agent",
        model_client=model_client,
        system_message="You are a helpful assistant that answers questions about Confluence documentation.",
    )


# ---------- eager instance (optional, keeps global warm) ----------
agent = build_agent()

# how to use the agent
# --------------------------------------------------------------
# from agents.router import agent   # single shared instance
# answer = await agent.ask("How do I reset my VPN token?")
# --------------------------------------------------------------
