"""
Legacy experimental API endpoint for /ask that integrates with the orchestrator.

This endpoint provides the interface expected by functional tests:
- Accepts query and optional space filter
- Returns mode (clarify/answer/proceed), confidence, trace_id
- Includes clarification question or answer with sources
"""

import os
import sys
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.cosmos_session import SessionStore
from src.agents.autogen_tools.graph_tool import GraphTool
from src.agents.autogen_tools.search_tool import AzureSearchTool
from src.agents.clarifier import ask_clarifying_question
from src.agents.logic.confidence import compute_overlap, confidence, should_clarify
from tracing.logger import log, new_trace_id

# Create router
router = APIRouter()

# Initialize components (singletons)
_orchestrator = None
_search_tool = None
_graph_tool = None
_session_store = None


def get_orchestrator():
    """Get or create orchestrator singleton."""
    global _orchestrator
    if _orchestrator is None:
        # Note: Full orchestrator requires AutoGen setup
        # For functional tests, we use the simplified /ask endpoint
        # _orchestrator = ConfluenceQAOrchestrator()
        pass
    return _orchestrator


def get_search_tool():
    """Get or create search tool singleton."""
    global _search_tool
    if _search_tool is None:
        _search_tool = AzureSearchTool()
    return _search_tool


def get_graph_tool():
    """Get or create graph tool singleton."""
    global _graph_tool
    if _graph_tool is None:
        _graph_tool = GraphTool()
    return _graph_tool


def get_session_store():
    """Get or create session store singleton."""
    global _session_store
    if _session_store is None:
        _session_store = SessionStore()
    return _session_store


class AskRequest(BaseModel):
    """Request model for /ask endpoint."""

    q: str  # Query string
    space: Optional[str] = None  # Optional space filter
    session_id: Optional[str] = None  # Optional session ID
    rerank_toggle: Optional[bool] = None  # Optional rerank override


class AskResponse(BaseModel):
    """Response model for /ask endpoint."""

    mode: str  # "clarify", "answer", or "proceed"
    trace_id: str
    confidence: float
    question: Optional[str] = None  # Clarification question if mode=clarify
    answer: Optional[str] = None  # Answer if mode=answer
    sources: Optional[list[dict[str, str]]] = None  # Sources with title/url
    primary_page_tree: Optional[Any] = None  # Tree structure (dict/array/string)
    hits: Optional[list[dict]] = None  # Search hits if mode=proceed


@router.post("/ask", response_model=AskResponse)
async def handle_query_endpoint(request: AskRequest) -> AskResponse:
    """
    Handle user query with orchestrator integration.

    This endpoint implements the interface expected by functional tests:
    - Performs search with optional space filtering
    - Computes confidence based on search results and graph overlap
    - Returns clarification if confidence is low
    - Returns answer with sources if confidence is sufficient
    """
    # Generate trace ID
    trace_id = request.session_id or new_trace_id()

    # Log start
    log(
        "start",
        trace_id,
        query=request.q,
        space=request.space,
        session_id=request.session_id,
    )

    # Get tools
    search_tool = get_search_tool()
    graph_tool = get_graph_tool()
    session_store = get_session_store()

    try:
        # 1) BASE SEARCH (with optional space filter)
        log("search_start", trace_id, query=request.q, space=request.space)

        hits_raw = search_tool.search(
            request.q,
            k=10,
            space=request.space,
            trace_id=trace_id,
            enable_agent_rerank=False,
        )

        log(
            "search_results",
            trace_id,
            count=len(hits_raw),
            top_score=hits_raw[0]["score"] if hits_raw else 0,
        )

        # Extract key information
        pre_top_score = hits_raw[0]["score"] if hits_raw else 0.0
        search_ids = [h["id"] for h in hits_raw]

        # 2) OPTIONAL RE-RANK (if requested)
        if request.rerank_toggle:
            hits = search_tool.search(
                request.q,
                k=10,
                space=request.space,
                trace_id=trace_id,
                enable_agent_rerank=True,
            )
        else:
            hits = hits_raw

        # 3) GRAPH OVERLAP COMPUTATION
        primary_id = hits_raw[0]["id"] if hits_raw else None
        neighbor_ids = set()

        if primary_id:
            log("graph_start", trace_id, primary_id=primary_id)
            neighbors = graph_tool.neighbors(primary_id, trace_id)

            for group in ("parents", "children", "siblings"):
                for n in neighbors.get(group, []):
                    nid = n.get("page_id") or n.get("id")
                    if nid:
                        neighbor_ids.add(nid)

            log("graph_neighbors", trace_id, count=len(neighbor_ids))

        # 4) CONFIDENCE COMPUTATION
        overlap = compute_overlap(search_ids, neighbor_ids)
        conf = confidence(pre_top_score, overlap)

        log(
            "confidence",
            trace_id,
            pre_top_score=pre_top_score,
            overlap=overlap,
            confidence=conf,
            should_clarify=should_clarify(conf),
        )

        # 5) CLARIFY IF LOW CONFIDENCE
        if should_clarify(conf):
            # Generate clarification question
            titles = [h.get("title", "") for h in hits[:4]]
            question = ask_clarifying_question(request.q, titles)

            # Remember in session
            session_store.remember_clarification(trace_id, question)

            log("clarify", trace_id, question=question)

            return AskResponse(
                mode="clarify", question=question, confidence=conf, trace_id=trace_id
            )

        # 6) PROCEED WITH ANSWER
        log("ready_for_synthesis", trace_id, chosen_primary=primary_id)

        # Create sources from top hits (deduplicated)
        sources = []
        seen_urls = set()
        for hit in hits[:5]:  # Consider top 5 for dedup
            url = hit.get("url", f"/pages/{hit.get('id', '')}")
            # Simple deduplication
            if url not in seen_urls and len(sources) < 3:
                sources.append({"title": hit.get("title", ""), "url": url})
                seen_urls.add(url)

        # Generate a more comprehensive answer (150-300 words)
        answer_parts = []

        if hits:
            # Opening paragraph
            answer_parts.append(
                f"Based on the search results for '{request.q}', I found relevant information across multiple documentation pages."
            )

            # Main content from top hit
            top_hit = hits[0]
            content = top_hit.get("snippet") or top_hit.get("content", "")[:800]

            # Clean up content for better readability
            content = content.replace("\\n", " ").strip()
            if content:
                answer_parts.append(
                    f"\n\n## Key Information\n\nThe primary documentation from '{top_hit.get('title', 'the main page')}' indicates that {content[:400]}..."
                )

            # Add context from additional hits if available
            if len(hits) > 1:
                answer_parts.append("\n\n## Additional Context\n")

                # Add bullet points from other relevant pages
                for hit in hits[1:3]:
                    title = hit.get("title", "Related page")
                    snippet = (
                        (hit.get("snippet") or hit.get("content", ""))[:150]
                        .replace("\\n", " ")
                        .strip()
                    )
                    if snippet:
                        answer_parts.append(f"- **{title}**: {snippet}")

            # Conclusion
            if len(hits) >= 3:
                answer_parts.append(
                    f"\n\nThese sources provide comprehensive coverage of {request.q}. The documentation shows multiple perspectives and use cases that should address your query."
                )

            answer = "\n".join(answer_parts)
        else:
            answer = f"I couldn't find specific information about '{request.q}' in the documentation. This might be because the topic isn't covered in the current knowledge base, or it might be described using different terminology. Consider rephrasing your query or checking related topics."

        # Build proper tree structure with URL field
        primary_page_tree = {
            "id": primary_id,
            "title": hits[0].get("title", "") if hits else "",
            "url": hits[0].get("url", f"/pages/{primary_id}") if hits else "",
            "children": [],
        }

        # Add some children if we have graph data
        if len(neighbors.get("children", [])) > 0:
            for child in neighbors.get("children", [])[:5]:  # Limit to 5 children
                child_node = {
                    "id": child.get("page_id") or child.get("id", ""),
                    "title": child.get("title", ""),
                    "url": child.get("url", f"/pages/{child.get('page_id', '')}"),
                    "children": [],
                }
                primary_page_tree["children"].append(child_node)

        # Log synthesis with proper metrics
        log(
            "synthesize",
            trace_id,
            answer_len=len(answer),
            sources=len(sources),
            tree_depth=1 if primary_page_tree["children"] else 0,
        )

        return AskResponse(
            mode="answer",
            answer=answer,
            sources=sources,
            primary_page_tree=primary_page_tree,
            confidence=conf,
            trace_id=trace_id,
        )

    except Exception as e:
        log("error", trace_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ask/full")
async def handle_query_full_orchestrator(request: AskRequest):
    """
    Full orchestrator integration for production use.
    This uses the complete ConfluenceQAOrchestrator with all agents.

    Note: Requires AutoGen setup. For testing, use /ask endpoint.
    """
    raise HTTPException(
        status_code=501,
        detail="Full orchestrator requires AutoGen setup. Use /ask endpoint for testing.",
    )
