"""
Simplified orchestrator for API endpoints.
Provides handle_query function that integrates with existing components.
"""

from typing import Any, Dict, Optional

from src.agents.clarifier import ask_clarifying_question
from src.agents.logic.confidence import compute_overlap, confidence, should_clarify
from src.services import GraphService, SearchService, SessionService
from tracing.autogen_tracer import log, new_trace_id


def handle_query(
    query: str,
    space: Optional[str] = None,
    session_id: Optional[str] = None,
    rerank_toggle: Optional[bool] = None,
    *,
    search: Optional[SearchService] = None,
    graph: Optional[GraphService] = None,
    session: Optional[SessionService] = None,
) -> Dict[str, Any]:
    """
    Handle user query with confidence-based clarification or answer.

    Args:
        query: User's question
        space: Optional Confluence space filter
        session_id: Optional session ID for conversation tracking
        rerank_toggle: Optional boolean to enable/disable reranking
        search: Optional injected SearchService
        graph: Optional injected GraphService
        session: Optional injected SessionService

    Returns:
        Dictionary with mode (clarify/answer/proceed), confidence, trace_id, and relevant data
    """
    # Generate trace ID
    trace_id = session_id or new_trace_id()

    # Log start
    log("start", trace_id, query=query, space=space, session_id=session_id)

    # Initialize tools
    search_service = search or SearchService()
    graph_service = graph or GraphService()
    session_service = session or SessionService()

    try:
        # 1) BASE SEARCH (with optional space filter)
        log("search_start", trace_id, query=query, space=space)

        hits_raw = search_service.search(
            query, k=10, space=space, trace_id=trace_id, enable_agent_rerank=False
        )

        if not hits_raw:
            log("no_results", trace_id)
            return {
                "mode": "answer",
                "trace_id": trace_id,
                "confidence": 0.0,
                "answer": f"I couldn't find specific information about '{query}' in the documentation.",
                "sources": [],
                "primary_page_tree": {},
            }

        log(
            "search_results",
            trace_id,
            count=len(hits_raw),
            top_score=hits_raw[0]["score"] if hits_raw else 0,
        )

        # Extract key information
        pre_top_score = hits_raw[0]["score"] if hits_raw else 0.0
        search_ids = [h["id"] for h in hits_raw]

        # 2) OPTIONAL RE-RANK
        if rerank_toggle:
            log("rerank_start", trace_id)
            hits = search_service.search(
                query, k=10, space=space, trace_id=trace_id, enable_agent_rerank=True
            )
            log("rerank_complete", trace_id, reranked_count=len(hits))
        else:
            hits = hits_raw

        # 3) GRAPH OVERLAP COMPUTATION
        primary_id = hits_raw[0]["id"] if hits_raw else None
        neighbor_ids = set()

        if primary_id:
            log("graph_start", trace_id, primary_id=primary_id)
            neighbors = graph_service.neighbors(primary_id, trace_id=trace_id)

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
            question = ask_clarifying_question(query, titles)

            # Remember in session
            if session_id:
                session_service.remember_clarification(session_id, question)

            log("clarify", trace_id, question=question)

            return {
                "mode": "clarify",
                "question": question,
                "confidence": conf,
                "trace_id": trace_id,
            }

        # 6) PROCEED WITH ANSWER
        log("ready_for_synthesis", trace_id, chosen_primary=primary_id)

        # Create sources from top hits
        sources = []
        seen_urls = set()
        for hit in hits[:5]:
            url = hit.get("url", f"/pages/{hit.get('id', '')}")
            if url not in seen_urls and len(sources) < 3:
                sources.append({"title": hit.get("title", ""), "url": url})
                seen_urls.add(url)

        # Generate answer
        answer_parts = []
        if hits:
            answer_parts.append(
                f"Based on the search results for '{query}', I found relevant information across multiple documentation pages."
            )

            top_hit = hits[0]
            content = top_hit.get("snippet") or top_hit.get("content", "")[:800]
            content = content.replace("\\n", " ").strip()

            if content:
                answer_parts.append(
                    f"\n\n## Key Information\n\nThe primary documentation from '{top_hit.get('title', 'the main page')}' indicates that {content[:400]}..."
                )

            if len(hits) > 1:
                answer_parts.append("\n\n## Additional Context\n")
                for hit in hits[1:3]:
                    title = hit.get("title", "Related page")
                    snippet = (
                        (hit.get("snippet") or hit.get("content", ""))[:150]
                        .replace("\\n", " ")
                        .strip()
                    )
                    if snippet:
                        answer_parts.append(f"- **{title}**: {snippet}")

            if len(hits) >= 3:
                answer_parts.append(
                    f"\n\nThese sources provide comprehensive coverage of {query}. The documentation shows multiple perspectives and use cases that should address your query."
                )

            answer = "\n".join(answer_parts)
        else:
            answer = f"I couldn't find specific information about '{query}' in the documentation."

        # Build tree structure
        primary_page_tree = {
            "id": primary_id,
            "title": hits[0].get("title", "") if hits else "",
            "url": hits[0].get("url", f"/pages/{primary_id}") if hits else "",
            "children": [],
        }

        # Add children from graph if available
        if primary_id and neighbors:
            for child in neighbors.get("children", [])[:5]:
                child_node = {
                    "id": child.get("page_id") or child.get("id", ""),
                    "title": child.get("title", ""),
                    "url": child.get("url", f"/pages/{child.get('page_id', '')}"),
                    "children": [],
                }
                primary_page_tree["children"].append(child_node)

        log(
            "synthesize",
            trace_id,
            answer_len=len(answer),
            sources=len(sources),
            tree_depth=1 if primary_page_tree.get("children") else 0,
        )

        return {
            "mode": "answer",
            "answer": answer,
            "sources": sources,
            "primary_page_tree": primary_page_tree,
            "confidence": conf,
            "trace_id": trace_id,
        }

    except Exception as e:
        log("error", trace_id, error=str(e))
        return {
            "mode": "error",
            "trace_id": trace_id,
            "error": str(e),
            "confidence": 0.0,
        }
