"""
Azure OpenAI GPT-4o based ReRankerAgent for search result reordering.
"""

from typing import Dict, List, Optional

from tracing import autogen_tracer as trace_logger

from ..llm.wrapper import call_json

# System prompt for the reranker
SYSTEM = (
    "You are ReRankerAgent using GPT-4o. Score each candidate for how well it answers the query. "
    'Return STRICT JSON: {"items":[{"id":"...","score":0..1}, ...]}. No extra text.'
)

# User prompt template
USER_TMPL = """Query:
{query}

Candidates:
{candidates}

Instructions:
- Judge strict topical relevance to the query (0=irrelevant, 1=direct answer).
- Consider title and snippet only (ignore URL).
- Score each candidate independently.
- Output STRICT JSON: {{"items":[{{"id":"...","score":float}}, ...]}}
"""


def agent_rerank(
    query: str, hits: List[Dict], maxk: int = 8, trace_id: Optional[str] = None
) -> List[Dict]:
    """
    Rerank search results using GPT-4o based on relevance to query.

    Args:
        query: Search query string
        hits: List of search hits to rerank
        maxk: Maximum number of hits to rerank (default 8)
        trace_id: Optional trace ID for logging

    Returns:
        List of hits reordered by relevance score
    """
    # Use OpenTelemetry span for tracing
    with trace_logger.trace_tool(
        "agent_rerank", query=query, hits_count=len(hits), maxk=maxk
    ) as (span, auto_trace_id):
        # Use provided trace_id or auto-generated one
        tid = trace_id or auto_trace_id

        # If no hits, return empty list
        if not hits:
            span.set_attribute("rerank.skipped", "no_hits")
            return hits

        # Prepare candidates for scoring (limit to maxk)
        candidates = []
        for h in hits[:maxk]:
            candidates.append(
                {
                    "id": h.get("id", "unknown"),
                    "title": h.get("title", "No Title"),
                    "snippet": h.get("snippet", h.get("content", "")[:500]),
                }
            )

        # Create prompt for GPT-4o
        prompt = USER_TMPL.format(
            query=query,
            candidates="\n".join(
                [
                    f"- id={c['id']} | {c['title']}\n  {c['snippet'][:500]}"
                    for c in candidates
                ]
            ),
        )

        span.add_event(
            "calling_llm", {"model": "gpt-4o", "candidates": len(candidates)}
        )

        try:
            # Call GPT-4o for reranking
            output = call_json(prompt, system=SYSTEM, max_tokens=1200)

            # Extract scores from response
            id2score = {}
            items = output.get("items", [])

            for item in items:
                if "id" in item and "score" in item:
                    try:
                        score = float(item["score"])
                        # Clamp score between 0 and 1
                        score = max(0.0, min(1.0, score))
                        id2score[item["id"]] = score
                    except (ValueError, TypeError):
                        continue

            span.set_attribute("rerank.scores_extracted", len(id2score))

            # If no valid scores, return original order
            if not id2score:
                span.set_attribute("rerank.fallback", "no_valid_scores")
                return hits

            # Create enriched results with agent scores
            enriched = []

            # First add all scored items
            for h in hits:
                h2 = dict(h)
                doc_id = h.get("id", "unknown")

                if doc_id in id2score:
                    h2["agent_score"] = id2score[doc_id]
                    h2["reranked"] = True
                else:
                    # Items beyond maxk or without scores get a low default
                    h2["agent_score"] = 0.0
                    h2["reranked"] = False

                enriched.append(h2)

            # Sort by agent_score (descending)
            enriched.sort(key=lambda x: x.get("agent_score", 0.0), reverse=True)

            span.set_attribute("rerank.completed", True)
            span.add_event(
                "rerank_complete",
                {
                    "top_id": enriched[0].get("id") if enriched else None,
                    "top_score": enriched[0].get("agent_score") if enriched else None,
                },
            )

            return enriched

        except Exception as e:
            # Log error and return original order
            span.record_exception(e)
            span.set_attribute("rerank.error", str(e))

            # Return original hits unchanged on error
            return hits


def batch_rerank(
    queries_and_hits: List[tuple[str, List[Dict]]], maxk: int = 8
) -> List[List[Dict]]:
    """
    Batch rerank multiple query-hit pairs.

    Args:
        queries_and_hits: List of (query, hits) tuples
        maxk: Maximum number of hits to rerank per query

    Returns:
        List of reranked hit lists
    """
    results = []

    for query, hits in queries_and_hits:
        reranked = agent_rerank(query, hits, maxk=maxk)
        results.append(reranked)

    return results
