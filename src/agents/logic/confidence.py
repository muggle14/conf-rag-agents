# confidence.py
"""
Confidence computation logic for Confluence Q&A system.
Combines semantic search scores with graph structure overlap for robust confidence assessment.
"""

from typing import List, Set


def _scale(score: float, smax: float = 5.0) -> float:
    """
    Scale a score to 0-1 range.

    Args:
        score: Raw score value
        smax: Maximum expected score value (default 5.0 for Azure Search scores)

    Returns:
        Scaled score between 0.0 and 1.0
    """
    return max(0.0, min(score / smax, 1.0))


def compute_overlap(search_ids: List[str], neighbor_ids: Set[str]) -> float:
    """
    Compute overlap ratio between search results and graph neighbors.

    Args:
        search_ids: List of document IDs from search results
        neighbor_ids: Set of document IDs from graph neighbors

    Returns:
        Overlap ratio (0.0 to 1.0)
    """
    if not search_ids:
        return 0.0

    search_set = set(search_ids)
    intersection = search_set.intersection(neighbor_ids)
    return len(intersection) / len(search_ids)


def confidence(pre_rerank_top_score: float, overlap_ratio: float) -> float:
    """
    Compute confidence score combining semantic search score and graph structure agreement.

    Args:
        pre_rerank_top_score: Top score from Azure Search before re-ranking
        overlap_ratio: Ratio of overlap between search results and graph neighbors

    Returns:
        Combined confidence score (0.0 to 1.0)
        Formula: 60% semantic top score + 40% graph structure agreement
    """
    # Scale the semantic score
    scaled_semantic_score = _scale(pre_rerank_top_score)

    # Combine with 60% weight on semantic score, 40% on graph overlap
    combined_confidence = 0.6 * scaled_semantic_score + 0.4 * overlap_ratio

    return combined_confidence


def should_clarify(conf: float, threshold: float = 0.55) -> bool:
    """
    Determine if clarification is needed based on confidence score.

    Args:
        conf: Confidence score (0.0 to 1.0)
        threshold: Confidence threshold below which clarification is needed (default 0.55)

    Returns:
        True if confidence is below threshold and clarification should be requested
    """
    return conf < threshold


def get_confidence_level(conf: float) -> str:
    """
    Get human-readable confidence level from score.

    Args:
        conf: Confidence score (0.0 to 1.0)

    Returns:
        Confidence level string: "high", "medium", or "low"
    """
    if conf >= 0.8:
        return "high"
    elif conf >= 0.55:
        return "medium"
    else:
        return "low"


def compute_multi_hop_confidence(hop_confidences: List[float]) -> float:
    """
    Compute confidence for multi-hop queries.
    Takes the minimum confidence across hops (weakest link principle).

    Args:
        hop_confidences: List of confidence scores for each hop

    Returns:
        Overall confidence score for multi-hop query
    """
    if not hop_confidences:
        return 0.0

    # Use minimum confidence (weakest link)
    # Could also use harmonic mean for a more balanced approach
    return min(hop_confidences)


def adjust_confidence_for_coverage(
    base_confidence: float, coverage_ratio: float, coverage_weight: float = 0.2
) -> float:
    """
    Adjust confidence based on how well the answer covers the query.

    Args:
        base_confidence: Initial confidence score
        coverage_ratio: Ratio of query aspects covered by answer (0.0 to 1.0)
        coverage_weight: Weight given to coverage in final confidence (default 0.2)

    Returns:
        Adjusted confidence score
    """
    # Blend base confidence with coverage ratio
    adjusted = (
        1 - coverage_weight
    ) * base_confidence + coverage_weight * coverage_ratio
    return adjusted


class ConfidenceTracker:
    """
    Track confidence scores throughout the Q&A process.
    Useful for debugging and understanding confidence degradation.
    """

    def __init__(self):
        self.scores = []
        self.factors = {}

    def add_score(self, step: str, score: float, **factors):
        """
        Add a confidence score for a processing step.

        Args:
            step: Name of the processing step
            score: Confidence score at this step
            **factors: Contributing factors to the score
        """
        self.scores.append({"step": step, "score": score, "factors": factors})

        # Track individual factors
        for key, value in factors.items():
            if key not in self.factors:
                self.factors[key] = []
            self.factors[key].append(value)

    def get_final_confidence(self) -> float:
        """
        Get the final confidence score.

        Returns:
            Final confidence score, or 0.0 if no scores tracked
        """
        if not self.scores:
            return 0.0
        return self.scores[-1]["score"]

    def get_confidence_breakdown(self) -> dict:
        """
        Get a breakdown of confidence scores and factors.

        Returns:
            Dictionary with confidence breakdown
        """
        return {
            "final_confidence": self.get_final_confidence(),
            "confidence_level": get_confidence_level(self.get_final_confidence()),
            "scores_by_step": self.scores,
            "factor_averages": {
                key: sum(values) / len(values) if values else 0.0
                for key, values in self.factors.items()
            },
        }
