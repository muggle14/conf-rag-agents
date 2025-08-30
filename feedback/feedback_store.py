# feedback/feedback_store.py
"""
Feedback store for collecting user feedback on Q&A responses.

Uses Cosmos DB SQL API (not Gremlin) for efficient feedback collection and analysis.
Partition key: /yyyymmdd for fast daily aggregation queries.
"""

import datetime
import logging
import os
from typing import Any, Dict, List, Optional

from azure.cosmos import CosmosClient, PartitionKey, exceptions

# Configuration
COSMOS_URL = os.environ.get("COSMOS_URL", os.environ.get("COSMOS_SQL_ENDPOINT", ""))
COSMOS_KEY = os.environ.get("COSMOS_KEY", os.environ.get("COSMOS_SQL_KEY", ""))
COSMOS_DB_NAME = os.getenv("COSMOS_DB_NAME", "rag_runtime")
FEEDBACK_CN = os.getenv("FEEDBACK_CONTAINER", "feedback")

# Logging
logger = logging.getLogger(__name__)


class FeedbackStore:
    """
    Store user feedback for Q&A responses with enrichment for accuracy analysis.

    Document schema:
    {
        "id": "fb_<trace_id>",
        "trace_id": "<trace_id>",
        "session_id": "<session_id>",
        "yyyymmdd": "20250830",  # partition key
        "ts_utc": "2025-08-30T17:20:44Z",
        "verdict": "correct|partial|incorrect|needs_more_context",
        "notes": "user's free text feedback",
        "better_doc_ids": ["page-123", "page-456"],

        # Enrichment fields (from trace/orchestrator)
        "space": "<space_filter>",
        "query": "<original_query>",
        "mode": "answer|clarify|proceed",
        "confidence": 0.73,
        "reranked": false,
        "sources_used": ["doc1", "doc2"],
        "answer_length": 245
    }
    """

    def __init__(self):
        """Initialize Cosmos DB feedback store."""
        if not COSMOS_URL or not COSMOS_KEY:
            raise ValueError(
                "COSMOS_URL and COSMOS_KEY environment variables must be set"
            )

        try:
            self.client = CosmosClient(COSMOS_URL, credential=COSMOS_KEY)
            self.db = self.client.create_database_if_not_exists(COSMOS_DB_NAME)

            # Create container with partition key /yyyymmdd
            self.ct = self.db.create_container_if_not_exists(
                id=FEEDBACK_CN,
                partition_key=PartitionKey(path="/yyyymmdd"),
                # Don't specify throughput for serverless accounts
            )

            logger.info(
                f"Connected to Cosmos DB feedback store: {COSMOS_DB_NAME}/{FEEDBACK_CN}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize feedback store: {e}")
            raise

    @staticmethod
    def _yyyymmdd_now() -> str:
        """Get current date in YYYYMMDD format for partition key."""
        d = datetime.datetime.utcnow()
        return f"{d.year:04d}{d.month:02d}{d.day:02d}"

    @staticmethod
    def _iso_now() -> str:
        """Get current timestamp in ISO format."""
        return datetime.datetime.utcnow().isoformat() + "Z"

    def save(
        self,
        trace_id: str,
        verdict: str,
        notes: str = "",
        better_doc_ids: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Save feedback for a Q&A interaction.

        Args:
            trace_id: Unique trace ID from the Q&A request
            verdict: User's verdict (correct|partial|incorrect|needs_more_context)
            notes: Optional free text feedback
            better_doc_ids: Optional list of document IDs that would have been better
            session_id: Optional session ID for grouping feedback
            extra: Optional enrichment data (query, confidence, etc.)

        Returns:
            The saved feedback document
        """
        # Validate verdict
        valid_verdicts = {"correct", "partial", "incorrect", "needs_more_context"}
        if verdict not in valid_verdicts:
            raise ValueError(
                f"Invalid verdict: {verdict}. Must be one of {valid_verdicts}"
            )

        # Create document
        doc = {
            "id": f"fb_{trace_id}",
            "trace_id": trace_id,
            "session_id": session_id,
            "yyyymmdd": self._yyyymmdd_now(),
            "ts_utc": self._iso_now(),
            "verdict": verdict,
            "notes": notes or "",
            "better_doc_ids": better_doc_ids or [],
        }

        # Add enrichment fields if provided
        if extra:
            # Only add non-None fields
            for key, value in extra.items():
                if value is not None:
                    doc[key] = value

        try:
            # Upsert to handle duplicate feedback submissions
            self.ct.upsert_item(doc)
            logger.info(f"Saved feedback for trace {trace_id}: verdict={verdict}")
            return doc

        except Exception as e:
            logger.error(f"Failed to save feedback for trace {trace_id}: {e}")
            raise

    def get_by_trace(
        self, trace_id: str, days_back: int = 7
    ) -> Optional[Dict[str, Any]]:
        """
        Get feedback by trace ID.

        Args:
            trace_id: The trace ID to look up
            days_back: How many days back to search (default: 7)

        Returns:
            The feedback document if found, None otherwise
        """
        try:
            # Try today's partition first
            doc_id = f"fb_{trace_id}"
            partition_key = self._yyyymmdd_now()

            try:
                return self.ct.read_item(item=doc_id, partition_key=partition_key)
            except exceptions.CosmosResourceNotFoundError:
                # If not found in today's partition, search recent days
                for days_ago in range(1, days_back + 1):
                    date = datetime.datetime.utcnow() - datetime.timedelta(
                        days=days_ago
                    )
                    partition_key = f"{date.year:04d}{date.month:02d}{date.day:02d}"

                    try:
                        return self.ct.read_item(
                            item=doc_id, partition_key=partition_key
                        )
                    except exceptions.CosmosResourceNotFoundError:
                        continue

                return None

        except Exception as e:
            logger.error(f"Error retrieving feedback for trace {trace_id}: {e}")
            return None

    def get_daily_feedback(
        self, date: Optional[datetime.date] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all feedback for a specific date (for nightly aggregation).

        Args:
            date: The date to query (default: today)

        Returns:
            List of feedback documents for that date
        """
        if date is None:
            date = datetime.date.today()

        partition_key = f"{date.year:04d}{date.month:02d}{date.day:02d}"

        try:
            query = "SELECT * FROM c WHERE c.yyyymmdd = @partition_key ORDER BY c.ts_utc DESC"
            items = list(
                self.ct.query_items(
                    query=query,
                    parameters=[{"name": "@partition_key", "value": partition_key}],
                    enable_cross_partition_query=False,  # Single partition query
                )
            )

            logger.info(f"Retrieved {len(items)} feedback items for {partition_key}")
            return items

        except Exception as e:
            logger.error(f"Error retrieving daily feedback for {partition_key}: {e}")
            return []

    def get_misfires(
        self, date: Optional[datetime.date] = None, confidence_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Get "misfires" - low confidence answers with negative feedback.

        Args:
            date: The date to query (default: today)
            confidence_threshold: Confidence below which to consider a misfire

        Returns:
            List of misfire feedback documents
        """
        all_feedback = self.get_daily_feedback(date)

        misfires = []
        for item in all_feedback:
            # Check for negative feedback
            is_negative = item.get("verdict") in ("incorrect", "needs_more_context")

            # Check for low confidence
            confidence = item.get("confidence", 1.0)
            is_low_confidence = confidence < confidence_threshold

            # Include if negative feedback OR low confidence with partial verdict
            if is_negative or (is_low_confidence and item.get("verdict") == "partial"):
                misfires.append(
                    {
                        "trace_id": item.get("trace_id"),
                        "query": item.get("query", ""),
                        "verdict": item.get("verdict"),
                        "confidence": confidence,
                        "mode": item.get("mode", ""),
                        "notes": item.get("notes", ""),
                        "better_doc_ids": item.get("better_doc_ids", []),
                        "sources_used": item.get("sources_used", []),
                        "space": item.get("space"),
                        "ts_utc": item.get("ts_utc"),
                    }
                )

        logger.info(
            f"Found {len(misfires)} misfires out of {len(all_feedback)} feedback items"
        )
        return misfires

    def get_stats(self, date: Optional[datetime.date] = None) -> Dict[str, Any]:
        """
        Get statistics for feedback on a specific date.

        Args:
            date: The date to analyze (default: today)

        Returns:
            Dictionary with feedback statistics
        """
        all_feedback = self.get_daily_feedback(date)

        if not all_feedback:
            return {
                "date": date.isoformat() if date else datetime.date.today().isoformat(),
                "total": 0,
                "verdict_distribution": {},
                "avg_confidence": 0,
                "modes": {},
                "with_notes": 0,
                "with_better_docs": 0,
            }

        # Calculate statistics
        verdict_counts = {}
        mode_counts = {}
        total_confidence = 0
        confidence_count = 0
        with_notes = 0
        with_better_docs = 0

        for item in all_feedback:
            # Verdict distribution
            verdict = item.get("verdict", "unknown")
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

            # Mode distribution
            mode = item.get("mode", "unknown")
            mode_counts[mode] = mode_counts.get(mode, 0) + 1

            # Confidence average
            if "confidence" in item:
                total_confidence += item["confidence"]
                confidence_count += 1

            # Count feedback with notes
            if item.get("notes"):
                with_notes += 1

            # Count feedback with better doc suggestions
            if item.get("better_doc_ids"):
                with_better_docs += 1

        return {
            "date": date.isoformat() if date else datetime.date.today().isoformat(),
            "total": len(all_feedback),
            "verdict_distribution": verdict_counts,
            "avg_confidence": (
                total_confidence / confidence_count if confidence_count > 0 else 0
            ),
            "modes": mode_counts,
            "with_notes": with_notes,
            "with_better_docs": with_better_docs,
            "accuracy_rate": (
                verdict_counts.get("correct", 0) / len(all_feedback)
                if all_feedback
                else 0
            ),
        }

    def cleanup_old_feedback(self, days_to_keep: int = 90):
        """
        Clean up feedback older than specified days.

        Args:
            days_to_keep: Number of days of feedback to retain
        """
        cutoff_date = datetime.datetime.utcnow() - datetime.timedelta(days=days_to_keep)
        cutoff_yyyymmdd = (
            f"{cutoff_date.year:04d}{cutoff_date.month:02d}{cutoff_date.day:02d}"
        )

        try:
            # Query for old feedback across partitions
            query = "SELECT c.id, c.yyyymmdd FROM c WHERE c.yyyymmdd < @cutoff"
            old_items = list(
                self.ct.query_items(
                    query=query,
                    parameters=[{"name": "@cutoff", "value": cutoff_yyyymmdd}],
                    enable_cross_partition_query=True,
                )
            )

            # Delete old items
            deleted_count = 0
            for item in old_items:
                try:
                    self.ct.delete_item(item=item["id"], partition_key=item["yyyymmdd"])
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Failed to delete old feedback {item['id']}: {e}")

            logger.info(
                f"Cleaned up {deleted_count} feedback items older than {days_to_keep} days"
            )

        except Exception as e:
            logger.error(f"Error during feedback cleanup: {e}")


# Example usage
if __name__ == "__main__":
    # Initialize store
    store = FeedbackStore()

    # Save some feedback
    trace_id = "test-trace-123"
    feedback = store.save(
        trace_id=trace_id,
        verdict="partial",
        notes="The answer was mostly correct but missed pricing details",
        better_doc_ids=["page-pricing-001", "page-features-002"],
        extra={
            "query": "What are the key features and pricing?",
            "confidence": 0.45,
            "mode": "answer",
            "space": "PRODUCT",
            "reranked": True,
        },
    )
    print(f"Saved feedback: {feedback}")

    # Retrieve feedback
    retrieved = store.get_by_trace(trace_id)
    print(f"Retrieved feedback: {retrieved}")

    # Get daily stats
    stats = store.get_stats()
    print(f"Today's stats: {stats}")

    # Get misfires
    misfires = store.get_misfires()
    print(f"Misfires: {misfires}")
