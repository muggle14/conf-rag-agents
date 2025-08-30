# api/trace_stream_otel.py
"""
SSE (Server-Sent Events) bridge for streaming OpenTelemetry spans to UI.
This module converts OpenTelemetry spans to SSE-compatible format for real-time UI updates.
"""

import json
import queue
import threading
from typing import Any, Dict

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor

router = APIRouter()

# Global queue for span events
_span_queue = queue.Queue(maxsize=1000)

# Store active trace subscriptions
_trace_subscriptions: Dict[str, queue.Queue] = {}
_subscription_lock = threading.Lock()


class SSESpanProcessor(SpanProcessor):
    """
    Custom span processor that publishes spans to SSE queues.
    """

    def on_start(self, span: ReadableSpan, parent_context=None) -> None:
        """Called when a span starts."""
        self._publish_span_event(span, "start")

    def on_end(self, span: ReadableSpan) -> None:
        """Called when a span ends."""
        self._publish_span_event(span, "end")

    def _publish_span_event(self, span: ReadableSpan, event_type: str) -> None:
        """Convert span to SSE-compatible event and publish to queues."""

        # Extract trace ID from span context
        span_context = span.get_span_context()
        trace_id = format(span_context.trace_id, "032x")
        span_id = format(span_context.span_id, "016x")

        # Get custom trace_id from attributes if available
        attributes = dict(span.attributes or {})
        custom_trace_id = attributes.get("trace.id")

        # Build event data
        event_data = {
            "trace_id": custom_trace_id or trace_id,
            "span_id": span_id,
            "parent_span_id": (
                format(span.parent.span_id, "016x") if span.parent else None
            ),
            "name": span.name,
            "event": event_type,
            "timestamp": span.start_time if event_type == "start" else span.end_time,
            "attributes": attributes,
            "status": span.status.status_code.name if span.status else None,
        }

        # Add to main queue
        try:
            _span_queue.put_nowait(event_data)
        except queue.Full:
            # Drop oldest event if queue is full
            try:
                _span_queue.get_nowait()
                _span_queue.put_nowait(event_data)
            except:
                pass

        # Publish to trace-specific subscriptions
        with _subscription_lock:
            trace_ids = [custom_trace_id, trace_id] if custom_trace_id else [trace_id]
            for tid in trace_ids:
                if tid in _trace_subscriptions:
                    sub_queue = _trace_subscriptions[tid]
                    try:
                        sub_queue.put_nowait(event_data)
                    except queue.Full:
                        pass

    def shutdown(self) -> None:
        """Clean up resources."""
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush any pending spans."""
        return True


def attach_otel_sse_processor():
    """
    Attach the SSE span processor to the current tracer provider.
    Must be called during application startup.
    """
    tracer_provider = trace.get_tracer_provider()
    if hasattr(tracer_provider, "add_span_processor"):
        processor = SSESpanProcessor()
        tracer_provider.add_span_processor(processor)
        print("✅ OpenTelemetry SSE processor attached")
    else:
        print("⚠️  Could not attach SSE processor - tracer provider doesn't support it")


@router.get("/trace/{trace_id}")
async def stream_trace_spans(trace_id: str):
    """
    Stream OpenTelemetry spans for a specific trace ID via SSE.

    Args:
        trace_id: The trace ID to filter spans

    Returns:
        SSE stream of span events
    """

    # Create a subscription queue for this trace
    sub_queue = queue.Queue(maxsize=100)

    with _subscription_lock:
        _trace_subscriptions[trace_id] = sub_queue

    def generate_events():
        """Generate SSE events from the subscription queue."""
        try:
            while True:
                try:
                    # Wait for events with timeout
                    event_data = sub_queue.get(timeout=30)

                    # Convert to SSE format
                    sse_event = {
                        "trace_id": event_data["trace_id"],
                        "step": event_data["name"],
                        "event": event_data["event"],
                        "span_id": event_data["span_id"],
                        "timestamp": event_data["timestamp"],
                        "payload": event_data["attributes"],
                    }

                    yield f"data: {json.dumps(sse_event)}\n\n"

                except queue.Empty:
                    # Send keep-alive ping
                    yield ": ping\n\n"

                except Exception as e:
                    # Send error event
                    error_event = {"error": str(e), "trace_id": trace_id}
                    yield f"data: {json.dumps(error_event)}\n\n"
                    break

        finally:
            # Clean up subscription
            with _subscription_lock:
                if trace_id in _trace_subscriptions:
                    del _trace_subscriptions[trace_id]

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/trace")
async def stream_all_spans():
    """
    Stream all OpenTelemetry spans (unfiltered) via SSE.
    Useful for debugging and monitoring.

    Returns:
        SSE stream of all span events
    """

    def generate_all_events():
        """Generate SSE events from the main queue."""
        while True:
            try:
                # Get event with timeout
                event_data = _span_queue.get(timeout=30)

                # Convert to SSE format
                sse_event = {
                    "trace_id": event_data["trace_id"],
                    "step": event_data["name"],
                    "event": event_data["event"],
                    "span_id": event_data["span_id"],
                    "timestamp": event_data["timestamp"],
                    "payload": event_data["attributes"],
                }

                yield f"data: {json.dumps(sse_event)}\n\n"

            except queue.Empty:
                # Send keep-alive ping
                yield ": ping\n\n"

    return StreamingResponse(
        generate_all_events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Backward compatibility: Support legacy log format
def convert_log_to_span_event(log_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert legacy log format to span event format.

    Args:
        log_data: Legacy log data with trace_id, step, payload

    Returns:
        Span event in SSE format
    """
    return {
        "trace_id": log_data.get("trace_id"),
        "step": log_data.get("step"),
        "event": "log",
        "timestamp": log_data.get("ts"),
        "payload": log_data.get("payload", {}),
    }
