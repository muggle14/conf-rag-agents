# tracing/autogen_tracer.py
"""
AutoGen-compatible tracer wrapper that maintains backward compatibility
while using OpenTelemetry under the hood.
"""

import uuid
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from tracing.otel_config import create_span_attributes, get_tracer

# Get the tracer for this module
_tracer = get_tracer("confluence_qa.agents")


class AutoGenTracer:
    """
    Wrapper around OpenTelemetry tracer for AutoGen agents.
    Maintains backward compatibility with existing trace_id pattern.
    """

    def __init__(self, tracer: Optional[trace.Tracer] = None):
        """Initialize with an optional custom tracer."""
        self.tracer = tracer or _tracer
        self._active_spans = {}

    def new_trace_id(self) -> str:
        """
        Generate a new trace ID (backward compatible).
        Returns a hex string that can be used as both trace_id and span_id.
        """
        return uuid.uuid4().hex

    @contextmanager
    def start_span(self, name: str, trace_id: Optional[str] = None, **attributes):
        """
        Context manager for creating spans with AutoGen-style attributes.

        Args:
            name: Name of the span (e.g., "search_start", "agent_invoke")
            trace_id: Optional trace ID for correlation
            **attributes: Additional span attributes

        Yields:
            The active span
        """
        # Create span attributes
        span_attributes = create_span_attributes(**attributes)

        # Add trace_id if provided
        if trace_id:
            span_attributes["trace.id"] = trace_id

        # Start the span
        with self.tracer.start_as_current_span(
            name, attributes=span_attributes
        ) as span:
            # Store span for potential updates
            if trace_id:
                self._active_spans[trace_id] = span

            try:
                yield span
            except Exception as e:
                # Record exception in span
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
            finally:
                # Clean up stored span
                if trace_id and trace_id in self._active_spans:
                    del self._active_spans[trace_id]

    def log(self, step: str, trace_id: str, **payload) -> Dict[str, Any]:
        """
        Backward compatible logging method that creates spans.

        Args:
            step: Step name (becomes span name)
            trace_id: Trace identifier
            **payload: Additional data (becomes span attributes)

        Returns:
            Event dictionary (for backward compatibility)
        """
        import datetime

        # Create a span for this log event
        with self.start_span(step, trace_id=trace_id, **payload):
            pass

        # Return backward-compatible event structure
        return {
            "trace_id": trace_id,
            "step": step,
            "ts": datetime.datetime.utcnow().isoformat() + "Z",
            "payload": payload,
        }

    def add_event(
        self, trace_id: str, name: str, attributes: Optional[Dict[str, Any]] = None
    ):
        """
        Add an event to an existing span.

        Args:
            trace_id: Trace identifier
            name: Event name
            attributes: Event attributes
        """
        span = self._active_spans.get(trace_id)
        if span:
            span.add_event(name, attributes or {})

    def set_attribute(self, trace_id: str, key: str, value: Any):
        """
        Set an attribute on an existing span.

        Args:
            trace_id: Trace identifier
            key: Attribute key
            value: Attribute value
        """
        span = self._active_spans.get(trace_id)
        if span:
            span.set_attribute(key, value)

    @contextmanager
    def trace_agent(self, agent_name: str, operation: str = "invoke", **attributes):
        """
        Trace an AutoGen agent operation.

        Args:
            agent_name: Name of the agent
            operation: Operation being performed
            **attributes: Additional attributes

        Yields:
            The span and trace_id tuple
        """
        trace_id = self.new_trace_id()
        span_name = f"agent.{agent_name}.{operation}"

        with self.start_span(
            span_name,
            trace_id=trace_id,
            agent_name=agent_name,
            operation=operation,
            **attributes,
        ) as span:
            yield span, trace_id

    @contextmanager
    def trace_tool(self, tool_name: str, **attributes):
        """
        Trace a tool execution.

        Args:
            tool_name: Name of the tool
            **attributes: Tool parameters and metadata

        Yields:
            The span and trace_id tuple
        """
        trace_id = self.new_trace_id()
        span_name = f"tool.{tool_name}"

        with self.start_span(
            span_name, trace_id=trace_id, tool_name=tool_name, **attributes
        ) as span:
            yield span, trace_id

    def trace_function(self, name: Optional[str] = None) -> Callable:
        """
        Decorator for tracing functions.

        Args:
            name: Optional span name (defaults to function name)

        Returns:
            Decorated function
        """

        def decorator(func: Callable) -> Callable:
            span_name = name or f"function.{func.__name__}"

            def wrapper(*args, **kwargs):
                with self.start_span(
                    span_name, function=func.__name__, module=func.__module__
                ):
                    return func(*args, **kwargs)

            return wrapper

        return decorator


# Global tracer instance
tracer = AutoGenTracer()

# Export convenience functions for backward compatibility
new_trace_id = tracer.new_trace_id
log = tracer.log
start_span = tracer.start_span
trace_agent = tracer.trace_agent
trace_tool = tracer.trace_tool
trace_function = tracer.trace_function
