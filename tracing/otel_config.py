# tracing/otel_config.py
"""
OpenTelemetry configuration for AutoGen agents with Azure Monitor integration.
This module sets up distributed tracing compatible with AutoGen's telemetry framework.
"""

import os
from typing import Optional

from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
from dotenv import load_dotenv
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Load environment variables
load_dotenv()

# Service configuration
SERVICE_NAME = "confluence-qa-agents"
SERVICE_VERSION = "1.0.0"


def configure_autogen_telemetry(
    service_name: str = SERVICE_NAME,
    use_azure_monitor: bool = True,
    use_otlp: bool = False,
    otlp_endpoint: Optional[str] = None,
    azure_connection_string: Optional[str] = None,
) -> TracerProvider:
    """
    Configure OpenTelemetry for AutoGen agents with multiple exporters.

    Args:
        service_name: Name of the service for telemetry
        use_azure_monitor: Enable Azure Application Insights exporter
        use_otlp: Enable OTLP exporter (for Jaeger/Zipkin)
        otlp_endpoint: OTLP collector endpoint (default: localhost:4317)
        azure_connection_string: Azure Monitor connection string

    Returns:
        Configured TracerProvider for AutoGen runtime
    """

    # Check if telemetry is disabled
    if os.getenv("AUTOGEN_DISABLE_RUNTIME_TRACING", "").lower() == "true":
        from opentelemetry.sdk.trace import NoOpTracerProvider

        return NoOpTracerProvider()

    # Create resource with service information
    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": SERVICE_VERSION,
            "service.instance.id": os.getenv("HOSTNAME", "local"),
            "deployment.environment": os.getenv("ENVIRONMENT", "development"),
        }
    )

    # Create tracer provider
    tracer_provider = TracerProvider(resource=resource)

    # Configure Azure Monitor exporter
    if use_azure_monitor:
        connection_string = azure_connection_string or os.getenv(
            "APPLICATIONINSIGHTS_CONNECTION_STRING"
        )
        if connection_string:
            try:
                azure_exporter = AzureMonitorTraceExporter(
                    connection_string=connection_string
                )
                tracer_provider.add_span_processor(BatchSpanProcessor(azure_exporter))
                print(f"✅ Azure Monitor telemetry configured for {service_name}")
            except Exception as e:
                print(f"⚠️  Failed to configure Azure Monitor: {e}")
        else:
            print("⚠️  Azure Monitor connection string not found")

    # Configure OTLP exporter (for Jaeger/Zipkin)
    if use_otlp:
        otlp_endpoint = otlp_endpoint or os.getenv("OTLP_ENDPOINT", "localhost:4317")
        try:
            otlp_exporter = OTLPSpanExporter(
                endpoint=otlp_endpoint,
                insecure=True,  # Use TLS in production
            )
            tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            print(f"✅ OTLP telemetry configured (endpoint: {otlp_endpoint})")
        except Exception as e:
            print(f"⚠️  Failed to configure OTLP exporter: {e}")

    # Set as global tracer provider
    trace.set_tracer_provider(tracer_provider)

    # Instrument OpenAI calls automatically
    try:
        OpenAIInstrumentor().instrument()
        print("✅ OpenAI instrumentation enabled")
    except Exception as e:
        print(f"⚠️  Failed to instrument OpenAI: {e}")

    return tracer_provider


def get_tracer(name: Optional[str] = None) -> trace.Tracer:
    """
    Get a tracer instance for creating spans.

    Args:
        name: Name of the tracer (defaults to module name)

    Returns:
        OpenTelemetry Tracer instance
    """
    return trace.get_tracer(name or __name__, SERVICE_VERSION)


def create_span_attributes(**kwargs) -> dict:
    """
    Create standardized span attributes for AutoGen agents.

    Args:
        **kwargs: Key-value pairs to include as span attributes

    Returns:
        Dictionary of span attributes with proper naming conventions
    """
    attributes = {}

    # Add AutoGen-specific attributes
    for key, value in kwargs.items():
        # Convert to OpenTelemetry semantic conventions
        if key == "query":
            attributes["gen_ai.prompt"] = value
        elif key == "agent_name":
            attributes["agent.name"] = value
        elif key == "tool_name":
            attributes["tool.name"] = value
        elif key == "result_count":
            attributes["search.result_count"] = value
        elif key == "search_mode":
            attributes["search.mode"] = value
        elif key == "index_name":
            attributes["db.name"] = value
        else:
            # Custom attributes with proper namespace
            attributes[f"confluence_qa.{key}"] = value

    return attributes


# Initialize telemetry on module import
_tracer_provider = None


def init():
    """Initialize telemetry configuration."""
    global _tracer_provider
    if _tracer_provider is None:
        _tracer_provider = configure_autogen_telemetry()
    return _tracer_provider


# Auto-initialize if not disabled
if os.getenv("AUTOGEN_DISABLE_RUNTIME_TRACING", "").lower() != "true":
    init()
