# api/app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.trace_stream import attach_trace_handler
from api.trace_stream import router as trace_router
from api.trace_stream_otel import attach_otel_sse_processor
from api.trace_stream_otel import router as otel_trace_router
from tracing import otel_config


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Conf RAG Agents",
        description="Confluence Q&A Agents with OpenTelemetry tracing",
        version="1.0.0",
    )

    # Add CORS middleware for UI access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize OpenTelemetry
    otel_config.init()

    # Attach trace handlers
    attach_trace_handler()  # Legacy JSON trace handler
    attach_otel_sse_processor()  # OpenTelemetry SSE processor

    # Include routers
    app.include_router(trace_router, prefix="/trace", tags=["Legacy Tracing"])
    app.include_router(
        otel_trace_router, prefix="/otel", tags=["OpenTelemetry Tracing"]
    )

    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "service": "Confluence RAG Agents",
            "status": "operational",
            "endpoints": {
                "legacy_trace": "/trace/{trace_id}",
                "otel_trace": "/otel/trace/{trace_id}",
                "otel_all": "/otel/trace",
                "health": "/health",
            },
        }

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "service": "Confluence RAG Agents",
            "tracing": {"legacy": "enabled", "opentelemetry": "enabled"},
        }

    @app.on_event("startup")
    async def startup_event():
        """Initialize services on startup."""
        print("🚀 Confluence RAG Agents API starting...")
        print("✅ Legacy tracing enabled at /trace/{trace_id}")
        print("✅ OpenTelemetry tracing enabled at /otel/trace/{trace_id}")
        print("📊 OpenTelemetry configured for AutoGen agents")

    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown."""
        print("👋 Shutting down Confluence RAG Agents API...")

    return app


# Create the application instance
app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
