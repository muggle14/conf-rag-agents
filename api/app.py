# api/app.py
from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.trace_stream_otel import attach_otel_sse_processor
from api.trace_stream_otel import router as trace_router
from src.orchestrator import handle_query
from tracing.autogen_tracer import log

# Create FastAPI app directly
app = FastAPI(title="Conf RAG Agents", version="0.1.0")

# CORS middleware (tighten in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Attach OTEL SSE processor and include router
attach_otel_sse_processor()
app.include_router(trace_router, prefix="/api")


@app.get("/api/health")
def health():
    """Health check endpoint."""
    return {"ok": True}


@app.post("/api/ask")
def ask(payload: dict = Body(...)):
    """Main ask endpoint that integrates with orchestrator."""
    q = payload.get("q")
    if not q:
        raise HTTPException(400, "Missing 'q'")

    space = payload.get("space")
    session_id = payload.get("session_id")
    rerank = payload.get("rerank")  # optional boolean

    return handle_query(q, space, session_id, rerank_toggle=rerank)


@app.post("/api/feedback")
def feedback(payload: dict = Body(...)):
    """Feedback endpoint for trace quality."""
    tid = payload.get("trace_id")
    verdict = payload.get("verdict")

    if not tid or not verdict:
        raise HTTPException(400, "trace_id and verdict are required")

    log(
        "feedback",
        tid,
        verdict=verdict,
        notes=payload.get("notes"),
        better_doc_ids=payload.get("better_doc_ids", []),
    )

    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
