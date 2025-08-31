# api/trace_stream.py
import asyncio
import json
import logging
import queue
import time

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

router = APIRouter()
_bus = queue.Queue()


class TraceQueueHandler(logging.Handler):
    def emit(self, record):
        try:
            data = json.loads(record.getMessage())
            _bus.put_nowait(data)
        except Exception:
            pass


# Hook once (from your FastAPI startup)
def attach_trace_handler():
    h = TraceQueueHandler()
    log = logging.getLogger("trace")
    log.addHandler(h)
    log.setLevel(logging.INFO)


@router.get("/{trace_id}")
async def stream_trace(trace_id: str, request: Request):
    async def gen():
        last_ping = time.time()

        while True:
            # Check if client disconnected
            if await request.is_disconnected():
                break

            try:
                # Non-blocking get from queue
                evt = _bus.get_nowait()
                if evt.get("trace_id") == trace_id:
                    yield f"data: {json.dumps(evt)}\n\n"
                    last_ping = time.time()
                else:
                    # Put back events for other trace_ids
                    _bus.put_nowait(evt)
            except queue.Empty:
                # Send ping every 15 seconds to keep connection alive
                if time.time() - last_ping > 15:
                    yield ": ping\n\n"
                    last_ping = time.time()
                # Small async sleep to prevent CPU spinning
                await asyncio.sleep(0.1)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable buffering for nginx
        },
    )
