# api/trace_stream.py
import json
import logging
import queue
import time

from fastapi import APIRouter
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


@router.get("/trace/{trace_id}")
def stream_trace(trace_id: str):
    def gen():
        while True:
            evt = _bus.get()
            if evt.get("trace_id") == trace_id:
                yield f"data: {json.dumps(evt)}\n\n"
            time.sleep(0.03)

    return StreamingResponse(gen(), media_type="text/event-stream")
