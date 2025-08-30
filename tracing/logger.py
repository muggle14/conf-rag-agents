# tracing/logger.py
import datetime
import json
import logging
import uuid

TRACE_LOGGER_NAME = "trace"
TRACE = logging.getLogger(TRACE_LOGGER_NAME)
TRACE.setLevel(logging.INFO)


def new_trace_id() -> str:
    return uuid.uuid4().hex


def log(step: str, trace_id: str, **payload):
    evt = {
        "trace_id": trace_id,
        "step": step,
        "ts": datetime.datetime.utcnow().isoformat() + "Z",
        "payload": payload,
    }
    # single-line JSON for easy ingestion (App Insights, Log Analytics, etc.)
    TRACE.info(json.dumps(evt))
    return evt
