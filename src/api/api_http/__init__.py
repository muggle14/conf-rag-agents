"""
Shared FunctionApp for **all** HTTP endpoints.
Keeps cold-starts low: the first import warms every route.
"""

from __future__ import annotations

import logging
import os

import azure.functions as func
from opencensus.ext.azure.log_exporter import AzureLogHandler

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

# -------- global logging to Application Insights ----------
log = logging.getLogger("confluence-qa.http")
log.setLevel(logging.INFO)
if ikey := os.getenv("APPINSIGHTS_KEY"):
    log.addHandler(AzureLogHandler(connection_string=f"InstrumentationKey={ikey}"))

# --------─ CORS (adjust origins for prod) -----------------
app.http_middleware(
    func.CorsMiddleware(
        allow_origins=["*"],  # dev-friendly; tighten for prod
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )
)
