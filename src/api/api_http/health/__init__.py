"""
HTTP GET /api/health
===================
Health check endpoint for monitoring system status.
"""

from __future__ import annotations

import json
from datetime import datetime

import azure.functions as func

# Import the shared FunctionApp instance
from .. import app


@app.function_name(name="health")
@app.route(route="health", methods=["GET"], auth_level="function")
async def health(req: func.HttpRequest) -> func.HttpResponse:
    """
    Health check endpoint for monitoring system status.
    """
    try:
        # TODO: Add actual health checks for Azure services
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "services": {
                "azure_search": "healthy",
                "cosmos_db": "healthy",
                "openai": "healthy",
                "storage": "healthy",
            },
        }

        return func.HttpResponse(json.dumps(health_status), mimetype="application/json")

    except Exception as e:
        error_status = {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e),
        }
        return func.HttpResponse(
            json.dumps(error_status), status_code=500, mimetype="application/json"
        )
