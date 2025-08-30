"""
HTTP POST /api/search_similar
============================
Search similar endpoint for finding similar questions or content.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import azure.functions as func

# Import the shared FunctionApp instance
from .. import app


@app.function_name(name="search_similar")
@app.route(route="search_similar", methods=["POST"], auth_level="function")
async def search_similar(req: func.HttpRequest) -> func.HttpResponse:
    """
    Search similar endpoint for finding similar questions or content.
    """
    try:
        body: Dict[str, Any] = json.loads(req.get_body())
        query = body.get("query")
        limit = body.get("limit", 10)

        if not query:
            return func.HttpResponse("❌ `query` is required", status_code=400)

        # TODO: Implement similar search logic
        response = {
            "query": query,
            "limit": limit,
            "similar_results": [],
            "status": "success",
        }

        return func.HttpResponse(json.dumps(response), mimetype="application/json")

    except ValueError:
        return func.HttpResponse("❌ Invalid JSON", status_code=400)
    except Exception as e:
        return func.HttpResponse(f"❌ Internal error: {str(e)}", status_code=500)
