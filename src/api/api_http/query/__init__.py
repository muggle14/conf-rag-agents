"""
HTTP POST /api/query
===================
Query endpoint for asking questions to the Confluence Q&A system.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict

import azure.functions as func

# project imports
from agents.router import agent
from utils.page_tree import build_tree

# Import the shared FunctionApp instance
from .. import app


@app.function_name(name="query")
@app.route(route="query", methods=["POST"], auth_level="function")
async def query(req: func.HttpRequest) -> func.HttpResponse:
    """
    Query endpoint for asking questions to the Confluence Q&A system.
    """
    try:
        body: Dict[str, Any] = json.loads(req.get_body())
        question = body.get("question") or body.get("query")

        if not question:
            return func.HttpResponse("❌ `question` is required", status_code=400)

        conversation_id = body.get("conversation_id") or str(uuid.uuid4())
        include_thinking_process = body.get("include_thinking_process", False)

        start_time = time.time()
        answer = await agent.ask(question)
        elapsed_time = round(time.time() - start_time, 2)

        response = {
            "conversation_id": conversation_id,
            "answer": answer,
            "time_sec": elapsed_time,
        }

        if include_thinking_process:
            # TODO: Implement thinking process extraction
            response["thinking_process"] = []

        return func.HttpResponse(json.dumps(response), mimetype="application/json")

    except ValueError:
        return func.HttpResponse("❌ Invalid JSON", status_code=400)
    except Exception as e:
        return func.HttpResponse(f"❌ Internal error: {str(e)}", status_code=500)
