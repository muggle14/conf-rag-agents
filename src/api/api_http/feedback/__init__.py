"""
HTTP POST /api/feedback
======================
Feedback endpoint for collecting user feedback on responses.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import azure.functions as func

# Import the shared FunctionApp instance
from .. import app


@app.function_name(name="feedback")
@app.route(route="feedback", methods=["POST"], auth_level="function")
async def feedback(req: func.HttpRequest) -> func.HttpResponse:
    """
    Feedback endpoint for collecting user feedback on responses.
    """
    try:
        body: Dict[str, Any] = json.loads(req.get_body())
        conversation_id = body.get("conversation_id")
        feedback_type = body.get("feedback_type")  # positive, negative, neutral
        feedback_text = body.get("feedback_text", "")

        if not conversation_id:
            return func.HttpResponse(
                "❌ `conversation_id` is required", status_code=400
            )

        if not feedback_type:
            return func.HttpResponse("❌ `feedback_type` is required", status_code=400)

        # TODO: Implement feedback collection logic
        response = {
            "conversation_id": conversation_id,
            "feedback_type": feedback_type,
            "feedback_text": feedback_text,
            "status": "success",
        }

        return func.HttpResponse(json.dumps(response), mimetype="application/json")

    except ValueError:
        return func.HttpResponse("❌ Invalid JSON", status_code=400)
    except Exception as e:
        return func.HttpResponse(f"❌ Internal error: {str(e)}", status_code=500)
