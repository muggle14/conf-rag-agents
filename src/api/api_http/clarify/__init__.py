from __future__ import annotations

import json
from typing import Any, Dict

import azure.functions as func

from agents.router import agent
from utils.conversation import append_msg
from utils.logging import log_step

from .. import app


@app.function_name(name="clarify")
@app.route(route="clarify/{conversation_id}", methods=["POST"], auth_level="function")
async def clarify(req: func.HttpRequest) -> func.HttpResponse:
    conv_id = req.route_params.get("conversation_id")
    try:
        body: Dict[str, Any] = json.loads(req.get_body())
        clar = body["clarification"]
    except (ValueError, KeyError):
        return func.HttpResponse("❌ JSON must contain `clarification`", 400)

    # Save the clarifying message
    append_msg(conv_id, role="user", content=clar)

    # Re-ask the agent in the same thread
    answer = await agent.ask(clar)

    append_msg(conv_id, role="assistant", content=answer)
    log_step("clarify-endpoint", "reply", f"cid={conv_id}", "ok")
    return func.HttpResponse(
        json.dumps({"conversation_id": conv_id, "answer": answer}),
        mimetype="application/json",
    )
