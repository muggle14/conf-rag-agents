from __future__ import annotations

import json

import azure.functions as func

from utils.conversation import fetch_conversation, soft_delete_conversation

from .. import app


# ---- GET conversation ------------------------------------------------------
@app.function_name(name="conversation_get")
@app.route(
    route="conversation/{conversation_id}", methods=["GET"], auth_level="function"
)
def get_conv(req: func.HttpRequest) -> func.HttpResponse:
    conv_id = req.route_params["conversation_id"]
    msgs = fetch_conversation(conv_id)
    return func.HttpResponse(json.dumps(msgs), mimetype="application/json")


# ---- DELETE conversation (soft) -------------------------------------------
@app.function_name(name="conversation_delete")
@app.route(
    route="conversation/{conversation_id}", methods=["DELETE"], auth_level="function"
)
def delete_conv(req: func.HttpRequest) -> func.HttpResponse:
    conv_id = req.route_params["conversation_id"]
    soft_delete_conversation(conv_id)
    return func.HttpResponse(
        json.dumps({"conversation_id": conv_id, "deleted": True}),
        mimetype="application/json",
    )
