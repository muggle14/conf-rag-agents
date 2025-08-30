"""
HTTP POST /api/ask
==================
Body:
    {
      "question":               string (required)
      "conversation_id":        string (optional) – if absent, server creates one
      "include_thinking_process": boolean (optional, default false)
    }

Returns JSON:
    {
      "conversation_id": "...",
      "answer": "...",
      "page_tree": "markdown …",        # document hierarchy with ⭐ sources
      "thinking_process": [ ... ],      # only when client asked for it
      "time_sec": 1.43
    }
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Dict

import azure.functions as func
from autogen_tools.azure_search import AzureAISearchTool

# project imports - Updated to match actual module structure
from agents.router import agent  # QAAgent singleton
from utils.conversation import append_msg
from utils.page_tree import build_tree

# shared FunctionApp instance
from .. import app

# ---------- logger ----------
log = logging.getLogger("confluence-qa.ask-endpoint")

# ---------- search tool (singleton) ----------
_search = AzureAISearchTool()


# --------------------------------------------------------------------------- #
#  Function endpoint                                                           #
# --------------------------------------------------------------------------- #
@app.function_name(name="ask")
@app.route(route="ask", methods=["POST"], auth_level="function")
async def ask(req: func.HttpRequest) -> func.HttpResponse:
    # ─── Parse body ────────────────────────────────────────────────
    try:
        body: Dict[str, Any] = json.loads(req.get_body())
    except ValueError:
        return func.HttpResponse("❌ Invalid JSON", status_code=400)

    question: str | None = body.get("question") or body.get("query")
    if not question:
        return func.HttpResponse("❌ `question` is required", status_code=400)

    conversation_id: str = body.get("conversation_id") or str(uuid.uuid4())
    include_thinking_process: bool = bool(body.get("include_thinking_process", False))

    # ------------- store user message first (fire-and-forget) ---------------
    append_msg(conversation_id, role="user", content=question)

    start_time = time.time()
    # ------------- Autogen ---------------------------------------------------
    answer = await agent.ask(question)  # main LLM answer
    elapsed_time = round(time.time() - start_time, 2)

    # ------------- store assistant reply ------------------------------------
    append_msg(conversation_id, role="assistant", content=answer)

    # ------------- derive page tree -----------------------------------------
    search_results = _search.raw_results(question, top=10)  # reuse hybrid search
    source_page_ids = {
        doc["page_id"] for doc in search_results[:3]
    }  # highlight best 3 using page_id
    page_tree_markdown = build_tree(search_results, source_page_ids)

    # ------------- optional thinking trace ----------------------------------
    thinking_process = None
    if include_thinking_process:
        thinking_process = list(agent.get_thoughts(limit=50))  # Autogen API

    # ------------- final payload --------------------------------------------
    response = {
        "conversation_id": conversation_id,
        "answer": answer,
        "page_tree": page_tree_markdown,
        "time_sec": elapsed_time,
    }
    if thinking_process is not None:
        response["thinking_process"] = thinking_process

    log.info(f"Ask endpoint: Completed in {elapsed_time}s")
    return func.HttpResponse(json.dumps(response), mimetype="application/json")


# --------------------------------------------------------------------------- #
#  Function endpoint                                                           #
# --------------------------------------------------------------------------- #
# Get your function key:
# az functionapp function keys list \
#       --function-name ask \
#       --resource-group <rg> \
#       --name <func-app> \
#       --query "default" -o tsv > KEY.txt

# curl -X POST "https://<func-app>.azurewebsites.net/api/ask" \
#      -H "x-functions-key: $(cat KEY.txt)" \
#      -H "Content-Type: application/json" \
#      -d '{"question":"How do I reset my VPN token?"}'
# --------------------------------------------------------------------------- #
# output
# {
#   "conversation_id": "e2c7d4d5-6a0e-4dd4-93c6-c0746a5b9e4d",
#   "answer": "To reset your VPN token, open the portal …",
#   "page_tree": "- [VPN](...) \n  - [Reset VPN Token](...) ⭐",
#   "time_sec": 1.43
# }
# --------------------------------------------------------------------------- #
