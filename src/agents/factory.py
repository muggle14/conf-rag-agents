# api_service.py
"""
Azure Functions service for Confluence Q&A system
Provides HTTP trigger functions for the AutoGen-based Q&A orchestrator
Integrates with Azure Cognitive Search, Cosmos DB, and Azure Storage
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Any

import azure.functions as func

# Import your existing modules
from orchestrator import AzureDataStore, ConfluenceQAOrchestrator

from utils.cache import ResponseCache
from utils.citations import CitationExtractor
from utils.config import Config
from utils.metrics import MetricsCollector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Function App
app = func.FunctionApp()

# Global instances (initialized on first request)
orchestrator = None
data_store = None
metrics = None
cache = None
config = None


def initialize_services():
    """Initialize services on first request"""
    global orchestrator, data_store, metrics, cache, config

    if orchestrator is None:
        config = Config.from_env()
        orchestrator = ConfluenceQAOrchestrator()
        data_store = AzureDataStore()
        metrics = MetricsCollector()
        cache = ResponseCache(ttl_seconds=3600)
        logger.info("Services initialized successfully")


# Health Check Function
@app.function_name(name="HealthCheck")
@app.route(route="health", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
async def health_check(req: func.HttpRequest) -> func.HttpResponse:
    """Health check endpoint"""
    logging.info("Health check endpoint called")

    initialize_services()

    return func.HttpResponse(
        json.dumps(
            {
                "status": "healthy",
                "version": "1.0.0",
                "metrics": metrics.get_metrics(),
                "timestamp": datetime.utcnow().isoformat(),
            }
        ),
        mimetype="application/json",
        status_code=200,
    )


# Process Query Function
@app.function_name(name="ProcessQuery")
@app.route(route="query", methods=["POST"], auth_level=func.AuthLevel.FUNCTION)
async def process_query(req: func.HttpRequest) -> func.HttpResponse:
    """Process a user query"""
    logging.info("Process query endpoint called")

    initialize_services()
    start_time = time.time()

    try:
        req_body = req.get_json()
    except ValueError:
        return func.HttpResponse(
            json.dumps({"error": "Invalid JSON in request body"}),
            mimetype="application/json",
            status_code=400,
        )

    # Extract request parameters
    query = req_body.get("query")
    if not query:
        return func.HttpResponse(
            json.dumps({"error": "Missing required field: query"}),
            mimetype="application/json",
            status_code=400,
        )

    conversation_id = req_body.get("conversation_id") or str(uuid.uuid4())
    include_thinking_process = req_body.get("include_thinking_process", True)
    max_wait_seconds = req_body.get("max_wait_seconds", 60)

    # Check cache first
    cached_response = cache.get(query)
    if cached_response:
        metrics.record_cache_hit(True)
        cached_response["conversation_id"] = conversation_id
        cached_response["response_time"] = time.time() - start_time
        return func.HttpResponse(
            json.dumps(cached_response), mimetype="application/json", status_code=200
        )

    metrics.record_cache_hit(False)

    try:
        # Process query with timeout
        result = await asyncio.wait_for(
            orchestrator.process_query(query, conversation_id), timeout=max_wait_seconds
        )

        response_time = time.time() - start_time

        # Handle different response types
        if result["status"] == "needs_clarification":
            metrics.record_clarification()
            return func.HttpResponse(
                json.dumps(
                    {
                        "status": "needs_clarification",
                        "clarification_message": result["clarification_message"],
                        "suggestions": result.get("suggestions", []),
                        "original_query": query,
                        "conversation_id": conversation_id,
                    }
                ),
                mimetype="application/json",
                status_code=200,
            )

        # Extract citations from answer
        citations = CitationExtractor.extract_citations(result["answer"])

        # Prepare successful response
        response_data = {
            "status": "success",
            "answer": result["answer"],
            "confidence": result.get(
                "confidence", result["verification"]["confidence"]
            ),
            "page_trees": result["page_trees"],
            "citations": citations,
            "conversation_id": conversation_id,
            "response_time": response_time,
        }

        if include_thinking_process:
            response_data["thinking_process"] = result["thinking_process"]

        if "sub_questions" in result:
            response_data["sub_questions"] = result["sub_questions"]

        # Record metrics
        metrics.record_query(
            success=True,
            response_time=response_time,
            hops=len(result.get("sub_questions", [])),
        )

        # Cache successful response
        cache.set(query, response_data)

        return func.HttpResponse(
            json.dumps(response_data), mimetype="application/json", status_code=200
        )

    except asyncio.TimeoutError:
        metrics.record_query(success=False, response_time=time.time() - start_time)
        return func.HttpResponse(
            json.dumps(
                {
                    "error": "Query processing timed out",
                    "details": "Please try a simpler query or increase timeout.",
                }
            ),
            mimetype="application/json",
            status_code=408,
        )

    except Exception as e:
        metrics.record_query(success=False, response_time=time.time() - start_time)
        logger.error(f"Error processing query: {str(e)}", exc_info=True)

        return func.HttpResponse(
            json.dumps({"error": "Failed to process query", "details": str(e)}),
            mimetype="application/json",
            status_code=500,
        )


# Submit Clarification Function
@app.function_name(name="SubmitClarification")
@app.route(
    route="clarify/{conversation_id}",
    methods=["POST"],
    auth_level=func.AuthLevel.FUNCTION,
)
async def submit_clarification(req: func.HttpRequest) -> func.HttpResponse:
    """Submit clarification for a previous query"""
    conversation_id = req.route_params.get("conversation_id")
    logging.info(f"Submit clarification called for conversation: {conversation_id}")

    initialize_services()

    try:
        req_body = req.get_json()
        clarification = req_body.get("clarification")
    except:
        return func.HttpResponse(
            json.dumps({"error": "Invalid request body"}),
            mimetype="application/json",
            status_code=400,
        )

    if not clarification:
        return func.HttpResponse(
            json.dumps({"error": "Missing clarification field"}),
            mimetype="application/json",
            status_code=400,
        )

    # Get conversation from Cosmos DB
    conversation = await data_store.get_conversation(conversation_id)
    if not conversation:
        return func.HttpResponse(
            json.dumps({"error": "Conversation not found"}),
            mimetype="application/json",
            status_code=404,
        )

    # Get original query from conversation
    messages = conversation.get("messages", [])
    if not messages:
        return func.HttpResponse(
            json.dumps({"error": "No previous query in conversation"}),
            mimetype="application/json",
            status_code=400,
        )

    # Find the last user query
    original_query = None
    for msg in reversed(messages):
        if msg["role"] == "user":
            original_query = msg["content"]
            break

    if not original_query:
        return func.HttpResponse(
            json.dumps({"error": "No user query found in conversation"}),
            mimetype="application/json",
            status_code=400,
        )

    # Create enhanced query with clarification
    enhanced_query = f"{original_query} (Clarification: {clarification})"

    # Process the enhanced query
    req.get_body = lambda: json.dumps(
        {
            "query": enhanced_query,
            "conversation_id": conversation_id,
            "include_thinking_process": True,
        }
    ).encode("utf-8")

    return await process_query(req)


# Get Conversation Function
@app.function_name(name="GetConversation")
@app.route(
    route="conversation/{conversation_id}",
    methods=["GET"],
    auth_level=func.AuthLevel.FUNCTION,
)
async def get_conversation(req: func.HttpRequest) -> func.HttpResponse:
    """Get conversation history"""
    conversation_id = req.route_params.get("conversation_id")
    logging.info(f"Get conversation called for ID: {conversation_id}")

    initialize_services()

    conversation = await data_store.get_conversation(conversation_id)
    if not conversation:
        return func.HttpResponse(
            json.dumps({"error": "Conversation not found"}),
            mimetype="application/json",
            status_code=404,
        )

    return func.HttpResponse(
        json.dumps(conversation), mimetype="application/json", status_code=200
    )


# Delete Conversation Function
@app.function_name(name="DeleteConversation")
@app.route(
    route="conversation/{conversation_id}",
    methods=["DELETE"],
    auth_level=func.AuthLevel.FUNCTION,
)
async def delete_conversation(req: func.HttpRequest) -> func.HttpResponse:
    """Delete conversation history (soft delete)"""
    conversation_id = req.route_params.get("conversation_id")
    logging.info(f"Delete conversation called for ID: {conversation_id}")

    initialize_services()

    conversation = await data_store.get_conversation(conversation_id)
    if not conversation:
        return func.HttpResponse(
            json.dumps({"error": "Conversation not found"}),
            mimetype="application/json",
            status_code=404,
        )

    # Mark as deleted (soft delete)
    conversation["deleted"] = True
    conversation["deletedAt"] = time.time()
    await data_store.save_conversation(conversation_id, conversation["messages"])

    return func.HttpResponse(
        json.dumps({"message": "Conversation marked as deleted"}),
        mimetype="application/json",
        status_code=200,
    )


# Get Metrics Function
@app.function_name(name="GetMetrics")
@app.route(route="metrics", methods=["GET"], auth_level=func.AuthLevel.FUNCTION)
async def get_metrics(req: func.HttpRequest) -> func.HttpResponse:
    """Get system metrics"""
    logging.info("Get metrics endpoint called")

    initialize_services()

    # Get conversation count from Cosmos DB
    conv_count_query = "SELECT VALUE COUNT(1) FROM c WHERE c.deleted != true"
    conv_count = await asyncio.to_thread(
        lambda: list(
            data_store.conversation_container.query_items(
                query=conv_count_query, enable_cross_partition_query=True
            )
        )[0]
    )

    return func.HttpResponse(
        json.dumps(
            {
                "metrics": metrics.get_metrics(),
                "cache_size": len(cache.cache),
                "active_conversations": conv_count,
                "timestamp": datetime.utcnow().isoformat(),
            }
        ),
        mimetype="application/json",
        status_code=200,
    )


# Submit Feedback Function
@app.function_name(name="SubmitFeedback")
@app.route(route="feedback", methods=["POST"], auth_level=func.AuthLevel.FUNCTION)
async def submit_feedback(req: func.HttpRequest) -> func.HttpResponse:
    """Submit feedback for a response"""
    logging.info("Submit feedback endpoint called")

    initialize_services()

    try:
        req_body = req.get_json()
    except:
        return func.HttpResponse(
            json.dumps({"error": "Invalid request body"}),
            mimetype="application/json",
            status_code=400,
        )

    conversation_id = req_body.get("conversation_id")
    helpful = req_body.get("helpful")
    feedback_text = req_body.get("feedback_text")

    if not conversation_id or helpful is None:
        return func.HttpResponse(
            json.dumps({"error": "Missing required fields"}),
            mimetype="application/json",
            status_code=400,
        )

    # Store feedback
    feedback_data = {
        "id": str(uuid.uuid4()),
        "conversation_id": conversation_id,
        "helpful": helpful,
        "feedback_text": feedback_text,
        "timestamp": datetime.utcnow().isoformat(),
    }

    # In production, save to Cosmos DB
    logger.info(f"Feedback received: {feedback_data}")

    return func.HttpResponse(
        json.dumps(
            {"message": "Feedback received", "feedback_id": feedback_data["id"]}
        ),
        mimetype="application/json",
        status_code=200,
    )


# Find Similar Queries Function
@app.function_name(name="FindSimilarQueries")
@app.route(route="search/similar", methods=["GET"], auth_level=func.AuthLevel.FUNCTION)
async def find_similar_queries(req: func.HttpRequest) -> func.HttpResponse:
    """Find similar previously answered queries"""
    logging.info("Find similar queries endpoint called")

    initialize_services()

    query = req.params.get("query")
    limit = int(req.params.get("limit", 5))

    if not query:
        return func.HttpResponse(
            json.dumps({"error": "Missing query parameter"}),
            mimetype="application/json",
            status_code=400,
        )

    # In production, use vector similarity search
    similar = []

    for key, entry in cache.cache.items():
        if "query" in entry:
            # Simple similarity check (in production use embeddings)
            if any(word in entry["query"].lower() for word in query.lower().split()):
                similar.append(
                    {
                        "query": entry["query"],
                        "answer_preview": entry["response"]["answer"][:200] + "...",
                        "confidence": entry["response"].get("confidence", 0),
                        "timestamp": datetime.fromtimestamp(
                            entry["timestamp"]
                        ).isoformat(),
                    }
                )

    # Sort by recency and limit
    similar.sort(key=lambda x: x["timestamp"], reverse=True)

    return func.HttpResponse(
        json.dumps({"similar_queries": similar[:limit], "count": len(similar[:limit])}),
        mimetype="application/json",
        status_code=200,
    )


# Timer Trigger for Cleanup (runs every hour)
@app.function_name(name="CleanupOldData")
@app.schedule(schedule="0 0 * * * *", arg_name="timer", run_on_startup=False)
async def cleanup_old_data(timer: func.TimerRequest) -> None:
    """Periodic cleanup of old conversations and cache"""
    logging.info("Cleanup timer triggered")

    initialize_services()

    # Clean expired cache entries
    cache.clear_expired()

    # Clean old conversations in Cosmos DB (older than 24 hours)
    cutoff_time = time.time() - 86400

    # Query for old conversations
    query = "SELECT c.id FROM c WHERE c.lastUpdated < @cutoff AND (c.deleted != true OR NOT IS_DEFINED(c.deleted))"
    parameters = [{"name": "@cutoff", "value": cutoff_time}]

    try:
        old_conversations = await asyncio.to_thread(
            lambda: list(
                data_store.conversation_container.query_items(
                    query=query,
                    parameters=parameters,
                    enable_cross_partition_query=True,
                )
            )
        )

        # Mark old conversations as deleted
        for conv in old_conversations:
            conv_id = conv["id"]
            conversation = await data_store.get_conversation(conv_id)
            if conversation:
                conversation["deleted"] = True
                conversation["deletedAt"] = time.time()
                await data_store.save_conversation(
                    conv_id, conversation.get("messages", [])
                )

        if old_conversations:
            logger.info(f"Marked {len(old_conversations)} old conversations as deleted")

    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")


# Queue Trigger for Streaming Updates (alternative to SSE)
@app.function_name(name="ProcessQueryAsync")
@app.queue_trigger(
    arg_name="msg", queue_name="query-processing", connection="AzureWebJobsStorage"
)
async def process_query_async(msg: func.QueueMessage) -> None:
    """Process queries asynchronously and store progress in Cosmos DB"""
    logging.info(f"Processing async query: {msg.get_body().decode('utf-8')}")

    initialize_services()

    try:
        message_data = json.loads(msg.get_body().decode("utf-8"))
        query = message_data["query"]
        conversation_id = message_data["conversation_id"]

        # Process query and update status in Cosmos DB
        result = await orchestrator.process_query(query, conversation_id)

        # Store result in Cosmos DB for retrieval
        await data_store.save_query_result(conversation_id, result)

    except Exception as e:
        logger.error(f"Error processing async query: {str(e)}")


# Error handling helper
def create_error_response(
    error: str, details: Any = None, status_code: int = 500
) -> func.HttpResponse:
    """Create standardized error response"""
    response_data = {"error": error, "timestamp": datetime.utcnow().isoformat()}

    if details:
        response_data["details"] = details

    return func.HttpResponse(
        json.dumps(response_data), mimetype="application/json", status_code=status_code
    )
