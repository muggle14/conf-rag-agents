import datetime
import logging
import os
import uuid

# Initialize logger
logger = logging.getLogger(__name__)

# Try to initialize Cosmos DB client, but don't fail if credentials are missing
_cosmos = None
_db = None
_thinking = None

try:
    from azure.cosmos import CosmosClient

    cosmos_endpoint = os.getenv("COSMOS_DB_ENDPOINT")
    cosmos_key = os.getenv("COSMOS_DB_KEY")
    cosmos_database = os.getenv("COSMOS_DB_DATABASE_NAME")
    cosmos_container = os.getenv("COSMOS_DB_THINKING_CONTAINER")

    if all([cosmos_endpoint, cosmos_key, cosmos_database, cosmos_container]):
        _cosmos = CosmosClient(cosmos_endpoint, cosmos_key)
        _db = _cosmos.get_database_client(cosmos_database)
        _thinking = _db.get_container_client(cosmos_container)
        logger.info("Cosmos DB client initialized successfully")
    else:
        logger.warning("Cosmos DB credentials not found, logging to console only")
except Exception as e:
    logger.warning(f"Failed to initialize Cosmos DB client: {e}")


def log_step(agent: str, action: str, reasoning: str, result: str):
    """Log a step to Cosmos DB if available, otherwise to console."""
    log_entry = {
        "id": str(uuid.uuid4()),
        "agent": agent,
        "action": action,
        "reasoning": reasoning,
        "result": result,
        "ts": datetime.datetime.utcnow().isoformat() + "Z",
    }

    if _thinking:
        try:
            _thinking.upsert_item(log_entry)
        except Exception as e:
            logger.error(f"Failed to log to Cosmos DB: {e}")
            logger.info(f"Log entry: {log_entry}")
    else:
        logger.info(f"Log entry: {log_entry}")
