"""
Environment configuration for search tools and agents.
"""

import os

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure Search Configuration
AZ_SEARCH_ENDPOINT = os.environ.get("AZURE_SEARCH_ENDPOINT")
AZ_SEARCH_INDEX = os.environ.get(
    "AZURE_SEARCH_INDEX_NAME", "confluence-graph-embeddings-v2"
)
AZ_SEARCH_KEY = os.environ.get("AZURE_SEARCH_KEY")
AZ_SEARCH_SEM_CFG = os.getenv("AZURE_SEARCH_SEMANTIC_CONFIG")  # optional

# Re-rank toggle + bounds
AGENT_RERANK_ENABLED = os.getenv("AGENT_RERANK_ENABLED", "false").lower() == "true"
AGENT_RERANK_MAXK = int(os.getenv("AGENT_RERANK_MAXK", "8"))

# LLM provider (for rerank/clarify/summarize)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "azure_openai")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")

# Azure OpenAI Configuration (for embeddings)
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AOAI_EMBED_DEPLOY = os.getenv("AOAI_EMBED_DEPLOY")
