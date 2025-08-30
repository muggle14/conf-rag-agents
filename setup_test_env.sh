#!/bin/bash
# Setup environment variables for testing

# Load from .env if exists
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# API Configuration
export API_BASE="${API_BASE:-http://localhost:8000}"

# Test Queries - Use queries that actually exist in the index
export TEST_SPECIFIC_QUERY="${TEST_SPECIFIC_QUERY:-synthtrace}"
export TEST_AMBIGUOUS_QUERY="${TEST_AMBIGUOUS_QUERY:-configuration}"
export TEST_SPACE="${TEST_SPACE:-}"

# Azure Search - already configured in .env
export AZURE_SEARCH_ENDPOINT="${AZURE_SEARCH_ENDPOINT}"
export AZURE_SEARCH_KEY="${AZURE_SEARCH_KEY}"
export AZURE_SEARCH_INDEX="${AZURE_SEARCH_INDEX:-${SEARCH_INDEX:-confluence-graph-embeddings-v2}}"

# Cosmos DB Graph - use actual values from .env
export COSMOS_GRAPH_DB_ENDPOINT="${COSMOS_GRAPH_DB_ENDPOINT:-${COSMOS_ENDPOINT:-cosmos-rag-conf.gremlin.cosmos.azure.com}}"
export COSMOS_GRAPH_DB_KEY="${COSMOS_GRAPH_DB_KEY:-${COSMOS_KEY}}"
export COSMOS_GRAPH_DB_DATABASE="${COSMOS_GRAPH_DB_DATABASE:-${COSMOS_DATABASE:-confluence-graph}}"
export COSMOS_GRAPH_DB_COLLECTION="${COSMOS_GRAPH_DB_COLLECTION:-${COSMOS_CONTAINER:-knowledge-graph}}"

# Gremlin connection strings (derived from Cosmos)
if [ ! -z "$COSMOS_GRAPH_DB_ENDPOINT" ]; then
    export GREMLIN_URL="wss://${COSMOS_GRAPH_DB_ENDPOINT}:443/"
    export GREMLIN_USER="/dbs/${COSMOS_GRAPH_DB_DATABASE}/colls/${COSMOS_GRAPH_DB_COLLECTION}"
    export GREMLIN_PASS="${COSMOS_GRAPH_DB_KEY}"
fi

# Azure OpenAI
export AZURE_OPENAI_ENDPOINT="${AZURE_OPENAI_ENDPOINT}"
export AZURE_OPENAI_API_KEY="${AZURE_OPENAI_API_KEY}"

echo "Environment configured for testing:"
echo "================================="
echo "API_BASE: $API_BASE"
echo "TEST_SPECIFIC_QUERY: $TEST_SPECIFIC_QUERY"
echo "TEST_AMBIGUOUS_QUERY: $TEST_AMBIGUOUS_QUERY"
echo "AZURE_SEARCH_ENDPOINT: ${AZURE_SEARCH_ENDPOINT:0:30}..."
echo "COSMOS_GRAPH_DB_ENDPOINT: ${COSMOS_GRAPH_DB_ENDPOINT:0:30}..."
echo ""
