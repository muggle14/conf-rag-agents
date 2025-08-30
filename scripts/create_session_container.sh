#!/bin/bash

# Create Sessions Container in Cosmos DB
# This script creates a new container for session storage in the existing Cosmos database

# Configuration
RESOURCE_GROUP="rg-rag-confluence"
COSMOS_ACCOUNT="cosmos-rag-conf"
DATABASE_NAME="confluence-graph"
CONTAINER_NAME="sessions"
PARTITION_KEY="/session_id"
THROUGHPUT=400

echo "🚀 Creating Sessions Container in Cosmos DB"
echo "=========================================="
echo "  Account: $COSMOS_ACCOUNT"
echo "  Database: $DATABASE_NAME"
echo "  Container: $CONTAINER_NAME"
echo "  Partition Key: $PARTITION_KEY"
echo "  Throughput: $THROUGHPUT RU/s"
echo ""

# Create the container
echo "📦 Creating container..."
az cosmosdb sql container create \
    --resource-group $RESOURCE_GROUP \
    --account-name $COSMOS_ACCOUNT \
    --database-name $DATABASE_NAME \
    --name $CONTAINER_NAME \
    --partition-key-path $PARTITION_KEY \
    --throughput $THROUGHPUT \
    --output json

if [ $? -eq 0 ]; then
    echo "✅ Container created successfully!"

    echo ""
    echo "📝 Add these to your .env file:"
    echo "=========================================="
    echo "# Session Storage Container"
    echo "COSMOS_SESSION_CONTAINER=sessions"
    echo "COSMOS_SESSION_PARTITION_KEY=/session_id"
    echo ""

    # Get the SQL API endpoint
    SQL_ENDPOINT=$(az cosmosdb show \
        --resource-group $RESOURCE_GROUP \
        --name $COSMOS_ACCOUNT \
        --query "documentEndpoint" \
        --output tsv)

    echo "# SQL API Endpoint (for session storage)"
    echo "COSMOS_SQL_ENDPOINT=$SQL_ENDPOINT"
    echo "=========================================="

else
    echo "❌ Failed to create container"
    exit 1
fi
