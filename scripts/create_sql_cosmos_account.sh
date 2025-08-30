#!/bin/bash

# Create a new Cosmos DB account for SQL API (session storage)
# This is separate from the existing Gremlin API account

# Configuration
RESOURCE_GROUP="rg-rag-confluence"
SQL_COSMOS_ACCOUNT="cosmos-rag-sessions"  # New account name
LOCATION="eastus"
DATABASE_NAME="rag-sessions"
CONTAINER_NAME="sessions"
PARTITION_KEY="/session_id"
THROUGHPUT=400

echo "🚀 Creating SQL API Cosmos DB Account for Session Storage"
echo "========================================================="
echo "  Resource Group: $RESOURCE_GROUP"
echo "  Account Name: $SQL_COSMOS_ACCOUNT"
echo "  Location: $LOCATION"
echo "  Database: $DATABASE_NAME"
echo "  Container: $CONTAINER_NAME"
echo ""

# Step 1: Create Cosmos DB account with SQL API
echo "📦 Step 1: Creating Cosmos DB account with SQL API..."
az cosmosdb create \
    --resource-group $RESOURCE_GROUP \
    --name $SQL_COSMOS_ACCOUNT \
    --kind GlobalDocumentDB \
    --locations regionName=$LOCATION failoverPriority=0 isZoneRedundant=False \
    --default-consistency-level Session \
    --enable-free-tier false \
    --output json

if [ $? -ne 0 ]; then
    echo "❌ Failed to create Cosmos DB account"
    exit 1
fi

echo "✅ Cosmos DB account created"

# Step 2: Create database
echo ""
echo "📦 Step 2: Creating database..."
az cosmosdb sql database create \
    --resource-group $RESOURCE_GROUP \
    --account-name $SQL_COSMOS_ACCOUNT \
    --name $DATABASE_NAME \
    --output json

if [ $? -ne 0 ]; then
    echo "❌ Failed to create database"
    exit 1
fi

echo "✅ Database created"

# Step 3: Create container
echo ""
echo "📦 Step 3: Creating container..."
az cosmosdb sql container create \
    --resource-group $RESOURCE_GROUP \
    --account-name $SQL_COSMOS_ACCOUNT \
    --database-name $DATABASE_NAME \
    --name $CONTAINER_NAME \
    --partition-key-path $PARTITION_KEY \
    --throughput $THROUGHPUT \
    --output json

if [ $? -ne 0 ]; then
    echo "❌ Failed to create container"
    exit 1
fi

echo "✅ Container created"

# Step 4: Get connection details
echo ""
echo "📝 Step 4: Getting connection details..."

# Get endpoint
SQL_ENDPOINT=$(az cosmosdb show \
    --resource-group $RESOURCE_GROUP \
    --name $SQL_COSMOS_ACCOUNT \
    --query "documentEndpoint" \
    --output tsv)

# Get primary key
SQL_KEY=$(az cosmosdb keys list \
    --resource-group $RESOURCE_GROUP \
    --name $SQL_COSMOS_ACCOUNT \
    --type keys \
    --query "primaryMasterKey" \
    --output tsv)

echo ""
echo "========================================================="
echo "✅ SQL Cosmos DB Account Created Successfully!"
echo "========================================================="
echo ""
echo "📝 Add these lines to your .env file:"
echo "========================================================="
echo ""
echo "# Cosmos DB SQL API (for session storage)"
echo "COSMOS_SQL_ACCOUNT=$SQL_COSMOS_ACCOUNT"
echo "COSMOS_SQL_ENDPOINT=$SQL_ENDPOINT"
echo "COSMOS_SQL_KEY=$SQL_KEY"
echo "COSMOS_SQL_DATABASE=$DATABASE_NAME"
echo "COSMOS_SQL_CONTAINER=$CONTAINER_NAME"
echo ""
echo "========================================================="
echo ""
echo "💰 Cost Note: This creates a new Cosmos account with 400 RU/s"
echo "   Estimated cost: ~\$24/month (can be reduced to 100 RU/s if needed)"
echo ""
