#!/bin/bash

# Deploy embedding model in Azure OpenAI
set -e

# Configuration
RESOURCE_GROUP="rg-rag-conf"
AOAI_RESOURCE="aoai-rag-conf"
MODEL_NAME="text-embedding-ada-002"
DEPLOYMENT_NAME="text-embedding-ada-002"

echo "🚀 Deploying embedding model in Azure OpenAI..."

# Check if deployment already exists
echo "📋 Checking existing deployments..."
EXISTING_DEPLOYMENT=$(az cognitiveservices account deployment list \
    --resource-group $RESOURCE_GROUP \
    --name $AOAI_RESOURCE \
    --query "[?name=='$DEPLOYMENT_NAME'].name" \
    --output tsv 2>/dev/null || echo "")

if [ "$EXISTING_DEPLOYMENT" = "$DEPLOYMENT_NAME" ]; then
    echo "✅ Deployment '$DEPLOYMENT_NAME' already exists!"
    exit 0
fi

# Get available models
echo "📋 Getting available models..."
MODELS=$(az cognitiveservices account model list \
    --resource-group $RESOURCE_GROUP \
    --name $AOAI_RESOURCE \
    --query "[?contains(name, 'embedding')].name" \
    --output tsv)

echo "Available embedding models:"
echo "$MODELS"

# Check if the model is available
if echo "$MODELS" | grep -q "$MODEL_NAME"; then
    echo "✅ Model '$MODEL_NAME' is available"
else
    echo "❌ Model '$MODEL_NAME' is not available"
    echo "Available models:"
    echo "$MODELS"
    exit 1
fi

# Deploy the model
echo "🚀 Deploying model '$MODEL_NAME' as '$DEPLOYMENT_NAME'..."
az cognitiveservices account deployment create \
    --resource-group $RESOURCE_GROUP \
    --name $AOAI_RESOURCE \
    --deployment-name $DEPLOYMENT_NAME \
    --model-name $MODEL_NAME \
    --model-version "2" \
    --scale-settings-scale-type "Standard" \
    --scale-settings-capacity "1"

echo "✅ Deployment completed successfully!"
echo "📋 Deployment details:"
az cognitiveservices account deployment show \
    --resource-group $RESOURCE_GROUP \
    --name $AOAI_RESOURCE \
    --deployment-name $DEPLOYMENT_NAME \
    --query "{name: name, model: model.name, status: properties.provisioningState}" \
    --output table

echo ""
echo "🔧 Update your .env file with:"
echo "AOAI_EMBED_DEPLOY=$DEPLOYMENT_NAME"
