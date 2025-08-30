# Quick Start Guide: Confluence Q&A System

## Prerequisites
- Azure subscription with the following services:
  - Azure Cognitive Search (Basic tier or higher)
  - Azure Cosmos DB (Serverless)
  - Azure Storage Account
  - Azure OpenAI resource
- Python 3.11+
- Confluence API access token

## Setup Steps

### 1. Clone and Install
```bash
git clone <repository-url>
cd confluence-qa-system
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the project root:

```bash
# Azure Subscription
AZ_SUBSCRIPTION_ID=<your-subscription-id>
AZ_RESOURCE_GROUP=rg-rag-confluence
AZ_LOCATION=westeurope

# Cosmos DB
COSMOS_ACCOUNT=<your-cosmos-account>
COSMOS_KEY=<your-cosmos-key>
COSMOS_DB=confluence
COSMOS_GRAPH=pages

# Storage
STORAGE_ACCOUNT=<your-storage-account>
STORAGE_KEY=<your-storage-key>

# Azure Cognitive Search
SEARCH_SERVICE=<your-search-service>
SEARCH_INDEX=confluence-idx
SEARCH_ENDPOINT=https://<your-search-service>.search.windows.net
SEARCH_KEY=<your-search-key>

# Azure OpenAI
AOAI_RESOURCE=<your-aoai-resource>
AOAI_ENDPOINT=https://<your-aoai-resource>.openai.azure.com
AOAI_KEY=<your-aoai-key>
AOAI_EMBED_DEPLOY=text-embedding-3-large
AOAI_CHAT_DEPLOY=gpt-4o

# Confluence
CONFLUENCE_BASE=https://<your-org>.atlassian.net/wiki/rest/api
CONFLUENCE_TOKEN=<your-pat-token>
CONFLUENCE_ORG=<your-org>

# Optional: Linear Integration
LINEAR_TEAM_ID=<your-linear-team-id>
```

### 3. Initialize Azure Resources

Run the initialization script to create Cosmos DB containers:

```python
# init_cosmos.py
import os
from azure.cosmos import CosmosClient, PartitionKey

client = CosmosClient(
    url=f"https://{os.environ['COSMOS_ACCOUNT']}.documents.azure.com",
    credential=os.environ['COSMOS_KEY']
)

database = client.create_database_if_not_exists(os.environ['COSMOS_DB'])

# Create containers
containers = [
    ("thinking_steps", "/conversationId"),
    ("conversations", "/id"),
    ("linear_tickets", "/linearIssueId")
]

for container_name, partition_key in containers:
    database.create_container_if_not_exists(
        id=container_name,
        partition_key=PartitionKey(path=partition_key)
    )

print("Cosmos DB initialized successfully!")
```

### 4. Start the API Service

```bash
# Development
uvicorn api_service:app --reload --host 0.0.0.0 --port 8000

# Production with Docker
docker-compose up -d
```

### 5. Test the System

```bash
# Health check
curl http://localhost:8000/health

# Ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I enable SSO for our application?",
    "include_thinking_process": true
  }'
```

## Usage Examples

### Python Client Example
```python
import requests
import json

# Ask a question
response = requests.post(
    "http://localhost:8000/query",
    json={
        "query": "What are the deployment steps for the payment service?",
        "include_thinking_process": True
    }
)

result = response.json()

# Display answer
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']:.2%}")

# Show page trees
for tree in result['page_trees']:
    print(f"\nPage Tree: {tree['root_title']}")
    print(tree['markdown'])

# Show thinking process
if 'thinking_process' in result:
    print("\nThinking Process:")
    for step in result['thinking_process']:
        print(f"  {step['step']}. [{step['agent']}] {step['action']}")
```

### Streaming Example
```python
import requests
import json

# Stream real-time updates
response = requests.post(
    "http://localhost:8000/query/stream",
    json={"query": "How does our authentication system work?"},
    stream=True
)

for line in response.iter_lines():
    if line and line.startswith(b'data: '):
        data = json.loads(line[6:])

        if data['event'] == 'thinking_step':
            print(f"[{data['agent']}] {data['action']}: {data['reasoning']}")
        elif data['event'] == 'completed':
            print(f"Final answer: {data['result']['answer']}")
```

## Common Operations

### Submit Clarification
```python
# If the system needs clarification
response = requests.post(
    f"http://localhost:8000/clarify/{conversation_id}",
    params={"clarification": "I meant the SSO setup for the mobile app"}
)
```

### View Conversation History
```python
response = requests.get(f"http://localhost:8000/conversation/{conversation_id}")
conversation = response.json()

for message in conversation['messages']:
    print(f"{message['role']}: {message['content']}")
```

### Check System Metrics
```python
response = requests.get("http://localhost:8000/metrics")
metrics = response.json()

print(f"Total Queries: {metrics['metrics']['queries_processed']}")
print(f"Success Rate: {metrics['metrics']['success_rate']:.2%}")
print(f"Active Conversations: {metrics['active_conversations']}")
```

## Troubleshooting

### Common Issues

1. **"Azure service not found" errors**
   - Verify all Azure resources are created
   - Check service endpoints and keys in `.env`
   - Ensure proper network connectivity

2. **"No results found" for queries**
   - Check if Confluence content is indexed
   - Verify search index schema includes embeddings
   - Run the indexing pipeline first

3. **Slow response times**
   - Check Azure service tiers (upgrade if needed)
   - Monitor thinking process for bottlenecks
   - Enable caching for repeated queries

4. **Authentication errors**
   - Verify Azure AD permissions
   - Check Confluence API token validity
   - Ensure proper RBAC roles assigned

## Next Steps

1. **Index Your Confluence Content**
   - Run the ingestion pipeline
   - Process and chunk documents
   - Generate embeddings and populate search index

2. **Customize Agents**
   - Modify prompts in `prompts.py`
   - Adjust confidence thresholds
   - Add domain-specific logic

3. **Enable Linear Integration**
   - Configure Linear API credentials
   - Set up team and label IDs
   - Monitor created tickets

4. **Deploy to Production**
   - Use managed identities for authentication
   - Enable HTTPS with proper certificates
   - Set up monitoring and alerting
   - Configure backup and disaster recovery

For detailed documentation, see the [Architecture Summary](confluence-qa-architecture).
