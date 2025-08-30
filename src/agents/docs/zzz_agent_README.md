# Confluence Q&A System Architecture Summary

## Overview
The Confluence Q&A system is built using Microsoft AutoGen for multi-agent orchestration and integrates with Azure services for scalable, enterprise-grade performance. The system uses **Azure Functions** with HTTP triggers for serverless API endpoints, providing automatic scaling and cost optimization.

## Key Components

### 1. **AutoGen Agents**
The system uses specialized AutoGen agents for different tasks:

- **Query Analyser Agent**: Classifies queries (Atomic, NeedsDecomposition, NeedsClarification)
- **Decomposer Agent**: Breaks complex queries into sub-questions
- **Path Planner Agent**: Plans traversal through the knowledge graph
- **Retriever Agent**: Executes searches on Azure Cognitive Search
- **Reranker Agent**: Applies semantic reranking to results
- **Synthesiser Agent**: Generates comprehensive answers with citations
- **Verifier Agent**: Validates answer accuracy and prevents hallucination
- **Clarifier Agent**: Handles ambiguous queries with natural language clarification
- **Tree Builder Agent**: Constructs page hierarchy visualizations

### 2. **Azure Service Integration**

#### **Azure Functions**
- **HTTP Triggers**: RESTful API endpoints for all operations
- **Timer Triggers**: Scheduled cleanup and maintenance tasks
- **Queue Triggers**: Asynchronous query processing
- **Consumption Plan**: Automatic scaling based on demand
- **Integrated Authentication**: Function-level and Azure AD authentication

#### **Azure Cognitive Search**
- Stores indexed Confluence content with embeddings
- Provides hybrid search (vector + keyword)
- Semantic search capabilities
- Metadata filtering for multi-hop queries

#### **Azure Cosmos DB**
- **Gremlin API**: Stores page relationships and graph structure
- **SQL API**: Stores conversations, thinking steps, and structured data
- Provides persistent storage for all system data

#### **Azure Blob Storage**
- **Raw Container**: Stores original Confluence JSON documents
- **Processed Container**: Stores chunked and processed content

#### **Azure OpenAI**
- Embeddings generation (text-embedding-3-large)
- LLM for AutoGen agents (gpt-4o)
- Semantic reranking capabilities

### 3. **Data Architecture**

#### **Persistent Data (Cosmos DB)**
- Conversation history
- Thinking process steps
- Page relationships (graph)
- User feedback
- System metrics

#### **Temporary Data (Dataclasses)**
- Query analysis results
- Search results from Azure Cognitive Search
- Runtime configuration

#### **Cached Data (In-Memory)**
- Recent query responses
- Frequently accessed embeddings

### 4. **Key Features**

#### **Transparent Thinking Process**
Every step taken by the AutoGen agents is logged to Cosmos DB with:
- Agent name
- Action taken
- Reasoning
- Results
- Timestamp

#### **Multi-Hop Graph Traversal**
The system can follow relationships through the knowledge graph:
- ParentOf relationships
- LinksTo relationships
- Custom edge types
- Maximum hop limit (configurable, default: 3)

#### **Page Tree Visualization**
Generates markdown trees showing:
- Document hierarchies
- Answer source pages (highlighted with ⭐)
- Multiple trees when answers span hierarchies

#### **Natural Language Clarification**
When queries are ambiguous:
- Provides specific clarifying questions
- Offers examples to guide users
- Maintains conversational context

### 5. **API Endpoints**

All endpoints are implemented as HTTP-triggered Azure Functions with the base URL: `https://{function-app-name}.azurewebsites.net/api`

#### **Query Processing**
- `POST /api/query`: Process a user query with full AutoGen orchestration
  - Auth Level: Function
  - Request Body: `{ "query": string, "conversation_id": string?, "include_thinking_process": boolean?, "max_wait_seconds": number? }`
  - Response: Answer with citations, page trees, and thinking process

#### **Conversation Management**
- `GET /api/conversation/{conversation_id}`: Retrieve conversation history
  - Auth Level: Function
  - Response: Complete conversation with messages and metadata

- `DELETE /api/conversation/{conversation_id}`: Soft delete conversation
  - Auth Level: Function
  - Response: Confirmation message

- `POST /api/clarify/{conversation_id}`: Submit clarification for ambiguous query
  - Auth Level: Function
  - Request Body: `{ "clarification": string }`
  - Response: Enhanced query results

#### **System Operations**
- `GET /api/health`: Health check endpoint
  - Auth Level: Anonymous
  - Response: System status, version, and basic metrics

- `GET /api/metrics`: Detailed system performance metrics
  - Auth Level: Function
  - Response: Query statistics, cache performance, active conversations

- `POST /api/feedback`: Submit user feedback
  - Auth Level: Function
  - Request Body: `{ "conversation_id": string, "helpful": boolean, "feedback_text": string? }`
  - Response: Feedback confirmation

#### **Search Operations**
- `GET /api/search/similar?query={query}&limit={limit}`: Find similar previously answered queries
  - Auth Level: Function
  - Response: List of similar queries with answer previews

#### **Background Functions**
- **Timer Function - CleanupOldData**: Runs hourly to archive old conversations and clear expired cache
- **Queue Function - ProcessQueryAsync**: Handles long-running queries asynchronously

### 6. **Performance Optimizations**

- **Response Caching**: Recent queries cached in memory
- **Batch Processing**: Documents processed in batches
- **Async Operations**: All I/O operations are async
- **Connection Pooling**: Reused connections to Azure services
- **Smart Indexing**: Optimized search indices with vector embeddings

### 7. **Reliability Features**

- **Soft Deletes**: Conversations archived, not deleted
- **Automatic Cleanup**: Old data cleaned periodically
- **Error Recovery**: Graceful handling of service failures
- **Timeout Protection**: Configurable timeouts for all operations

### 8. **Security Considerations**

- **Function-Level Authentication**: API key protection for functions
- **Azure AD Integration**: Enterprise authentication support
- **Managed Identity**: Secure service-to-service communication
- **Encrypted Storage**: All data encrypted at rest
- **HTTPS Only**: All API communications encrypted
- **CORS Configuration**: Configurable for production environments

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Client Layer                              │
├─────────────────┬───────────────────┬─────────────────────────────┤
│   React Web UI  │  Mobile App       │  API Integrations           │
└────────┬────────┴─────────┬─────────┴───────────┬─────────────────┘
         │                  │                     │
         └──────────────────┴─────────────────────┘
                            │
                    HTTPS (REST API)
                            │
┌───────────────────────────▼─────────────────────────────────────────┐
│                    Azure Functions (API Layer)                       │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              HTTP Triggered Functions                        │   │
│  ├─────────────────────────────────────────────────────────────┤   │
│  │ • ProcessQuery    • GetConversation   • SubmitFeedback      │   │
│  │ • HealthCheck     • DeleteConversation• FindSimilarQueries  │   │
│  │ • GetMetrics      • SubmitClarification                     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │          Timer & Queue Triggered Functions                   │   │
│  ├─────────────────────────────────────────────────────────────┤   │
│  │ • CleanupOldData (Timer - Hourly)                           │   │
│  │ • ProcessQueryAsync (Queue - Async Processing)              │   │
│  └─────────────────────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
              Internal Service Calls
                       │
┌──────────────────────▼──────────────────────────────────────────────┐
│                    AutoGen Agent Orchestration                       │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │Query Analyser│  │  Decomposer  │  │Path Planner  │              │
│  └──────┬──────┘  └──────┬───────┘  └──────┬───────┘              │
│         │                │                  │                       │
│  ┌──────▼──────┐  ┌──────▼───────┐  ┌──────▼───────┐              │
│  │  Retriever  │  │   Reranker   │  │ Tree Builder │              │
│  └──────┬──────┘  └──────┬───────┘  └──────┬───────┘              │
│         │                │                  │                       │
│  ┌──────▼──────┐  ┌──────▼───────┐  ┌──────▼───────┐              │
│  │ Synthesiser │  │   Verifier   │  │  Clarifier   │              │
│  └─────────────┘  └──────────────┘  └──────────────┘              │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
                Azure SDK Calls
                       │
┌──────────────────────▼──────────────────────────────────────────────┐
│                        Azure Services Layer                          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │ Azure Cognitive │  │   Azure Cosmos   │  │  Azure Storage   │  │
│  │     Search      │  │       DB         │  │  (Blob & Queue)  │  │
│  ├─────────────────┤  ├──────────────────┤  ├──────────────────┤  │
│  │ • Vector Search │  │ • SQL API        │  │ • Raw Docs       │  │
│  │ • Hybrid Search │  │ • Gremlin API    │  │ • Processed Docs │  │
│  │ • Semantic Rank │  │ • Conversations  │  │ • Query Queue    │  │
│  └─────────────────┘  └──────────────────┘  └──────────────────┘  │
│                                                                      │
│  ┌─────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │  Azure OpenAI   │  │    Application   │  │   Azure Key      │  │
│  │                 │  │     Insights     │  │     Vault        │  │
│  ├─────────────────┤  ├──────────────────┤  ├──────────────────┤  │
│  │ • GPT-4o        │  │ • Metrics        │  │ • API Keys       │  │
│  │ • Embeddings    │  │ • Logs           │  │ • Certificates   │  │
│  │ • Reranking     │  │ • Traces         │  │ • Secrets        │  │
│  └─────────────────┘  └──────────────────┘  └──────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘

```

### Infrastructure Details:

- **Azure Functions App**:
  - Runtime: Python 3.11
  - Plan: Consumption (Y1) for automatic scaling
  - Region: Same as other resources for low latency
  - Authentication: Function keys and Azure AD

- **Networking**:
  - All services communicate over Azure backbone network
  - Private endpoints available for enhanced security
  - Application Gateway optional for custom domains

- **Monitoring Stack**:
  - Application Insights for APM
  - Log Analytics Workspace for centralized logging
  - Azure Monitor alerts for proactive monitoring

## Cost Optimization

The system is designed for cost efficiency:
- **Serverless Architecture**: Pay only for actual execution time
- **Consumption Plans**: No idle compute costs
- **Serverless Cosmos DB**: Pay per request pricing
- **Free/Basic Tiers**: Where possible (Search, App Service)
- **Smart Caching**: Reduces API calls
- **Efficient Queries**: Optimized graph traversals

## Monitoring and Observability

- **Thinking Process Logs**: Complete audit trail
- **Performance Metrics**: Response times, success rates
- **Resource Usage**: Azure monitor integration
- **Error Tracking**: Detailed error logs with context
- **Real-time Dashboards**: Application Insights Live Metrics
- **Custom Alerts**: Configurable thresholds for key me
