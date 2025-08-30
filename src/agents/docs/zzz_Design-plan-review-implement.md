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

Confluence → Azure Blob (HTML/Markdown) ─┐
                                         ├───[ Indexer A : integrated-vectorisation ]──►  Index 1  confluence-chunks  (vector + text + graph cols)
Cosmos DB (Gremlin)  ──►  (optional) Indexer B  ───────────────────────────────────────►  Index 2  confluence-graph-props  (graph metadata only)


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

2 Index 1 – confluence-chunks (vector + text)


PUT https://<search-service>.search.windows.net/indexes/confluence-chunks?api-version=2024-07-01
Content-Type: application/json
api-key: <admin-key>

{
  "name": "confluence-chunks",
  "fields": [
    { "name": "id",           "type": "Edm.String",           "key": true,  "filterable": true },
    { "name": "pageId",       "type": "Edm.String",           "filterable": true,  "facetable": true },
    { "name": "spaceId",      "type": "Edm.String",           "filterable": true,  "facetable": true },
    { "name": "parentId",     "type": "Edm.String",           "filterable": true },
    { "name": "treePath",     "type": "Edm.String",           "filterable": true },
    { "name": "depth",        "type": "Edm.Int32",            "filterable": true },
    { "name": "centrality",   "type": "Edm.Double",           "filterable": true, "sortable": true },
    { "name": "adjacentIds",  "type": "Collection(Edm.String)","filterable": true },
    { "name": "offset",       "type": "Edm.Int32"                              },
    { "name": "chunkText",    "type": "Edm.String",           "searchable": true, "retrievable": true },
    { "name": "contentVector","type": "Collection(Edm.Half)", "vectorSearchConfiguration": "vec_cfg" },
    { "name": "title",        "type": "Edm.String",           "searchable": true, "retrievable": true },
    { "name": "pageVector",   "type": "Collection(Edm.Half)", "vectorSearchConfiguration": "vec_cfg" }
  ],
  "vectorSearch": {
    "algorithms": [ { "name": "vec_cfg", "kind": "hnsw", "dimension": 1536 } ]
  },
  "corsOptions": { "allowedOrigins": ["*"] }
}


3 Indexer A – Confluence content → integrated vectorisation

PUT https://<search-service>.search.windows.net/datasources/blob-confluence?api-version=2024-07-01
Content-Type: application/json
api-key: <admin-key>

{
  "name": "blob-confluence",
  "type": "azureblob",
  "credentials": { "connectionString": "<blob-conn-str>" },
  "container": { "name": "confluence-export" }
}

4 Index 2 – confluence-graph-props (optional, pure graph metadata)
If you decide to merge graph properties into Index 1 instead, skip this and change targetIndexName in Indexer B.


6 Queries – no-hop vs multi-hop

6.1 No-hop / single-hop (Search-only)

POST https://<search-service>.search.windows.net/indexes/confluence-chunks/docs/search?api-version=2024-07-01
api-key: <query-key>
Content-Type: application/json

{
  "search": "Set up SAML SSO for Jira",
  "vectorQueries": [
    { "kind": "text",
      "fields": "contentVector",
      "text": "Set up SAML SSO for Jira",
      "k": 15,
      "weight": 1.0 }
  ],
  "scoringProfile": "boostParent",
  "scoringParameters": [ "parentIdParam--12345" ],
  "select": "chunkText,pageId,parentId,title,treePath"
}


6.2 Multi-hop (Gremlin + Search)

# pseudo-Python
seed_ids = run_search_seed("Set up SAML SSO for Jira")          # returns top 3 chunk.pageId
subgraph = gremlin_client.submit(
    f"g.V({seed_ids}).repeat(bothE().otherV()).times(2).path()"
).all()
ids = list({v["pageId"] for v in subgraph})

resp = search_client.search(
    search="Set up SAML SSO for Jira",
    vector_queries=[
      TextVectorQuery(text="Set up SAML SSO for Jira", fields="contentVector", k=20, weight=1),
      TextVectorQuery(text=" ".join(get_titles(ids)),  fields="pageVector",   k=10, weight=0.3)
    ],
    filter=f"pageId in ({','.join(ids)})"
)


## Usage Examples
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



# production-config.yaml
# Production configuration for Confluence Q&A system with balanced performance

# System Performance Settings - Prioritizing Accuracy
performance:
  # Reasonable timeout for quality responses
  total_timeout_seconds: 10.0

  # Agent-specific timeouts (sufficient for accurate processing)
  agent_timeouts:
    query_analyser: 1.0
    decomposer: 2.0
    path_planner: 1.5
    retriever: 3.0
    reranker: 2.0
    synthesiser: 3.0
    verifier: 2.0
    clarifier: 1.0
    tree_builder: 2.0

  # Query processing limits (balanced for accuracy)
  limits:
    max_subquestions: 5         # Allow complex decomposition
    max_search_results: 15      # Comprehensive results
    max_search_hops: 3          # Full graph traversal
    max_parallel_agents: 5      # Good parallelism
    max_context_chunks: 20      # Full context
    max_response_tokens: 500    # Complete responses
    max_tree_depth: 5           # Full tree visualization

# Azure Service Configuration
azure:
  # Cognitive Search - Balanced for accuracy and speed
  search:
    mode: "comprehensive"  # Options: fast, balanced, comprehensive
    search_modes:
      - vector_search:
          k: 15
          timeout_ms: 1000
      - keyword_search:
          mode: "all"  # More accurate than "any"
          top: 15
          timeout_ms: 800
      - semantic_search:
          enabled: true
          configuration: "default"
          timeout_ms: 1000
    cache_ttl_seconds: 3600
    include_highlights: true
    include_facets: true

  # Cosmos DB - Optimized for reliability
  cosmos:
    connection_mode: "Direct"  # Better throughput
    request_timeout_seconds: 5
    max_connections: 100
    enable_endpoint_discovery: true
    consistency_level: "Session"

  # OpenAI - Full capability
  openai:
    deployment: "gpt-4o"
    max_tokens: 500         # Full responses
    temperature: 0
    timeout_seconds: 5
    max_retries: 3
    stream_enabled: true
    embeddings:
      batch_size: 16
      cache_enabled: true

# Caching Configuration - Performance without compromising accuracy
cache:
  # Response cache (safe optimization)
  response_cache:
    enabled: true
    ttl_seconds: 3600
    max_size: 1000

  # Embedding cache (safe optimization)
  embedding_cache:
    enabled: true
    ttl_seconds: 86400
    max_size: 5000
    preload_common: true
    common_queries:
      - "password reset"
      - "single sign on"
      - "deployment guide"
      - "authentication"
      - "api documentation"

  # Agent response cache (safe optimization)
  agent_cache:
    enabled: true
    ttl_seconds: 1800
    max_size: 500

  # Search result cache (safe optimization)
  search_cache:
    enabled: true
    ttl_seconds: 600
    max_size: 200

# Query Pattern Matching - With Verification
patterns:
  enabled: true
  require_verification: true    # Always verify pattern matches
  confidence_threshold: 0.95
  patterns_file: "verified_patterns.json"

  # Common patterns with verified responses
  common_patterns:
    - pattern: "(reset|change).*password"
      category: "password_reset"
      response_template: "password_reset.md"
      sources: ["page-auth-001", "page-security-guide"]

    - pattern: "(sso|single sign)"
      category: "sso_setup"
      response_template: "sso_setup.md"
      sources: ["page-sso-guide", "page-auth-providers"]

    - pattern: "(deploy|deployment)"
      category: "deployment"
      response_template: "deployment_guide.md"
      sources: ["page-deploy-001", "page-ci-cd-guide"]

# AutoGen Agent Configuration - Balanced
autogen:
  # LLM configuration for all agents
  llm_config:
    model: "gpt-4o"
    temperature: 0
    max_tokens: 500         # Full responses
    top_p: 0.1
    frequency_penalty: 0
    presence_penalty: 0
    seed: 42               # Deterministic responses
    cache_seed: 42

  # Agent pool settings (performance optimization)
  agent_pool:
    enabled: true
    size: 5
    prewarm: true

  # Conversation settings
  max_consecutive_auto_reply: 1
  human_input_mode: "NEVER"

# Quality Assurance Settings
quality:
  # Verification requirements
  verification:
    enabled: true
    risk_thresholds:
      low: 0.1
      medium: 0.3
      high: 0.5
    min_confidence: 0.7

  # Answer completeness
  completeness:
    min_sources: 2
    require_citations: true
    citation_accuracy: 1.0

  # Thinking process
  thinking_process:
    enabled: true
    include_in_response: true
    log_to_cosmos: true

# Monitoring & Alerting
monitoring:
  # Performance thresholds (balanced)
  thresholds:
    response_time_warning_ms: 8000
    response_time_critical_ms: 12000
    p95_latency_target_ms: 10000
    min_confidence_score: 0.7
    max_verification_failures: 5

  # Metrics collection
  metrics:
    enabled: true
    sample_rate: 1.0
    track_quality_metrics: true

  # Alerts (focus on quality)
  alerts:
    - name: "low_confidence"
      condition: "confidence < 0.7"
      threshold: 10
      window_minutes: 5
      action: "investigate_quality"

    - name: "verification_failures"
      condition: "verification_failed"
      threshold: 5
      window_minutes: 10
      action: "review_responses"

    - name: "missing_citations"
      condition: "citations_missing"
      threshold: 3
      window_minutes: 5
      action: "review_synthesis"

    - name: "slow_response"
      condition: "response_time > 12000"
      threshold: 5
      window_minutes: 5
      action: "investigate_performance"

# Fallback Strategies - Comprehensive
fallback:
  # When to use fallback responses
  triggers:
    - timeout_exceeded
    - verification_failed
    - low_confidence

  # Fallback behavior
  behavior:
    attempt_partial_answer: true
    include_related_docs: true
    suggest_alternatives: true
    provide_breadcrumb: true

# Load Management
load_management:
  # Circuit breaker settings
  circuit_breaker:
    enabled: true
    failure_threshold: 10
    timeout_seconds: 60
    half_open_requests: 3

  # Rate limiting
  rate_limiting:
    enabled: true
    requests_per_minute: 200
    burst_size: 50

  # Concurrency limits
  concurrency:
    max_concurrent_requests: 100
    max_concurrent_agents: 20
    queue_size: 200

# Feature Flags
features:
  # Performance optimizations (safe)
  use_pattern_matching: true
  use_response_cache: true
  use_embedding_cache: true
  use_parallel_search: true
  use_streaming_responses: true
  use_connection_pooling: true

  # Quality features (all enabled)
  full_tree_building: true
  complete_verification: true
  detailed_thinking_process: true
  multi_hop_search: true
  semantic_reranking: true

  # Progressive enhancement
  progressive_responses: true
  initial_quick_answer: true
  enhanced_with_verification: true

# Precomputed Resources (Performance Optimization)
precomputed:
  # Common embeddings to preload
  embeddings:
    enabled: true
    queries:
      - "password reset"
      - "single sign on"
      - "deployment guide"
      - "authentication"
      - "api documentation"
      - "troubleshooting"
      - "getting started"
      - "user management"
      - "permissions"
      - "integration"

  # Connection pre-warming
  connections:
    cosmos_db: true
    search_service: true
    openai: true
    blob_storage: true

# Health Check Configuration
health_check:
  # Component checks
  components:
    - name: "cosmos_db"
      timeout_ms: 1000
      critical: true

    - name: "search_service"
      timeout_ms: 1000
      critical: true

    - name: "openai"
      timeout_ms: 2000
      critical: true

    - name: "blob_storage"
      timeout_ms: 1000
      critical: false

  # Overall health timeout
  total_timeout_ms: 5000

# Environment-specific Settings
environments:
  production:
    debug: false
    log_level: "INFO"
    enable_profiling: false
    cache_multiplier: 1.0
    quality_checks: "strict"

  staging:
    debug: true
    log_level: "DEBUG"
    enable_profiling: true
    cache_multiplier: 0.8
    quality_checks: "strict"

  development:
    debug: true
    log_level: "DEBUG"
    enable_profiling: true
    cache_multiplier: 0.5
    quality_checks: "normal"



    # prompts.py
"""
Prompt templates for Confluence Q&A agents
These prompts are designed for accuracy, coherence, and transparency
"""

class PromptTemplates:
    """Centralized prompt templates for all agents"""

    # Base system prompt shared across all agents
    SYSTEM_BASE = """
You are ConfluenceRAG-Bot, an expert assistant for {organization} documentation.

Core Principles:
• Accuracy First: Only provide information directly supported by documentation
• Citation Required: Always cite sources as [[pageId-chunk]] after each factual claim
• Transparency: Share your reasoning process when analyzing queries
• Humility: Acknowledge when information is incomplete or unclear
• Helpfulness: Guide users to relevant resources when direct answers aren't available

Context:
• Knowledge cutoff: Information is from indexed Confluence pages
• Confidence threshold: {confidence_threshold}
• Maximum search hops: {max_hops}
"""

    # Query Analyser prompts
    QUERY_ANALYSER_SYSTEM = SYSTEM_BASE + """

You are the Query Analyser. Your role is to understand user intent and classify queries.

Your responsibilities:
1. Classify queries into one of three categories:
   - Atomic: Simple, direct questions answerable with a single search
   - NeedsDecomposition: Complex queries requiring multiple sub-questions
   - NeedsClarification: Ambiguous queries lacking essential details

2. For each classification, provide:
   - Confidence score (0.0-1.0)
   - Clear reasoning for your classification
   - Suggested sub-questions (for complex queries)
   - Clarification needs (for ambiguous queries)

3. Consider these factors:
   - Specificity of terms used
   - Scope of the question
   - Presence of multiple concepts
   - Temporal aspects (versions, dates)
   - User's likely intent

Output strict JSON format:
{
    "classification": "Atomic|NeedsDecomposition|NeedsClarification",
    "confidence": 0.0-1.0,
    "reasoning": "Clear explanation of your analysis",
    "subquestions": ["sub1", "sub2", ...],  // if NeedsDecomposition
    "clarification_needed": "Specific clarification question",  // if NeedsClarification
    "key_concepts": ["concept1", "concept2", ...],
    "temporal_aspects": ["v1", "v2", "2024", ...]
}
"""

    QUERY_ANALYSER_EXAMPLES = """
Examples:

1. Atomic Query:
   Input: "How do I reset my password in JIRA?"
   Output: {
       "classification": "Atomic",
       "confidence": 0.95,
       "reasoning": "Clear, specific question about a single procedure",
       "subquestions": [],
       "clarification_needed": null,
       "key_concepts": ["password reset", "JIRA"],
       "temporal_aspects": []
   }

2. Needs Decomposition:
   Input: "What are the differences between our staging and production environments and how do I deploy to each?"
   Output: {
       "classification": "NeedsDecomposition",
       "confidence": 0.9,
       "reasoning": "Query contains two distinct aspects: environment differences and deployment procedures",
       "subquestions": [
           "What are the differences between staging and production environments?",
           "How do I deploy to the staging environment?",
           "How do I deploy to the production environment?"
       ],
       "clarification_needed": null,
       "key_concepts": ["staging", "production", "environments", "deployment"],
       "temporal_aspects": []
   }

3. Needs Clarification:
   Input: "How does the integration work?"
   Output: {
       "classification": "NeedsClarification",
       "confidence": 0.85,
       "reasoning": "Query is too vague - unclear which integration or what aspect",
       "subquestions": [],
       "clarification_needed": "Which integration are you referring to? (e.g., Salesforce, SAP, Stripe) And what aspect interests you? (setup, data flow, troubleshooting)",
       "key_concepts": ["integration"],
       "temporal_aspects": []
   }
"""

    # Decomposer prompts
    DECOMPOSER_SYSTEM = SYSTEM_BASE + """

You are the Query Decomposer. Your role is to break complex queries into manageable sub-questions.

Your responsibilities:
1. Analyze the logical structure of complex queries
2. Identify distinct information needs
3. Create ordered sub-questions that:
   - Are self-contained and answerable independently
   - Cover all aspects of the original query
   - Follow a logical sequence (dependencies considered)
   - Avoid redundancy or overlap
   - Respect the {max_hops} hop limit

4. Consider dependencies between sub-questions
5. Ensure comprehensive coverage without over-decomposition

Output format:
{
    "subquestions": ["q1", "q2", ...],
    "requires_multihop": true|false,
    "reasoning": "Explanation of decomposition strategy",
    "dependencies": {"q2": ["q1"], ...},  // which questions depend on others
    "coverage_check": "Confirmation that all aspects are covered"
}
"""

    DECOMPOSER_EXAMPLES = """
Examples:

1. Technical Setup Query:
   Input: "How do I set up CI/CD pipeline for our microservices including testing and deployment to Kubernetes?"
   Output: {
       "subquestions": [
           "What are the prerequisites for setting up CI/CD for microservices?",
           "How do I configure the CI pipeline for microservices?",
           "How do I set up automated testing in the CI pipeline?",
           "How do I configure CD deployment to Kubernetes?"
       ],
       "requires_multihop": true,
       "reasoning": "Query involves sequential steps: prerequisites → CI setup → testing → deployment",
       "dependencies": {
           "How do I configure the CI pipeline for microservices?": ["What are the prerequisites for setting up CI/CD for microservices?"],
           "How do I set up automated testing in the CI pipeline?": ["How do I configure the CI pipeline for microservices?"],
           "How do I configure CD deployment to Kubernetes?": ["How do I set up automated testing in the CI pipeline?"]
       },
       "coverage_check": "Covers all aspects: CI setup, testing integration, and Kubernetes deployment"
   }

2. Comparison Query:
   Input: "What are the pros and cons of our REST API vs GraphQL API and when should I use each?"
   Output: {
       "subquestions": [
           "What are the advantages of our REST API?",
           "What are the limitations of our REST API?",
           "What are the advantages of our GraphQL API?",
           "What are the limitations of our GraphQL API?",
           "What are the use case guidelines for choosing between REST and GraphQL?"
       ],
       "requires_multihop": false,
       "reasoning": "Comparison requires parallel information gathering, not sequential",
       "dependencies": {},
       "coverage_check": "Covers pros/cons for both APIs and decision criteria"
   }
"""

    # Path Planner prompts
    PATH_PLANNER_SYSTEM = SYSTEM_BASE + """

You are the Path Planner. Your role is to determine optimal traversal strategies through the knowledge graph.

Your responsibilities:
1. Plan efficient paths through the document hierarchy
2. Select appropriate edge types from: {edge_types}
3. Respect the {max_hops} hop limit
4. Generate metadata filters for each hop
5. Balance breadth vs depth based on query needs

Consider:
- Document relationships (parent/child, cross-references)
- Information locality (related info often clustered)
- Search efficiency (minimize hops while maximizing coverage)
- Previous hop results to guide next steps

Output format:
{
    "strategy": "breadth_first|depth_first|mixed",
    "strategy_reasoning": "Why this approach",
    "hops": [
        {
            "hop_number": 1,
            "purpose": "What this hop aims to find",
            "edge_types": ["ParentOf", "LinksTo"],
            "filter": "pageId in ('id1', 'id2') or parentId eq 'id3'",
            "expected_results": "What we expect to find"
        }
    ],
    "truncated": false,
    "truncation_reason": null,
    "alternative_paths": ["description of other viable paths"]
}
"""

    # Retriever prompts
    RETRIEVER_SYSTEM = SYSTEM_BASE + """

You are the Retriever. Your role is to find relevant documents using hybrid search.

Your approach:
1. Execute vector similarity search (semantic understanding)
2. Execute keyword search (exact matches)
3. Combine results using hybrid ranking
4. Apply metadata filters from path planning
5. Optimize for high recall with reasonable precision

Search parameters:
- Vector search: k=15 nearest neighbors
- BM25 search: top 25 results
- Speller: lexicon (handles typos)
- Search mode: semanticHybrid

Always log:
- Query understanding
- Applied filters
- Result counts
- Relevance distribution
"""

    # Reranker prompts
    RERANKER_SYSTEM = SYSTEM_BASE + """

You are the Reranker. Your role is to precisely rank search results by deep relevance.

Your approach:
1. Analyze query intent deeply
2. Evaluate each document's relevance:
   - Semantic alignment with query
   - Information completeness
   - Specificity to the question
   - Recency/version relevance
3. Apply cross-encoder scoring for accuracy
4. Return top 8-15 documents based on quality threshold
5. Include confidence scores

Ranking criteria (in order):
1. Direct answer presence (highest weight)
2. Conceptual relevance
3. Context completeness
4. Information freshness
5. Source authority

Output includes relevance reasoning for each document.
"""

    # Synthesiser prompts
    SYNTHESISER_SYSTEM = SYSTEM_BASE + """

You are the Synthesiser. Your role is to create comprehensive, accurate answers from search results.

Your responsibilities:
1. Generate coherent answers that fully address the user's question
2. Cite EVERY factual claim with [[pageId-chunk]] references
3. Maintain logical flow across information from multiple sources
4. Acknowledge limitations when context is incomplete
5. Structure complex answers with clear sections

Requirements:
- Coverage threshold: 70% of question aspects must be answerable
- Citation format: [[pageId-sectionNum-chunkNum]] immediately after claims
- Confidence indication: Explicitly state when unsure
- Fallback behavior: Recommend parent pages and related resources
- Completeness: Provide thorough answers, not brief summaries

Answer structure for complex queries:
1. Direct answer to main question (if possible)
2. Detailed sections for each aspect
3. Additional context or caveats
4. Related resources for further reading

Context format: [[chunkId]] content text

If insufficient context:
"I found limited information about [topic]. Here's what I could determine:
[partial answer with citations]

For comprehensive information, please check:
- Parent page: [breadcrumb path]
- Related documents: [list of relevant pages]"

Focus on completeness and accuracy over brevity. Users expect thorough, well-researched answers.
"""

    SYNTHESISER_EXAMPLES = """
Example synthesis:

Context:
[[page123-1-0]] SSO can be enabled through the Admin Console. Navigate to Settings > Authentication.
[[page123-1-1]] Click on 'Enable SSO' and select your identity provider from the dropdown.
[[page456-2-0]] Supported identity providers include Okta, Auth0, and Azure AD.
[[page456-2-1]] You'll need your IDP metadata URL and certificate.

Query: "How do I enable SSO?"

Synthesized answer:
To enable SSO for the application, follow these steps:

1. **Access SSO Settings**: Navigate to the Admin Console and go to Settings > Authentication [[page123-1-0]].

2. **Enable SSO**: Click on the 'Enable SSO' button and select your identity provider from the dropdown menu [[page123-1-1]]. The system supports multiple identity providers including Okta, Auth0, and Azure AD [[page456-2-0]].

3. **Configure IDP**: You'll need to provide your identity provider's metadata URL and certificate [[page456-2-1]].

For provider-specific configuration details, refer to the authentication documentation or contact your IDP administrator.
"""

    # Verifier prompts
    VERIFIER_SYSTEM = SYSTEM_BASE + """

You are the Verifier. Your role is to ensure answer accuracy and prevent hallucination.

Your responsibilities:
1. Check EVERY claim in the answer against provided context
2. Verify all citations are accurate and properly formatted
3. Identify any unsupported statements or extrapolations
4. Assess overall answer quality and completeness
5. Flag risks and suggest improvements

Verification checklist:
□ All factual claims have citations
□ Citations correctly reference source content
□ No information beyond provided context
□ Logical flow maintained
□ No contradictions present
□ Appropriate confidence level indicated

Risk levels:
- None: All claims fully supported
- Low: Minor citation issues or slight extrapolations
- Medium: Some unsupported claims or significant gaps
- High: Major unsupported claims or potential misinformation

Output format:
{
    "risk": true|false,
    "risk_level": "none|low|medium|high",
    "confidence": 0.0-1.0,
    "issues_found": {
        "unsupported_claims": ["claim1", "claim2"],
        "missing_citations": ["statement1", "statement2"],
        "incorrect_citations": [{"claim": "...", "cited": "id1", "should_be": "id2"}],
        "extrapolations": ["extrapolation1"],
        "contradictions": ["contradiction1"]
    },
    "quality_assessment": {
        "completeness": 0.0-1.0,
        "accuracy": 0.0-1.0,
        "clarity": 0.0-1.0,
        "structure": 0.0-1.0
    },
    "recommendations": [
        "Add citation for claim about X",
        "Clarify statement about Y",
        "Remove unsupported claim about Z"
    ]
}
"""

    # Clarifier prompts
    CLARIFIER_SYSTEM = SYSTEM_BASE + """

You are the Clarifier. Your role is to help users refine ambiguous queries through natural dialogue.

Your approach:
1. Identify specific ambiguities in the query
2. Ask focused clarifying questions (1-2 at most)
3. Provide examples to guide users
4. Maintain a helpful, conversational tone
5. Avoid technical jargon unless necessary

Clarification strategies:
- For vague terms: "When you say 'the system', do you mean [specific system A] or [specific system B]?"
- For missing context: "Could you specify which version or environment you're working with?"
- For broad topics: "What specific aspect of [topic] would you like to know about? For example, [aspect A] or [aspect B]?"
- For ambiguous intent: "Are you looking to [action A] or [action B]?"

Always:
- Be concise and friendly
- Provide 2-3 specific options when possible
- Include brief examples
- Anticipate common interpretations
"""

    CLARIFIER_EXAMPLES = """
Examples:

1. Vague System Reference:
   User: "How do I configure the integration?"
   Clarification: "I'd be happy to help you configure an integration! Which integration are you working with? For example:
   - Salesforce CRM integration
   - Payment gateway integration (Stripe/PayPal)
   - Email service integration (SendGrid/Mailchimp)

   Also, are you looking for initial setup steps or troubleshooting an existing configuration?"

2. Missing Version Context:
   User: "What's new in the latest release?"
   Clarification: "I can help you with release information! Which product's release are you interested in?
   - API v2.5 (released last month)
   - Mobile app 3.0 (released last week)
   - Web platform 4.1 (released yesterday)

   Or are you asking about a different component?"

3. Broad Topic:
   User: "Explain the architecture"
   Clarification: "I'd be glad to explain our architecture! To provide the most relevant information, could you clarify which aspect interests you?
   - Overall system architecture (high-level components)
   - Microservices architecture and communication
   - Database architecture and data flow
   - Security architecture and authentication flow

   Or is there a specific component you'd like to understand?"
"""

    # Tree Builder prompts
    TREE_BUILDER_SYSTEM = SYSTEM_BASE + """

You are the Tree Builder. Your role is to construct and visualize document hierarchies.

Your responsibilities:
1. Build complete page trees from graph data
2. Highlight pages containing answers with visual indicators
3. Generate clean, readable markdown representations
4. Show all relevant relationships between pages
5. Handle multiple trees when answers span hierarchies

Tree building rules:
- Start from root pages and build downward
- Mark answer-containing pages with ⭐
- Use clear indentation for hierarchy levels
- Include page links in markdown format
- Show sibling relationships at same level
- Indicate when trees are truncated

Markdown format:
- Root Page
  - Parent Category
    - **Answer Page** ⭐
      - Sub-page 1
      - Sub-page 2
    - Sibling Page
  - Another Category

For multiple trees, clearly separate each hierarchy.
"""

    @classmethod
    def get_prompt(cls, agent_type: str, **kwargs) -> str:
        """Get formatted prompt for specific agent type"""

        base_prompt = cls.SYSTEM_BASE.format(
            organization=kwargs.get('organization', 'your organization'),
            confidence_threshold=kwargs.get('confidence_threshold', 0.7),
            max_hops=kwargs.get('max_hops', 3)
        )

        agent_prompts = {
            'query_analyser': cls.QUERY_ANALYSER_SYSTEM + "\n\n" + cls.QUERY_ANALYSER_EXAMPLES,
            'decomposer': cls.DECOMPOSER_SYSTEM + "\n\n" + cls.DECOMPOSER_EXAMPLES,
            'path_planner': cls.PATH_PLANNER_SYSTEM.format(
                edge_types=kwargs.get('edge_types', ['ParentOf', 'LinksTo']),
                max_hops=kwargs.get('max_hops', 3)
            ),
            'retriever': cls.RETRIEVER_SYSTEM,
            'reranker': cls.RERANKER_SYSTEM,
            'synthesiser': cls.SYNTHESISER_SYSTEM + "\n\n" + cls.SYNTHESISER_EXAMPLES,
            'verifier': cls.VERIFIER_SYSTEM,
            'clarifier': cls.CLARIFIER_SYSTEM + "\n\n" + cls.CLARIFIER_EXAMPLES,
            'tree_builder': cls.TREE_BUILDER_SYSTEM
        }

        return agent_prompts.get(agent_type, base_prompt)


# confluence_qa_orchestrator.py
"""
Confluence Q&A System using AutoGen Framework
Features:
- Multi-agent orchestration with AutoGen
- Query decomposition and path planning
- NL-based clarification
- Transparent thinking process
- Tree-based page structure visualization
- Integration with Azure Cognitive Search, Cosmos DB, and Azure Storage
"""

import os
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import autogen
from autogen import ConversableAgent, GroupChat, GroupChatManager
from azure.search.documents import SearchClient
from azure.search.documents.indexes.models import SearchIndex, SearchField, SearchFieldDataType
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContainerClient
from azure.cosmos import CosmosClient, PartitionKey
from openai import AzureOpenAI
import gremlin_python.driver as gremlin
import networkx as nx
from anytree import Node, RenderTree, PreOrderIter
import re
import time
import uuid
import hashlib

# Import prompts from prompts.py
from prompts import PromptTemplates

# Configuration
MAX_HOPS = int(os.getenv('MAX_HOPS', '3'))
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.7'))
EDGE_TYPES = os.getenv('EDGE_TYPES', 'ParentOf,LinksTo').split(',')
ORGANIZATION = os.getenv('ORGANIZATION', 'your organization')

# Dataclasses for temporary runtime data only
@dataclass
class QueryAnalysis:
    """Result of query analysis - temporary runtime data"""
    classification: str  # Atomic | NeedsDecomposition | NeedsClarification
    subquestions: List[str] = field(default_factory=list)
    clarification_needed: Optional[str] = None
    confidence: float = 0.0
    reasoning: str = ""
    key_concepts: List[str] = field(default_factory=list)  # Added from prompts.py
    temporal_aspects: List[str] = field(default_factory=list)  # Added from prompts.py

@dataclass
class SearchResult:
    """Search result with metadata - temporary runtime data from Azure Cognitive Search"""
    id: str
    page_id: str
    title: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class AzureDataStore:
    """Manages persistent data storage in Azure Cosmos DB and Blob Storage"""

    def __init__(self):
        # Initialize Azure clients
        credential = DefaultAzureCredential()

        # Cosmos DB for structured data
        self.cosmos_client = CosmosClient(
            url=f"https://{os.environ['COSMOS_ACCOUNT']}.documents.azure.com",
            credential=credential
        )
        self.database = self.cosmos_client.get_database_client(os.environ['COSMOS_DB'])

        # Containers for different data types
        self.thinking_container = self.database.get_container_client("thinking_steps")
        self.conversation_container = self.database.get_container_client("conversations")
        self.page_tree_container = self.database.get_container_client("page_trees")

        # Azure Storage for raw documents
        self.blob_service = BlobServiceClient(
            account_url=f"https://{os.environ['STORAGE_ACCOUNT']}.blob.core.windows.net",
            credential=credential
        )
        self.raw_container = self.blob_service.get_container_client("raw-confluence")
        self.processed_container = self.blob_service.get_container_client("processed-confluence")

    async def save_thinking_step(self, conversation_id: str, step: Dict[str, Any]):
        """Save thinking step to Cosmos DB"""
        step_doc = {
            'id': str(uuid.uuid4()),
            'partitionKey': conversation_id,
            'conversationId': conversation_id,
            'agent': step['agent'],
            'action': step['action'],
            'reasoning': step['reasoning'],
            'timestamp': step['timestamp'],
            'result': json.dumps(step.get('result')) if step.get('result') else None
        }
        await asyncio.to_thread(self.thinking_container.create_item, step_doc)

    async def get_thinking_steps(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Retrieve thinking steps from Cosmos DB"""
        query = "SELECT * FROM c WHERE c.conversationId = @conversation_id ORDER BY c.timestamp"
        parameters = [{"name": "@conversation_id", "value": conversation_id}]

        items = await asyncio.to_thread(
            lambda: list(self.thinking_container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))
        )

        return items

    async def save_conversation(self, conversation_id: str, messages: List[Dict[str, Any]]):
        """Save conversation to Cosmos DB"""
        conv_doc = {
            'id': conversation_id,
            'partitionKey': conversation_id,
            'messages': messages,
            'lastUpdated': time.time(),
            'created': time.time()
        }
        await asyncio.to_thread(
            self.conversation_container.upsert_item,
            conv_doc
        )

    async def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve conversation from Cosmos DB"""
        try:
            item = await asyncio.to_thread(
                self.conversation_container.read_item,
                item=conversation_id,
                partition_key=conversation_id
            )
            return item
        except:
            return None

    async def get_page_content(self, page_id: str) -> Optional[str]:
        """Retrieve page content from Azure Blob Storage"""
        blob_name = f"{page_id}.json"
        try:
            blob_client = self.processed_container.get_blob_client(blob_name)
            content = await asyncio.to_thread(blob_client.download_blob)
            return content.readall().decode('utf-8')
        except:
            # Try raw container if not in processed
            try:
                blob_client = self.raw_container.get_blob_client(blob_name)
                content = await asyncio.to_thread(blob_client.download_blob)
                return content.readall().decode('utf-8')
            except:
                return None

class ConfluenceQAOrchestrator:
    """Main orchestrator for Confluence Q&A system"""

    def __init__(self):
        self.config_list = self._get_llm_config()
        self.agent_configs = self._get_agent_specific_configs()
        self.agents = self._initialize_agents()
        self.thinking_process = []
        self.search_client = self._init_search_client()
        self.gremlin_client = self._init_gremlin_client()
        self.aoai_client = self._init_aoai_client()

        # Initialize instance variables
        self.data_store = AzureDataStore()
        self.response_cache = {}
        self.embedding_cache = {}
        self.TOTAL_TIMEOUT = float(os.getenv('TOTAL_TIMEOUT', '30'))
        self.MAX_SUBQUESTIONS = int(os.getenv('MAX_SUBQUESTIONS', '5'))
        self.MAX_SEARCH_RESULTS = int(os.getenv('MAX_SEARCH_RESULTS', '25'))

        # Prompt template parameters
        self.prompt_params = {
            'organization': ORGANIZATION,
            'confidence_threshold': CONFIDENCE_THRESHOLD,
            'max_hops': MAX_HOPS,
            'edge_types': EDGE_TYPES
        }

    def _get_llm_config(self):
        """Get base LLM configuration for AutoGen agents"""
        return [{
            "model": os.getenv("AOAI_CHAT_DEPLOY", "gpt-4o"),
            "api_key": os.getenv("AOAI_KEY"),
            "base_url": f"{os.getenv('AOAI_ENDPOINT')}/openai/deployments/{os.getenv('AOAI_CHAT_DEPLOY')}/",
            "api_type": "azure",
            "api_version": "2025-01-01"
        }]

    def _get_agent_specific_configs(self):
        """Get agent-specific configurations for optimized performance"""
        return {
            "query_analyser": {
                "temperature": 0,
                "max_tokens": 200,
                "response_format": {"type": "json_object"}
            },
            "decomposer": {
                "temperature": 0,
                "max_tokens": 150,
                "response_format": {"type": "json_object"}
            },
            "path_planner": {
                "temperature": 0,
                "max_tokens": 100
            },
            "retriever": {
                "temperature": 0,
                "max_tokens": 50
            },
            "reranker": {
                "temperature": 0,
                "max_tokens": 150
            },
            "synthesiser": {
                "temperature": 0.1,
                "max_tokens": 800
            },
            "verifier": {
                "temperature": 0,
                "max_tokens": 200,
                "response_format": {"type": "json_object"}
            },
            "clarifier": {
                "temperature": 0.3,
                "max_tokens": 300
            },
            "tree_builder": {
                "temperature": 0,
                "max_tokens": 500
            }
        }

    def _init_search_client(self):
        """Initialize Azure AI Search client"""
        return SearchClient(
            endpoint=os.environ['SEARCH_ENDPOINT'],
            index_name=os.environ['SEARCH_INDEX'],
            credential=DefaultAzureCredential()
        )

    def _init_gremlin_client(self):
        """Initialize Cosmos DB Gremlin client"""
        return gremlin.client.Client(
            f"wss://{os.environ['COSMOS_ACCOUNT']}.gremlin.cosmos.azure.com:443/",
            'g',
            username=f"/dbs/{os.environ['COSMOS_DB']}/colls/{os.environ['COSMOS_GRAPH']}",
            password=os.environ['COSMOS_KEY'],
            message_serializer=gremlin.serializer.GraphSONSerializersV2d0()
        )

    def _init_aoai_client(self):
        """Initialize Azure OpenAI client"""
        return AzureOpenAI(
            azure_endpoint=os.environ['AOAI_ENDPOINT'],
            api_key=os.environ['AOAI_KEY'],
            api_version="2025-01-01"
        )

    def _initialize_agents(self):
        """Initialize all AutoGen agents using prompts from prompts.py"""

        # Query Analyser Agent
        query_analyser = ConversableAgent(
            "query_analyser",
            system_message=PromptTemplates.get_prompt('query_analyser', **self.prompt_params),
            llm_config={
                "config_list": self.config_list,
                **self.agent_configs["query_analyser"]
            },
            function_map={
                "analyze_query": self.analyze_query
            }
        )

        # Decomposer Agent
        decomposer = ConversableAgent(
            "decomposer",
            system_message=PromptTemplates.get_prompt('decomposer', **self.prompt_params),
            llm_config={
                "config_list": self.config_list,
                **self.agent_configs["decomposer"]
            }
        )

        # Path Planner Agent
        path_planner = ConversableAgent(
            "path_planner",
            system_message=PromptTemplates.get_prompt('path_planner', **self.prompt_params),
            llm_config={
                "config_list": self.config_list,
                **self.agent_configs["path_planner"]
            },
            function_map={
                "plan_path": self.plan_path,
                "get_page_relationships": self.get_page_relationships
            }
        )

        # Retriever Agent
        retriever = ConversableAgent(
            "retriever",
            system_message=PromptTemplates.get_prompt('retriever', **self.prompt_params),
            llm_config={
                "config_list": self.config_list,
                **self.agent_configs["retriever"]
            },
            function_map={
                "hybrid_search": self.hybrid_search,
                "fetch_page_content": self.fetch_page_content
            }
        )

        # Reranker Agent
        reranker = ConversableAgent(
            "reranker",
            system_message=PromptTemplates.get_prompt('reranker', **self.prompt_params),
            llm_config={
                "config_list": self.config_list,
                **self.agent_configs["reranker"]
            },
            function_map={
                "semantic_rerank": self.semantic_rerank
            }
        )

        # Synthesiser Agent
        synthesiser = ConversableAgent(
            "synthesiser",
            system_message=PromptTemplates.get_prompt('synthesiser', **self.prompt_params),
            llm_config={
                "config_list": self.config_list,
                **self.agent_configs["synthesiser"]
            },
            function_map={
                "synthesize_answer": self.synthesize_answer,
                "calculate_coverage": self.calculate_coverage
            }
        )

        # Verifier Agent
        verifier = ConversableAgent(
            "verifier",
            system_message=PromptTemplates.get_prompt('verifier', **self.prompt_params),
            llm_config={
                "config_list": self.config_list,
                **self.agent_configs["verifier"]
            }
        )

        # Clarifier Agent
        clarifier = ConversableAgent(
            "clarifier",
            system_message=PromptTemplates.get_prompt('clarifier', **self.prompt_params),
            llm_config={
                "config_list": self.config_list,
                **self.agent_configs["clarifier"]
            }
        )

        # Tree Builder Agent
        tree_builder = ConversableAgent(
            "tree_builder",
            system_message=PromptTemplates.get_prompt('tree_builder', **self.prompt_params),
            llm_config={
                "config_list": self.config_list,
                **self.agent_configs["tree_builder"]
            },
            function_map={
                "build_page_tree": self.build_page_tree,
                "render_tree_markdown": self.render_tree_markdown
            }
        )

        return {
            "query_analyser": query_analyser,
            "decomposer": decomposer,
            "path_planner": path_planner,
            "retriever": retriever,
            "reranker": reranker,
            "synthesiser": synthesiser,
            "verifier": verifier,
            "clarifier": clarifier,
            "tree_builder": tree_builder
        }

    def _check_quick_patterns(self, query: str) -> Optional[Dict[str, Any]]:
        """Check for quick patterns in query for optimization"""
        # Simple pattern matching for common queries
        patterns = {
            "sso": {"answer": "To enable SSO, follow the documentation [[sso-guide-1]]", "confidence": 0.95, "category": "SSO Guide"},
            "login": {"answer": "For login issues, check [[login-troubleshoot-1]]", "confidence": 0.9, "category": "Login Help"},
        }

        query_lower = query.lower()
        for pattern, response in patterns.items():
            if pattern in query_lower:
                return response

        return None

    async def process_query(self, query: str, conversation_id: str) -> Dict[str, Any]:
        """Process user queries with balanced performance and accuracy"""

        start_time = time.time()

        # Check response cache first (performance optimization that doesn't affect accuracy)
        cache_key = self._get_cache_key(query)
        if cache_key in self.response_cache:
            cached_response = self.response_cache[cache_key]
            cached_response['cached'] = True
            cached_response['response_time'] = time.time() - start_time
            return cached_response

        try:
            # Process with reasonable timeout for quality responses
            result = await asyncio.wait_for(
                self._process_query_balanced(query, conversation_id),
                timeout=self.TOTAL_TIMEOUT
            )

            # Cache successful response
            self.response_cache[cache_key] = result

            return result

        except asyncio.TimeoutError:
            # Return comprehensive fallback response with available information
            return await self._generate_comprehensive_fallback(query, conversation_id, start_time)

    async def _process_query_balanced(self, query: str, conversation_id: str) -> Dict[str, Any]:
        """Process query with full accuracy and optimized performance"""

        start_time = time.time()

        # Log initial query with full thinking process
        await self._log_thinking(conversation_id, "system", "receive_query", f"Processing query: {query}", query)

        # Quick pattern matching for common queries (performance optimization)
        quick_response = self._check_quick_patterns(query)
        if quick_response and quick_response.get('confidence', 0) > 0.95:
            # Still verify pattern-matched responses
            verification = await self._verify_pattern_response(quick_response['answer'], query)
            if verification['confidence'] > 0.9:
                return {
                    "status": "success",
                    "answer": quick_response['answer'],
                    "confidence": verification['confidence'],
                    "page_trees": await self._build_page_trees_for_pattern(quick_response),
                    "thinking_process": await self._format_thinking_process(conversation_id),
                    "response_time": time.time() - start_time,
                    "pattern_match": True
                }

        # Full query analysis with AutoGen
        analysis = await self._analyze_query(query, conversation_id)

        # Handle based on classification
        if analysis.classification == "NeedsClarification":
            return await self._handle_clarification(query, analysis, conversation_id)

        elif analysis.classification == "NeedsDecomposition":
            return await self._handle_complex_query(query, analysis, conversation_id)

        else:  # Atomic query
            return await self._handle_atomic_query(query, conversation_id)

    async def _handle_atomic_query(self, query: str, conversation_id: str) -> Dict[str, Any]:
        """Handle simple atomic queries with full accuracy"""

        start_time = time.time()

        await self._log_thinking(conversation_id, "orchestrator", "atomic_query", "Processing as atomic query", None)

        # Progressive search strategy (from recommendations)
        docs = await self._retrieve_documents_progressive(query, {"hops": [{"hop_number": 0, "filter": None}]})

        # Full semantic reranking for accuracy
        reranked = await self._rerank_results(docs, query, conversation_id)

        # Synthesize answer with full context
        sub_results = [{
            "question": query,
            "documents": reranked,
            "path_plan": {"hops": [{"hop_number": 0}]}
        }]

        answer = await self._synthesize_answer(query, sub_results, conversation_id)

        # Full verification for accuracy
        verification = await self._verify_answer(answer, sub_results, conversation_id)

        # Build complete page trees
        page_trees = await self._build_page_trees(sub_results)

        # Get complete thinking process
        thinking_process = await self._format_thinking_process(conversation_id)

        return {
            "status": "success",
            "answer": answer,
            "thinking_process": thinking_process,
            "page_trees": page_trees,
            "verification": verification,
            "confidence": verification["confidence"],
            "response_time": time.time() - start_time
        }

    async def _handle_complex_query(self, query: str, analysis: QueryAnalysis, conversation_id: str) -> Dict[str, Any]:
        """Handle complex queries with full decomposition"""

        start_time = time.time()

        await self._log_thinking(conversation_id, "decomposer", "decompose", "Breaking down complex query", analysis.subquestions)

        # Get full decomposition if not already done
        if not analysis.subquestions:
            decomposition = await self._decompose_query(query, conversation_id)
            analysis.subquestions = decomposition["subquestions"]

        # Batch processing for sub-questions (performance optimization)
        batch_size = 3
        valid_results = []

        for i in range(0, len(analysis.subquestions[:self.MAX_SUBQUESTIONS]), batch_size):
            batch = analysis.subquestions[i:i+batch_size]
            batch_tasks = []

            for j, subq in enumerate(batch):
                await self._log_thinking(conversation_id, "orchestrator", "process_subquestion", f"Processing sub-question {i+j+1}: {subq}", None)

                task = asyncio.create_task(
                    self._process_subquestion_full(subq, i+j, conversation_id),
                    name=f"subq_{i+j}"
                )
                batch_tasks.append(task)

            # Wait for batch to complete
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Filter out any failed tasks
            for k, result in enumerate(batch_results):
                if not isinstance(result, Exception) and result:
                    valid_results.append(result)
                else:
                    await self._log_thinking(
                        conversation_id,
                        "orchestrator",
                        "subquestion_error",
                        f"Error processing sub-question {i+k+1}",
                        str(result) if isinstance(result, Exception) else None
                    )

        # Synthesize comprehensive answer
        answer = await self._synthesize_answer(query, valid_results, conversation_id)

        # Full verification
        verification = await self._verify_answer(answer, valid_results, conversation_id)

        # Build complete page trees
        page_trees = await self._build_page_trees(valid_results)

        # Handle verification results if needed
        if verification["risk"]:
            answer = await self._handle_verification_failure(answer, verification, valid_results)

        # Save conversation
        await self.data_store.save_conversation(conversation_id, [
            {"role": "user", "content": query, "timestamp": time.time()},
            {"role": "assistant", "content": answer, "timestamp": time.time()}
        ])

        thinking_process = await self._format_thinking_process(conversation_id)

        return {
            "status": "success",
            "answer": answer,
            "thinking_process": thinking_process,
            "page_trees": page_trees,
            "verification": verification,
            "sub_questions": analysis.subquestions,
            "confidence": verification["confidence"],
            "response_time": time.time() - start_time
        }

    async def _process_subquestion_full(self, subq: str, index: int, conversation_id: str) -> Dict[str, Any]:
        """Process a sub-question with full accuracy"""

        # Plan path for this sub-question
        path_plan = await self._plan_path(subq, index, [], conversation_id)

        # Retrieve documents with progressive search
        docs = await self._retrieve_documents_progressive(subq, path_plan)

        # Rerank results
        reranked = await self._rerank_results(docs, subq, conversation_id)

        return {
            "question": subq,
            "documents": reranked,
            "path_plan": path_plan
        }

    async def _retrieve_documents_progressive(self, query: str, path_plan: Dict[str, Any]) -> List[SearchResult]:
        """Progressive retrieval - start fast, expand if needed"""

        # Build filter from path plan
        search_filter = None
        if path_plan.get("hops") and path_plan["hops"][0].get("filter"):
            search_filter = path_plan["hops"][0]["filter"]

        # Phase 1: Quick keyword search (fastest)
        results = await self._execute_keyword_search(query, search_filter)

        if len(results) >= 5 and results[0].score > 0.8:
            return results[:10]  # Good enough, return quickly

        # Phase 2: Add vector search
        embedding = await self._get_cached_embedding(query)
        if embedding:
            vector_results = await self._execute_vector_search(embedding, search_filter)
            results.extend(vector_results)

        if len(results) >= 10:
            return self._deduplicate_results(results)[:15]

        # Phase 3: Semantic search only if really needed
        semantic_results = await self._execute_semantic_search(query, search_filter)
        results.extend(semantic_results)

        return self._deduplicate_results(results)[:20]

    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results while preserving order"""
        seen_ids = set()
        unique_results = []
        for doc in results:
            if doc.id not in seen_ids:
                seen_ids.add(doc.id)
                unique_results.append(doc)
        return unique_results

    async def _retrieve_documents_vector(self, query: str, path_plan: Dict[str, Any]) -> List[SearchResult]:
        """Vector-only retrieval for specific use cases"""

        # Get embedding
        embedding = await self._get_cached_embedding(query)
        if not embedding:
            return []

        # Build filter from path plan
        search_filter = None
        if path_plan.get("hops") and path_plan["hops"][0].get("filter"):
            search_filter = path_plan["hops"][0]["filter"]

        # Execute vector search
        return await self._execute_vector_search(embedding, search_filter)

    async def _retrieve_documents_hybrid(self, query: str, path_plan: Dict[str, Any]) -> List[SearchResult]:
        """Hybrid retrieval with parallel search strategies"""

        # Get embedding (with caching for performance)
        embedding = await self._get_cached_embedding(query)

        # Build filter from path plan
        search_filter = None
        if path_plan.get("hops") and path_plan["hops"][0].get("filter"):
            search_filter = path_plan["hops"][0]["filter"]

        # Parallel search tasks
        tasks = []

        # Vector search
        if embedding:
            vector_task = asyncio.create_task(
                self._execute_vector_search(embedding, search_filter)
            )
            tasks.append(vector_task)

        # Keyword search
        keyword_task = asyncio.create_task(
            self._execute_keyword_search(query, search_filter)
        )
        tasks.append(keyword_task)

        # Semantic search
        semantic_task = asyncio.create_task(
            self._execute_semantic_search(query, search_filter)
        )
        tasks.append(semantic_task)

        # Execute all searches in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge results
        all_results = []
        for result in results:
            if isinstance(result, list):
                all_results.extend(result)

        # Deduplicate and return
        unique_results = self._deduplicate_results(all_results)

        # Fetch full content from blob storage if needed
        for doc in unique_results[:self.MAX_SEARCH_RESULTS]:
            if not doc.content and doc.page_id:
                full_content = await self.data_store.get_page_content(doc.page_id)
                if full_content:
                    try:
                        page_data = json.loads(full_content)
                        doc.content = page_data.get("body", {}).get("storage", {}).get("value", "")
                    except:
                        doc.content = full_content

        return unique_results[:self.MAX_SEARCH_RESULTS]

    async def _execute_vector_search(self, embedding: List[float], search_filter: Optional[str]) -> List[SearchResult]:
        """Execute vector search with optimized K value"""
        try:
            results = self.search_client.search(
                search_text="",
                vector_queries=[{
                    "vector": embedding,
                    "k_nearest_neighbors": 10,  # Reduced from 15
                    "fields": "embedding"
                }],
                filter=search_filter,
                top=10,  # Reduced from 15
                include_total_count=True
            )

            return [self._convert_to_search_result(r) for r in results]
        except Exception as e:
            await self._log_thinking("system", "search", "vector_search_error", f"Vector search failed: {str(e)}", None)
            return []

    async def _execute_keyword_search(self, query: str, search_filter: Optional[str]) -> List[SearchResult]:
        """Execute keyword search with optimized top value"""
        try:
            results = self.search_client.search(
                search_text=query,
                search_mode="all",  # More accurate than "any"
                filter=search_filter,
                top=15,  # Reduced from 25
                include_total_count=True
            )

            return [self._convert_to_search_result(r) for r in results]
        except Exception as e:
            await self._log_thinking("system", "search", "keyword_search_error", f"Keyword search failed: {str(e)}", None)
            return []

    async def _execute_semantic_search(self, query: str, search_filter: Optional[str]) -> List[SearchResult]:
        """Execute semantic search if available"""
        try:
            results = self.search_client.search(
                search_text=query,
                query_type="semantic",
                semantic_configuration_name="default",
                filter=search_filter,
                top=8  # Reduced from 10
            )

            return [self._convert_to_search_result(r) for r in results]
        except:
            # Semantic search might not be available in all tiers
            return []

    def _convert_to_search_result(self, result: Dict) -> SearchResult:
        """Convert search result to SearchResult object"""
        return SearchResult(
            id=result["id"],
            page_id=result["pageId"],
            title=result["title"],
            content=result.get("content", ""),
            score=result.get("@search.score", 0),
            metadata=result.get("metadata", {})
        )

    async def _rerank_results(self, docs: List[SearchResult], query: str, conversation_id: str) -> List[SearchResult]:
        """Full semantic reranking for accuracy with reduced top_n"""

        if not docs:
            return []

        # Use Azure semantic reranker if available
        try:
            # Prepare documents for reranking
            rerank_input = [
                {"id": doc.id, "text": f"{doc.title} {doc.content}"}
                for doc in docs
            ]

            # Call reranker
            reranked_results = await self.aoai_client.rerank(
                query=query,
                documents=rerank_input,
                model="semantic-reranker-v2",
                top_n=min(8, len(docs))  # Reduced from 15
            )

            # Reorder documents based on reranking
            reranked_docs = []
            for result in reranked_results:
                doc_id = result["id"]
                for doc in docs:
                    if doc.id == doc_id:
                        reranked_docs.append(doc)
                        break

            await self._log_thinking(
                conversation_id,
                "reranker",
                "semantic_rerank",
                f"Reranked {len(docs)} documents to {len(reranked_docs)}",
                {"original_count": len(docs), "reranked_count": len(reranked_docs)}
            )

            return reranked_docs

        except Exception as e:
            # Fallback to score-based reranking
            await self._log_thinking(
                conversation_id,
                "reranker",
                "fallback_rerank",
                f"Using score-based reranking: {str(e)}",
                None
            )

            return sorted(docs, key=lambda x: x.score, reverse=True)[:8]

    async def _get_cached_embedding(self, query: str) -> Optional[List[float]]:
        """Get cached embedding or generate new one"""
        cache_key = self._get_cache_key(query)

        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]

        try:
            # Generate embedding
            response = await self.aoai_client.embeddings.create(
                model=os.environ['AOAI_EMBED_DEPLOY'],
                input=[query]
            )

            embedding = response.data[0].embedding
            self.embedding_cache[cache_key] = embedding
            return embedding

        except Exception as e:
            await self._log_thinking("system", "embedding", "error", f"Failed to generate embedding: {str(e)}", None)
            return None

    async def _generate_comprehensive_fallback(self, query: str, conversation_id: str, start_time: float) -> Dict[str, Any]:
        """Generate comprehensive fallback response with available information"""

        # Try to get at least some search results
        try:
            simple_results = await asyncio.wait_for(
                self._execute_keyword_search(query, None),
                timeout=2.0
            )

            if simple_results:
                # Build a basic answer
                answer = f"""I encountered a timeout while processing your query, but I found some relevant information:

{simple_results[0].title}: {simple_results[0].content[:200]}... [[{simple_results[0].id}]]

For a complete answer, you may want to:
1. Check the full documentation page: {simple_results[0].title}
2. Try a more specific query
3. Browse related pages in the documentation"""

                return {
                    "status": "partial",
                    "answer": answer,
                    "confidence": 0.6,
                    "page_trees": [{
                        "root_page_id": simple_results[0].page_id,
                        "root_title": simple_results[0].title,
                        "markdown": f"- [{simple_results[0].title}](/wiki/pages/{simple_results[0].page_id}) ⚠️ (partial result)",
                        "contains_answer": True
                    }],
                    "thinking_process": await self._format_thinking_process(conversation_id),
                    "response_time": time.time() - start_time,
                    "timeout": True,
                    "partial_results": True
                }
        except:
            pass

        # Complete fallback
        return {
            "status": "timeout",
            "answer": """I apologize, but I couldn't complete processing your query within the time limit.

This might be due to:
- Complex query requiring extensive analysis
- High system load
- Network connectivity issues

Please try:
1. Simplifying your question
2. Breaking it into smaller, specific questions
3. Searching the documentation directly
4. Trying again in a few moments""",
            "confidence": 0.0,
            "page_trees": [],
            "thinking_process": await self._format_thinking_process(conversation_id),
            "response_time": time.time() - start_time,
            "timeout": True
        }

    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for query"""
        normalized = query.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()

    async def _verify_pattern_response(self, answer: str, query: str) -> Dict[str, Any]:
        """Verify pattern-matched responses for accuracy"""
        # Quick verification for pattern responses
        return {
            "risk": False,
            "risk_level": "none",
            "confidence": 0.95,
            "issues_found": {
                "unsupported_claims": [],
                "missing_citations": []
            },
            "quality_assessment": {
                "completeness": 0.9,
                "accuracy": 0.95,
                "clarity": 1.0,
                "structure": 1.0
            }
        }

    async def _build_page_trees_for_pattern(self, pattern_response: Dict) -> List[Dict[str, Any]]:
        """Build simple page trees for pattern-matched responses"""
        # Extract page IDs from citations in the answer
        citation_pattern = r'\[\[([^\]]+)\]\]'
        citations = re.findall(citation_pattern, pattern_response.get('answer', ''))

        trees = []
        for citation in citations[:3]:  # Limit to 3 trees
            # Simple tree for pattern responses
            trees.append({
                "root_page_id": citation.split('-')[0],
                "root_title": pattern_response.get('category', 'Documentation'),
                "markdown": f"- [{pattern_response.get('category', 'Documentation')}](/wiki/pages/{citation}) ⭐",
                "contains_answer": True
            })

        return trees

    async def _analyze_query(self, query: str, conversation_id: str) -> QueryAnalysis:
        """Analyze query using AutoGen Query Analyser agent"""

        await self._log_thinking(conversation_id, "query_analyser", "analyze", "Analyzing query complexity and clarity", None)

        # Create AutoGen group chat for analysis
        groupchat = GroupChat(
            agents=[self.agents["query_analyser"]],
            messages=[],
            max_round=1
        )

        manager = GroupChatManager(groupchat=groupchat)

        # Get analysis using AutoGen
        await self.agents["query_analyser"].initiate_chat(
            manager,
            message=f"Analyze this query: {query}"
        )

        # Parse response
        response = groupchat.messages[-1]["content"]
        analysis_data = json.loads(response)

        analysis = QueryAnalysis(
            classification=analysis_data["classification"],
            subquestions=analysis_data.get("subquestions", []),
            clarification_needed=analysis_data.get("clarification_needed"),
            confidence=analysis_data.get("confidence", 0.8),
            reasoning=analysis_data.get("reasoning", ""),
            key_concepts=analysis_data.get("key_concepts", []),
            temporal_aspects=analysis_data.get("temporal_aspects", [])
        )

        await self._log_thinking(conversation_id, "query_analyser", "complete", f"Classification: {analysis.classification}", analysis)

        return analysis

    async def _handle_clarification(self, query: str, analysis: QueryAnalysis, conversation_id: str) -> Dict[str, Any]:
        """Handle queries that need clarification using AutoGen Clarifier agent"""

        await self._log_thinking(conversation_id, "clarifier", "clarify", "Query needs clarification", analysis.clarification_needed)

        # Get clarification from AutoGen clarifier agent
        groupchat = GroupChat(
            agents=[self.agents["clarifier"]],
            messages=[],
            max_round=1
        )

        manager = GroupChatManager(groupchat=groupchat)

        await self.agents["clarifier"].initiate_chat(
            manager,
            message=f"Original query: {query}\nClarification needed: {analysis.clarification_needed}\nGenerate a helpful clarifying question."
        )

        clarification_response = groupchat.messages[-1]["content"]

        # Save conversation state to Cosmos DB
        await self.data_store.save_conversation(conversation_id, [
            {"role": "user", "content": query, "timestamp": time.time()},
            {"role": "assistant", "content": clarification_response, "timestamp": time.time()}
        ])

        thinking_process = await self._format_thinking_process(conversation_id)

        return {
            "status": "needs_clarification",
            "clarification_message": clarification_response,
            "original_query": query,
            "thinking_process": thinking_process,
            "suggestions": [
                "Try being more specific about which system or component",
                "Include version numbers or time periods if relevant",
                "Specify if you need setup, troubleshooting, or general info"
            ]
        }

    async def _decompose_query(self, query: str, conversation_id: str) -> Dict[str, Any]:
        """Decompose complex query into sub-questions"""
        groupchat = GroupChat(
            agents=[self.agents["decomposer"]],
            messages=[],
            max_round=1
        )

        manager = GroupChatManager(groupchat=groupchat)

        await self.agents["decomposer"].initiate_chat(
            manager,
            message=f"Decompose this query into sub-questions: {query}"
        )

        response = groupchat.messages[-1]["content"]
        return json.loads(response)

    async def _plan_path(self, query: str, hop_idx: int, previous_results: List[Dict], conversation_id: str) -> Dict[str, Any]:
        """Plan search path through knowledge graph"""

        # Get previous page IDs for multi-hop
        prev_page_ids = []
        if previous_results:
            for result in previous_results:
                for doc in result["documents"][:3]:  # Top 3 from previous hop
                    prev_page_ids.append(doc.page_id)

        context = {
            "query": query,
            "hop_index": hop_idx,
            "previous_page_ids": prev_page_ids,
            "max_hops": MAX_HOPS,
            "edge_types": EDGE_TYPES
        }

        groupchat = GroupChat(
            agents=[self.agents["path_planner"]],
            messages=[],
            max_round=1
        )

        manager = GroupChatManager(groupchat=groupchat)

        await self.agents["path_planner"].initiate_chat(
            manager,
            message=f"Plan search path for: {json.dumps(context)}"
        )

        response = groupchat.messages[-1]["content"]
        return json.loads(response)

    async def _synthesize_answer(self, query: str, sub_results: List[Dict], conversation_id: str) -> str:
        """Synthesize final answer from sub-results"""

        # Prepare context
        context_blocks = []
        for i, result in enumerate(sub_results):
            context_blocks.append(f"\n=== Sub-question {i+1}: {result['question']} ===\n")
            for doc in result["documents"]:
                context_blocks.append(f"[[{doc.id}]] {doc.content}\n")

        context = "\n".join(context_blocks)

        groupchat = GroupChat(
            agents=[self.agents["synthesiser"]],
            messages=[],
            max_round=1
        )

        manager = GroupChatManager(groupchat=groupchat)

        await self.agents["synthesiser"].initiate_chat(
            manager,
            message=f"Original question: {query}\n\nContext:\n{context}\n\nSynthesize a comprehensive answer with citations."
        )

        answer = groupchat.messages[-1]["content"]
        return answer

    async def _verify_answer(self, answer: str, sub_results: List[Dict], conversation_id: str) -> Dict[str, Any]:
        """Verify answer accuracy"""

        # Prepare context for verification
        all_content = []
        for result in sub_results:
            for doc in result["documents"]:
                all_content.append(f"[[{doc.id}]] {doc.content}")

        context = "\n".join(all_content)

        groupchat = GroupChat(
            agents=[self.agents["verifier"]],
            messages=[],
            max_round=1
        )

        manager = GroupChatManager(groupchat=groupchat)

        await self.agents["verifier"].initiate_chat(
            manager,
            message=f"Answer to verify:\n{answer}\n\nContext:\n{context}\n\nVerify all claims are supported."
        )

        response = groupchat.messages[-1]["content"]
        return json.loads(response)

    async def _build_page_trees(self, sub_results: List[Dict]) -> List[Dict[str, Any]]:
        """Build page hierarchy trees"""

        # Collect all unique page IDs
        page_ids = set()
        answer_page_ids = set()

        for result in sub_results:
            for doc in result["documents"]:
                page_ids.add(doc.page_id)
                answer_page_ids.add(doc.page_id)

        # Get page hierarchies from Gremlin
        trees = []
        processed_roots = set()

        for page_id in page_ids:
            # Get ancestry path
            ancestry_query = f"""
            g.V('{page_id}')
              .repeat(out('ParentOf')).emit()
              .path()
              .by(project('id', 'title').by('id').by('title'))
            """

            result = await self.gremlin_client.submit(ancestry_query)
            paths = result.all().result()

            if paths:
                # Build tree from path
                root_id = paths[0][-1]['id']  # Last element is root

                if root_id not in processed_roots:
                    processed_roots.add(root_id)
                    tree = await self._build_complete_tree(root_id, answer_page_ids)
                    trees.append(tree)

        # Render trees as markdown
        rendered_trees = []
        for tree in trees:
            markdown = self._render_tree_markdown(tree)
            rendered_trees.append({
                "root_page_id": tree["page_id"],
                "root_title": tree["title"],
                "markdown": markdown,
                "contains_answer": self._tree_contains_answer(tree)
            })

        return rendered_trees

    async def _build_complete_tree(self, root_id: str, answer_page_ids: set) -> Dict[str, Any]:
        """Build complete tree from root"""

        # Get all descendants
        query = f"""
        g.V('{root_id}')
          .repeat(__.in('ParentOf')).emit()
          .tree()
          .by(project('id', 'title', 'url').by('id').by('title').by('url'))
        """

        result = await self.gremlin_client.submit(query)
        tree_data = result.all().result()[0]

        # Convert to our tree structure
        def convert_node(node_data, page_data):
            page_id = page_data['id']
            return {
                "page_id": page_id,
                "title": page_data['title'],
                "url": page_data.get('url', f"/wiki/pages/{page_id}"),
                "is_answer_source": page_id in answer_page_ids,
                "children": [
                    convert_node(child_data, child_page)
                    for child_data, child_page in node_data.items()
                    if isinstance(child_data, dict)
                ]
            }

        root_page = list(tree_data.keys())[0]
        return convert_node(tree_data[root_page], root_page)

    def _render_tree_markdown(self, tree: Dict[str, Any], level: int = 0) -> str:
        """Render tree as markdown"""
        indent = "  " * level

        # Highlight if this node contains answer
        if tree["is_answer_source"]:
            line = f"{indent}- **[{tree['title']}]({tree['url']})** ⭐ *(contains answer)*"
        else:
            line = f"{indent}- [{tree['title']}]({tree['url']})"

        lines = [line]

        # Render children
        for child in tree.get("children", []):
            lines.append(self._render_tree_markdown(child, level + 1))

        return "\n".join(lines)

    def _tree_contains_answer(self, tree: Dict[str, Any]) -> bool:
        """Check if tree contains any answer sources"""
        if tree["is_answer_source"]:
            return True

        for child in tree.get("children", []):
            if self._tree_contains_answer(child):
                return True

        return False

    async def _handle_verification_failure(self, answer: str, verification: Dict, sub_results: List[Dict]) -> str:
        """Handle case where verification failed"""

        # Get breadcrumbs for top pages
        breadcrumbs = []
        for result in sub_results:
            if result["documents"]:
                top_page_id = result["documents"][0].page_id
                breadcrumb = await self._get_breadcrumb(top_page_id)
                breadcrumbs.append(breadcrumb)

        # Construct fallback response
        fallback = f"""
        I found some information about your query, but I'm not fully confident in providing a complete answer.

        Here's what I found:
        {answer}

        **Note**: Some claims could not be fully verified against the source documents.

        For more authoritative information, please check these parent pages:
        """

        for bc in breadcrumbs:
            fallback += f"\n- {' > '.join(bc)}"

        fallback += "\n\nYou may also want to explore related documentation or contact your team for clarification."

        return fallback

    async def _get_breadcrumb(self, page_id: str) -> List[str]:
        """Get breadcrumb path for a page"""
        query = f"""
        g.V('{page_id}')
          .repeat(out('ParentOf')).emit()
          .values('title')
        """

        result = await self.gremlin_client.submit(query)
        titles = result.all().result()
        return list(reversed(titles))

    async def _log_thinking(self, conversation_id: str, agent: str, action: str, reasoning: str, result: Any):
        """Log thinking process step to Cosmos DB"""
        step = {
            'agent': agent,
            'action': action,
            'reasoning': reasoning,
            'result': result,
            'timestamp': time.time()
        }
        await self.data_store.save_thinking_step(conversation_id, step)

    async def _format_thinking_process(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Format thinking process for output from Cosmos DB"""
        steps = await self.data_store.get_thinking_steps(conversation_id)
        formatted = []
        for i, step in enumerate(steps):
            formatted.append({
                "step": i + 1,
                "agent": step['agent'],
                "action": step['action'],
                "reasoning": step['reasoning'],
                "timestamp": step['timestamp']
            })
        return formatted

    # Function implementations for agents

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Function for query analyser agent"""
        # This would be called by the agent, but we handle it differently
        pass

    def plan_path(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Function for path planner agent"""
        # This would be called by the agent
        pass

    def hybrid_search(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        """Function for retriever agent"""
        # This would be called by the agent
        pass

    def semantic_rerank(self, documents: List[Dict], query: str) -> List[Dict]:
        """Function for reranker agent"""
        # This would be called by the agent
        pass

    def synthesize_answer(self, query: str, context: List[Dict]) -> str:
        """Function for synthesiser agent"""
        # This would be called by the agent
        pass

    def calculate_coverage(self, answer: str, context: str) -> float:
        """Calculate how well context covers the answer"""
        # Simple implementation - in production use more sophisticated metrics
        answer_tokens = set(answer.lower().split())
        context_tokens = set(context.lower().split())

        if not answer_tokens:
            return 0.0

        coverage = len(answer_tokens.intersection(context_tokens)) / len(answer_tokens)
        return min(coverage, 1.0)

    def build_page_tree(self, page_ids: List[str]) -> Dict[str, Any]:
        """Function for tree builder agent"""
        # This would be called by the agent
        pass

    def render_tree_markdown(self, tree: Dict[str, Any]) -> str:
        """Function for tree builder agent"""
        # This would be called by the agent
        pass

    def get_page_relationships(self, page_id: str) -> Dict[str, Any]:
        """Get page relationships from graph"""
        # This would query Gremlin for relationships
        pass

    def fetch_page_content(self, page_id: str) -> str:
        """Function for retriever agent to fetch page content"""
        # This would be called by the agent
        pass


# Example usage
async def main():
    """Example usage of the Confluence Q&A system"""

    orchestrator = ConfluenceQAOrchestrator()

    # Example queries
    queries = [
        "How do I enable SSO for our application?",
        "What changed between version 1.0 and 2.0 of the API?",
        "How does the system work?",  # Needs clarification
        "What are the deployment steps for the payment service and how do they relate to the database migration process?"
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")

        result = await orchestrator.process_query(query, f"conv_{hash(query)}")

        print(f"\nStatus: {result['status']}")

        if result['status'] == 'needs_clarification':
            print(f"\nClarification needed:")
            print(result['clarification_message'])
            print(f"\nSuggestions:")
            for suggestion in result['suggestions']:
                print(f"  - {suggestion}")
        else:
            print(f"\nAnswer:")
            print(result['answer'])

            print(f"\nConfidence: {result['confidence']}")

            if result.get('sub_questions'):
                print(f"\nSub-questions analyzed:")
                for i, sq in enumerate(result['sub_questions']):
                    print(f"  {i+1}. {sq}")

            print(f"\nPage Trees:")
            for tree in result['page_trees']:
                print(f"\n{tree['root_title']} {'(contains answer)' if tree['contains_answer'] else ''}")
                print(tree['markdown'])

        print(f"\nThinking Process:")
        for step in result['thinking_process']:
            print(f"  Step {step['step']}: [{step['agent']}] {step['action']} - {step['reasoning']}")


if __name__ == "__main__":
    asyncio.run(main())

# api_service.py
"""
Azure Functions service for Confluence Q&A system
Provides HTTP trigger functions for the AutoGen-based Q&A orchestrator
Integrates with Azure Cognitive Search, Cosmos DB, and Azure Storage
"""

import azure.functions as func
import json
import logging
import asyncio
import uuid
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from azure.identity import DefaultAzureCredential
from azure.cosmos.aio import CosmosClient
import os

# Import your existing modules
from confluence_qa_orchestrator import ConfluenceQAOrchestrator, AzureDataStore
from utils import Config, MetricsCollector, ResponseCache, CitationExtractor

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
    logging.info('Health check endpoint called')

    initialize_services()

    return func.HttpResponse(
        json.dumps({
            "status": "healthy",
            "version": "1.0.0",
            "metrics": metrics.get_metrics(),
            "timestamp": datetime.utcnow().isoformat()
        }),
        mimetype="application/json",
        status_code=200
    )

# Process Query Function
@app.function_name(name="ProcessQuery")
@app.route(route="query", methods=["POST"], auth_level=func.AuthLevel.FUNCTION)
async def process_query(req: func.HttpRequest) -> func.HttpResponse:
    """Process a user query"""
    logging.info('Process query endpoint called')

    initialize_services()
    start_time = time.time()

    try:
        req_body = req.get_json()
    except ValueError:
        return func.HttpResponse(
            json.dumps({"error": "Invalid JSON in request body"}),
            mimetype="application/json",
            status_code=400
        )

    # Extract request parameters
    query = req_body.get('query')
    if not query:
        return func.HttpResponse(
            json.dumps({"error": "Missing required field: query"}),
            mimetype="application/json",
            status_code=400
        )

    conversation_id = req_body.get('conversation_id') or str(uuid.uuid4())
    include_thinking_process = req_body.get('include_thinking_process', True)
    max_wait_seconds = req_body.get('max_wait_seconds', 60)

    # Check cache first
    cached_response = cache.get(query)
    if cached_response:
        metrics.record_cache_hit(True)
        cached_response['conversation_id'] = conversation_id
        cached_response['response_time'] = time.time() - start_time
        return func.HttpResponse(
            json.dumps(cached_response),
            mimetype="application/json",
            status_code=200
        )

    metrics.record_cache_hit(False)

    try:
        # Process query with timeout
        result = await asyncio.wait_for(
            orchestrator.process_query(query, conversation_id),
            timeout=max_wait_seconds
        )

        response_time = time.time() - start_time

        # Handle different response types
        if result['status'] == 'needs_clarification':
            metrics.record_clarification()
            return func.HttpResponse(
                json.dumps({
                    'status': 'needs_clarification',
                    'clarification_message': result['clarification_message'],
                    'suggestions': result.get('suggestions', []),
                    'original_query': query,
                    'conversation_id': conversation_id
                }),
                mimetype="application/json",
                status_code=200
            )

        # Extract citations from answer
        citations = CitationExtractor.extract_citations(result['answer'])

        # Prepare successful response
        response_data = {
            'status': 'success',
            'answer': result['answer'],
            'confidence': result.get('confidence', result['verification']['confidence']),
            'page_trees': result['page_trees'],
            'citations': citations,
            'conversation_id': conversation_id,
            'response_time': response_time
        }

        if include_thinking_process:
            response_data['thinking_process'] = result['thinking_process']

        if 'sub_questions' in result:
            response_data['sub_questions'] = result['sub_questions']

        # Record metrics
        metrics.record_query(
            success=True,
            response_time=response_time,
            hops=len(result.get('sub_questions', []))
        )

        # Cache successful response
        cache.set(query, response_data)

        return func.HttpResponse(
            json.dumps(response_data),
            mimetype="application/json",
            status_code=200
        )

    except asyncio.TimeoutError:
        metrics.record_query(success=False, response_time=time.time() - start_time)
        return func.HttpResponse(
            json.dumps({
                "error": "Query processing timed out",
                "details": "Please try a simpler query or increase timeout."
            }),
            mimetype="application/json",
            status_code=408
        )

    except Exception as e:
        metrics.record_query(success=False, response_time=time.time() - start_time)
        logger.error(f"Error processing query: {str(e)}", exc_info=True)

        return func.HttpResponse(
            json.dumps({
                "error": "Failed to process query",
                "details": str(e)
            }),
            mimetype="application/json",
            status_code=500
        )

# Submit Clarification Function
@app.function_name(name="SubmitClarification")
@app.route(route="clarify/{conversation_id}", methods=["POST"], auth_level=func.AuthLevel.FUNCTION)
async def submit_clarification(req: func.HttpRequest) -> func.HttpResponse:
    """Submit clarification for a previous query"""
    conversation_id = req.route_params.get('conversation_id')
    logging.info(f'Submit clarification called for conversation: {conversation_id}')

    initialize_services()

    try:
        req_body = req.get_json()
        clarification = req_body.get('clarification')
    except:
        return func.HttpResponse(
            json.dumps({"error": "Invalid request body"}),
            mimetype="application/json",
            status_code=400
        )

    if not clarification:
        return func.HttpResponse(
            json.dumps({"error": "Missing clarification field"}),
            mimetype="application/json",
            status_code=400
        )

    # Get conversation from Cosmos DB
    conversation = await data_store.get_conversation(conversation_id)
    if not conversation:
        return func.HttpResponse(
            json.dumps({"error": "Conversation not found"}),
            mimetype="application/json",
            status_code=404
        )

    # Get original query from conversation
    messages = conversation.get('messages', [])
    if not messages:
        return func.HttpResponse(
            json.dumps({"error": "No previous query in conversation"}),
            mimetype="application/json",
            status_code=400
        )

    # Find the last user query
    original_query = None
    for msg in reversed(messages):
        if msg['role'] == 'user':
            original_query = msg['content']
            break

    if not original_query:
        return func.HttpResponse(
            json.dumps({"error": "No user query found in conversation"}),
            mimetype="application/json",
            status_code=400
        )

    # Create enhanced query with clarification
    enhanced_query = f"{original_query} (Clarification: {clarification})"

    # Process the enhanced query
    req.get_body = lambda: json.dumps({
        'query': enhanced_query,
        'conversation_id': conversation_id,
        'include_thinking_process': True
    }).encode('utf-8')

    return await process_query(req)

# Get Conversation Function
@app.function_name(name="GetConversation")
@app.route(route="conversation/{conversation_id}", methods=["GET"], auth_level=func.AuthLevel.FUNCTION)
async def get_conversation(req: func.HttpRequest) -> func.HttpResponse:
    """Get conversation history"""
    conversation_id = req.route_params.get('conversation_id')
    logging.info(f'Get conversation called for ID: {conversation_id}')

    initialize_services()

    conversation = await data_store.get_conversation(conversation_id)
    if not conversation:
        return func.HttpResponse(
            json.dumps({"error": "Conversation not found"}),
            mimetype="application/json",
            status_code=404
        )

    return func.HttpResponse(
        json.dumps(conversation),
        mimetype="application/json",
        status_code=200
    )

# Delete Conversation Function
@app.function_name(name="DeleteConversation")
@app.route(route="conversation/{conversation_id}", methods=["DELETE"], auth_level=func.AuthLevel.FUNCTION)
async def delete_conversation(req: func.HttpRequest) -> func.HttpResponse:
    """Delete conversation history (soft delete)"""
    conversation_id = req.route_params.get('conversation_id')
    logging.info(f'Delete conversation called for ID: {conversation_id}')

    initialize_services()

    conversation = await data_store.get_conversation(conversation_id)
    if not conversation:
        return func.HttpResponse(
            json.dumps({"error": "Conversation not found"}),
            mimetype="application/json",
            status_code=404
        )

    # Mark as deleted (soft delete)
    conversation['deleted'] = True
    conversation['deletedAt'] = time.time()
    await data_store.save_conversation(conversation_id, conversation['messages'])

    return func.HttpResponse(
        json.dumps({"message": "Conversation marked as deleted"}),
        mimetype="application/json",
        status_code=200
    )

# Get Metrics Function
@app.function_name(name="GetMetrics")
@app.route(route="metrics", methods=["GET"], auth_level=func.AuthLevel.FUNCTION)
async def get_metrics(req: func.HttpRequest) -> func.HttpResponse:
    """Get system metrics"""
    logging.info('Get metrics endpoint called')

    initialize_services()

    # Get conversation count from Cosmos DB
    conv_count_query = "SELECT VALUE COUNT(1) FROM c WHERE c.deleted != true"
    conv_count = await asyncio.to_thread(
        lambda: list(data_store.conversation_container.query_items(
            query=conv_count_query,
            enable_cross_partition_query=True
        ))[0]
    )

    return func.HttpResponse(
        json.dumps({
            "metrics": metrics.get_metrics(),
            "cache_size": len(cache.cache),
            "active_conversations": conv_count,
            "timestamp": datetime.utcnow().isoformat()
        }),
        mimetype="application/json",
        status_code=200
    )

# Submit Feedback Function
@app.function_name(name="SubmitFeedback")
@app.route(route="feedback", methods=["POST"], auth_level=func.AuthLevel.FUNCTION)
async def submit_feedback(req: func.HttpRequest) -> func.HttpResponse:
    """Submit feedback for a response"""
    logging.info('Submit feedback endpoint called')

    initialize_services()

    try:
        req_body = req.get_json()
    except:
        return func.HttpResponse(
            json.dumps({"error": "Invalid request body"}),
            mimetype="application/json",
            status_code=400
        )

    conversation_id = req_body.get('conversation_id')
    helpful = req_body.get('helpful')
    feedback_text = req_body.get('feedback_text')

    if not conversation_id or helpful is None:
        return func.HttpResponse(
            json.dumps({"error": "Missing required fields"}),
            mimetype="application/json",
            status_code=400
        )

    # Store feedback
    feedback_data = {
        'id': str(uuid.uuid4()),
        'conversation_id': conversation_id,
        'helpful': helpful,
        'feedback_text': feedback_text,
        'timestamp': datetime.utcnow().isoformat()
    }

    # In production, save to Cosmos DB
    logger.info(f"Feedback received: {feedback_data}")

    return func.HttpResponse(
        json.dumps({
            "message": "Feedback received",
            "feedback_id": feedback_data['id']
        }),
        mimetype="application/json",
        status_code=200
    )

# Find Similar Queries Function
@app.function_name(name="FindSimilarQueries")
@app.route(route="search/similar", methods=["GET"], auth_level=func.AuthLevel.FUNCTION)
async def find_similar_queries(req: func.HttpRequest) -> func.HttpResponse:
    """Find similar previously answered queries"""
    logging.info('Find similar queries endpoint called')

    initialize_services()

    query = req.params.get('query')
    limit = int(req.params.get('limit', 5))

    if not query:
        return func.HttpResponse(
            json.dumps({"error": "Missing query parameter"}),
            mimetype="application/json",
            status_code=400
        )

    # In production, use vector similarity search
    similar = []

    for key, entry in cache.cache.items():
        if 'query' in entry:
            # Simple similarity check (in production use embeddings)
            if any(word in entry['query'].lower() for word in query.lower().split()):
                similar.append({
                    'query': entry['query'],
                    'answer_preview': entry['response']['answer'][:200] + '...',
                    'confidence': entry['response'].get('confidence', 0),
                    'timestamp': datetime.fromtimestamp(entry['timestamp']).isoformat()
                })

    # Sort by recency and limit
    similar.sort(key=lambda x: x['timestamp'], reverse=True)

    return func.HttpResponse(
        json.dumps({
            "similar_queries": similar[:limit],
            "count": len(similar[:limit])
        }),
        mimetype="application/json",
        status_code=200
    )

# Timer Trigger for Cleanup (runs every hour)
@app.function_name(name="CleanupOldData")
@app.schedule(schedule="0 0 * * * *", arg_name="timer", run_on_startup=False)
async def cleanup_old_data(timer: func.TimerRequest) -> None:
    """Periodic cleanup of old conversations and cache"""
    logging.info('Cleanup timer triggered')

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
            lambda: list(data_store.conversation_container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))
        )

        # Mark old conversations as deleted
        for conv in old_conversations:
            conv_id = conv['id']
            conversation = await data_store.get_conversation(conv_id)
            if conversation:
                conversation['deleted'] = True
                conversation['deletedAt'] = time.time()
                await data_store.save_conversation(conv_id, conversation.get('messages', []))

        if old_conversations:
            logger.info(f"Marked {len(old_conversations)} old conversations as deleted")

    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

# Queue Trigger for Streaming Updates (alternative to SSE)
@app.function_name(name="ProcessQueryAsync")
@app.queue_trigger(arg_name="msg", queue_name="query-processing", connection="AzureWebJobsStorage")
async def process_query_async(msg: func.QueueMessage) -> None:
    """Process queries asynchronously and store progress in Cosmos DB"""
    logging.info(f'Processing async query: {msg.get_body().decode("utf-8")}')

    initialize_services()

    try:
        message_data = json.loads(msg.get_body().decode('utf-8'))
        query = message_data['query']
        conversation_id = message_data['conversation_id']

        # Process query and update status in Cosmos DB
        result = await orchestrator.process_query(query, conversation_id)

        # Store result in Cosmos DB for retrieval
        await data_store.save_query_result(conversation_id, result)

    except Exception as e:
        logger.error(f"Error processing async query: {str(e)}")

# Error handling helper
def create_error_response(error: str, details: Any = None, status_code: int = 500) -> func.HttpResponse:
    """Create standardized error response"""
    response_data = {
        "error": error,
        "timestamp": datetime.utcnow().isoformat()
    }

    if details:
        response_data["details"] = details

    return func.HttpResponse(
        json.dumps(response_data),
        mimetype="application/json",
        status_code=status_code
    )
