# Confluence Q&A Agents

A sophisticated Q&A system for Confluence using AutoGen and Azure Services.

## Features

- **AutoGen Integration**: Multi-agent conversations for complex queries
- **Azure AI Search**: Semantic search across Confluence documents
- **Cosmos DB**: Graph database for knowledge relationships
- **Azure Functions**: Serverless API endpoints
- **OpenAI Integration**: Advanced language model capabilities

## OpenTelemetry Integration

The system uses AutoGen-compatible OpenTelemetry for distributed tracing and observability.

### Core Infrastructure
- ✅ Installed OpenTelemetry SDK with Azure Monitor exporter
- ✅ Created `tracing/otel_config.py` - Configures AutoGen-compatible telemetry
- ✅ Created `tracing/autogen_tracer.py` - Wrapper maintaining backward compatibility

### SSE Streaming
- ✅ Created `api/trace_stream_otel.py` - Converts spans to SSE events
- ✅ Real-time UI streaming capability maintained
- ✅ Backward compatible with existing trace_id pattern

### Usage

```python
# Import the AutoGen-compatible tracer
from tracing import autogen_tracer as tracer

# Create spans for agent operations
with tracer.trace_agent("my_agent", operation="search") as (span, trace_id):
    # Your agent code here
    span.set_attribute("query", "confluence pricing")
    span.add_event("processing", {"step": "embedding"})
    # Results automatically traced

# Create spans for tool executions
with tracer.trace_tool("azure_search", query="deployment guide") as (span, trace_id):
    # Tool execution code
    results = search_tool.search(query)
    span.set_attribute("result_count", len(results))

# Backward compatible logging (creates spans internally)
tracer.log("search_start", trace_id, query="test", mode="vector")
```


                 ├─ each agent calls logger.log(..., trace_id=XYZ)
                 │
            tracing/logger.py
                 │
             Queue (buffer)
                 │
        SSE stream (/api/trace/{trace_id} via FastAPI)
                 │
               Browser UI  ← live, ordered, per-run events
                 │
        (optional) Persist to Blob/DB/MLflow for offline eval

### Configuration

Set environment variables for telemetry backends:

```bash
# Azure Monitor (Application Insights)
export APPLICATIONINSIGHTS_CONNECTION_STRING="InstrumentationKey=...;..."

# OTLP Endpoint (for Jaeger/Zipkin)
export OTLP_ENDPOINT="localhost:4317"

# Disable telemetry if needed
export AUTOGEN_DISABLE_RUNTIME_TRACING="true"
```

### Features
- **Distributed Tracing**: Automatic context propagation across agents
- **OpenAI Instrumentation**: API calls automatically traced
- **Azure Integration**: Direct export to Application Insights
- **Multi-Backend Support**: OTLP for Jaeger/Zipkin compatibility
- **Real-time Streaming**: SSE endpoints for UI updates at `/api/trace/{trace_id}`

## Installation

```bash
# Install in development mode
pip install -e .[dev]

# Install with Azure dependencies
pip install -e .[azure]
```

## Quick Start

```python
from agents.router import agent

# Ask a question
answer = await agent.ask("What is the deployment process?")
print(answer)
```

## API Endpoints

- `POST /api/ask` - Ask questions
- `GET /api/health` - Health check
- `GET /api/trace/{trace_id}` - SSE trace stream (OTEL)

Notes:
- Canonical base is `/api`. Backward-compatible aliases at root: `/ask`, `/health`, `/trace/{trace_id}`.

## Testing

### Prerequisites

Before running tests, ensure you have:
1. Set up environment variables in `.env` file
2. Installed test dependencies: `pip install pytest pytest-asyncio`
3. Seeded test data (see below)

### Seeding Test Data

Seed the test environments with sample data:

```bash
# Seed Azure AI Search with sample Confluence pages
python -m scripts.seed_search_index --dir .sample_pages

# Seed Cosmos DB Graph with test vertices and edges
python -m scripts.seed_graph

# Seed Azure Blob Storage with test documents
python -m scripts.seed_blob
```

### Running Tests

#### Prerequisites
Set environment variables for integration tests:
```bash
export API_BASE="http://localhost:8000/api"
export TEST_SPECIFIC_QUERY="Graph Enrichment Skill"
export TEST_AMBIGUOUS_QUERY="architecture"
export AZURE_SEARCH_ENDPOINT="https://your-search.search.windows.net"
export AZURE_SEARCH_KEY="your-key"
export COSMOS_GRAPH_DB_ENDPOINT="your-cosmos.gremlin.cosmos.azure.com"
export COSMOS_GRAPH_DB_KEY="your-cosmos-key"
```

### Environment Variable Names (Unified + Legacy)

The app accepts unified variable names with legacy fallbacks for compatibility:

- Azure Search:
  - Unified: `AZURE_SEARCH_ENDPOINT`, `AZURE_SEARCH_KEY`, `AZURE_SEARCH_INDEX_NAME`
  - Legacy: `SEARCH_ENDPOINT`, `AOAI_SEARCH_KEY`, `SEARCH_INDEX`
- Azure OpenAI:
  - Unified: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AOAI_EMBED_DEPLOY`
  - Legacy: `AOAI_ENDPOINT`, `AOAI_KEY`, `AOAI_EMBED_DEPLOY`
- Cosmos Gremlin (Graph):
  - Unified: `COSMOS_GRAPH_DB_ENDPOINT` (host only), `COSMOS_GRAPH_DB_KEY`, `COSMOS_GRAPH_DB_DATABASE`, `COSMOS_GRAPH_DB_COLLECTION`
  - Legacy: `COSMOS_DB_ENDPOINT` (any format), `COSMOS_DB_KEY`, `COSMOS_DB_DATABASE`, `COSMOS_DB_CONTAINER`
- Cosmos SQL (Sessions):
  - Unified: `COSMOS_SQL_ENDPOINT`, `COSMOS_SQL_KEY`, `COSMOS_SQL_DATABASE`, `COSMOS_SQL_CONTAINER`
  - Legacy: `COSMOS_URL`, `COSMOS_KEY`, `COSMOS_SESSION_DATABASE`, `COSMOS_SESSION_CONTAINER`
- Storage:
  - Unified: `STORAGE_ACCOUNT`, `STORAGE_KEY`, `RAW_CONTAINER`, `PROC_CONTAINER`
  - Legacy: `AZURE_STORAGE_CONNECTION_STRING`, `AZURE_STORAGE_CONTAINER_NAME`

#### Unit Tests
Run all unit tests:
```bash
pytest tests/unit -v
```

Run specific unit test:
```bash
pytest tests/unit/test_azure_search.py -v
pytest tests/unit/test_graph_feedback.py -v
pytest tests/unit/test_graph_tool.py::TestGraphNormalize -v -s
```

#### Smoke Tests
Smoke tests verify basic functionality of each component after deployment or configuration changes.

Run all smoke tests:
```bash
pytest tests/smoke -v -m smoke
```

Run individual smoke tests:
```bash
# Basic functionality tests
pytest tests/smoke/test_smoke_basic.py -v

# Azure Search smoke tests
pytest tests/smoke/test_smoke_azure_search.py -v

# Cosmos DB Graph smoke tests
pytest tests/smoke/test_smoke_graph.py -v

# Blob Storage smoke tests
pytest tests/smoke/test_smoke_blob.py -v
```

**What Each Smoke Test Covers:**

- **Basic Tests** (`test_smoke_basic.py`): Module imports, basic connections
- **Azure Search** (`test_smoke_azure_search.py`): Connection, vector search, hybrid search, filters
- **Cosmos DB Graph** (`test_smoke_graph.py`): Connection, graph traversal, multi-hop traversal, edge creation
- **Blob Storage** (`test_smoke_blob.py`): Connection, container verification, blob listing/reading, metadata

#### Functional Tests
Functional tests validate end-to-end API behavior with real services:
```bash
# Run all functional tests
pytest tests/functional -v

# Run specific functional tests
pytest tests/functional/test_api_ask.py::test_specific_query_answers_and_has_tree -v -s
pytest tests/functional/test_space_filter.py::test_space_filter_limits_sources -v -s
pytest tests/functional/test_trace_sse.py::test_trace_has_expected_steps_and_payload_fields -v -s
```

- **API /ask endpoint** (`test_api_ask.py`): Response schema validation, clarification word count, answer sources and tree validation
- **Space filtering** (`test_space_filter.py`): Space parameter limits sources, empty space handling, filter comparison tests
- **SSE trace streaming** (`test_trace_sse.py`): Event sequence validation, payload field verification, timing constraints (<5s), confidence range validation

#### Integration Tests
Integration tests validate service interactions and performance:
```bash
# Run all integration tests
pytest tests/integration -v -m integration

# Run specific integration test files
pytest tests/integration/test_retrieval_api.py::TestRetrievalAPI::test_specific_query_answers_and_has_tree -v -s
pytest tests/integration/test_rerank_toggle.py::TestRerankToggle::test_rerank_changes_order_when_enabled -v -s
pytest tests/integration/test_direct_services.py::TestDirectAzureSearch::test_search_topk_has_fields -v -s
pytest tests/integration/test_graph_search_overlap.py::TestGraphSearchOverlap::test_neighbor_overlap_with_search_results -v -s

# Run performance tests only
pytest tests/integration -k "performance" -v -s

# Run with specific environment
API_BASE=http://localhost:8000 pytest tests/integration/test_retrieval_api.py -v -s
```

- **Retrieval API** (`test_retrieval_api.py`): /ask endpoint validation, latency guardrails (P95 <8s), primary_page_tree structure, confidence score correlation
- **Rerank toggle** (`test_rerank_toggle.py`): Rerank flag behavior, order changes only when enabled, performance overhead measurement, quality improvements
- **Direct services** (`test_direct_services.py`): Azure Search field validation, Cosmos Gremlin traversal, search-graph overlap calculation (≥5%)
- **Graph-search overlap** (`test_graph_search_overlap.py`): Neighbor overlap analysis, tree generation performance (<250ms P95), graph enrichment value assessment
- **Graph tool** (`test_graph_tool_integration.py`): GraphTool neighbors/tree methods, performance metrics, concurrent request handling
- **Search reranker** (`test_search_reranker_integration.py`): GPT-4o reranking, order changes validation, space filter compatibility
- **Orchestrator tests** (`test_orchestrator_*.py`): Agent decomposition, clarification flow, dependency injection patterns

#### All Tests
Run the complete test suite:
```bash
pytest
```

Run with coverage report:
```bash
pytest --cov=src --cov-report=html
```

### Test Configuration

Tests are configured in `pyproject.toml` with the following markers:
- `unit` - Unit tests (fast, isolated)
- `smoke` - Smoke tests (basic functionality checks)
- `integration` - Integration tests (requires services)
- `slow` - Slow running tests

Skip specific test types:
```bash
pytest -m "not integration"  # Skip integration tests
pytest -m "not slow"         # Skip slow tests
```

### Troubleshooting Tests

If tests fail:

1. **Check environment variables** are set correctly in `.env`
2. **Verify Azure resources** are accessible and properly configured
3. **Ensure seeding scripts** have been run successfully
4. **Check Azure service firewall/network settings** allow connections
5. **Review error messages** - most common issues are:
   - Missing environment variables
   - Incorrect Azure service endpoints
   - Authentication/authorization failures
   - Network connectivity issues

## Development

### Code Quality

Format code with Black and Ruff:
```bash
# Format code
black src tests
ruff check --fix src tests

# Check code style
black --check src tests
ruff check src tests

# Type checking
mypy src
```

### Pre-commit Hooks

Install pre-commit hooks:
```bash
pre-commit install
```

Run hooks manually:
```bash
pre-commit run --all-files
```

## CI (Real Data)

GitHub Actions workflow `.github/workflows/real-data-tests.yml` runs smoke, integration, and functional tests against a local API instance using real Azure services.

Configure repository secrets for:
- `AZURE_SEARCH_ENDPOINT`, `AZURE_SEARCH_KEY`, `AZURE_SEARCH_INDEX_NAME`
- `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AOAI_EMBED_DEPLOY`
- Optional: `COSMOS_GRAPH_DB_*`, `COSMOS_SQL_*`, `STORAGE_ACCOUNT`, `STORAGE_KEY`, `RAW_CONTAINER`, `PROC_CONTAINER`

The workflow starts `uvicorn api.app:app` and uses `API_BASE=http://localhost:8000/api`.

## License

MIT License
