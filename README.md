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
        SSE stream (/trace/XYZ via FastAPI)
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
- `POST /api/query` - Query the system
- `GET /api/health` - Health check
- `GET /api/metrics` - System metrics

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

#### Unit Tests
Run all unit tests:
```bash
pytest tests/unit -v
```

Run specific unit test:
```bash
pytest tests/unit/test_azure_search.py -v
pytest tests/unit/test_graph_feedback.py -v
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

## License

MIT License
