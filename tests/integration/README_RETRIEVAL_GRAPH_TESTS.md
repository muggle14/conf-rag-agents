# Aspect 2: Retrieval + Graph Integration Tests

## Overview
Comprehensive test suite for validating retrieval and graph integration functionality using **real Azure services** (no mocks).

## Test Files Created

### 1. `test_retrieval_api.py`
Tests the `/ask` API endpoint with real Azure services:
- Response schema validation with proper `mode`, `trace_id`, `confidence`
- Clarification path validation (≤20 words)
- Answer path validation (sources and primary_page_tree)
- Space filtering functionality
- Latency guardrails (P95 < 8s)
- Confidence score validity and correlation with modes
- Error handling for invalid queries

### 2. `test_rerank_toggle.py`
Tests reranking behavior with Azure OpenAI:
- Rerank flag toggle functionality
- Order changes only when `rerank=true`
- Performance overhead measurement
- Quality improvements with reranking
- Consistency of reranked results
- Space filter compatibility with reranking

### 3. `test_direct_services.py`
Direct validation of Azure services:
- **Azure Search Tests:**
  - Field presence validation
  - Space filtering
  - Scoring and ranking
  - Vector search capabilities (if configured)
- **Cosmos Gremlin Tests:**
  - Connection validation
  - Vertex structure verification
  - Parent-child relationship traversal
- **Overlap Analysis:**
  - Search-graph neighbor overlap calculation
  - Graph enrichment value assessment

### 4. `test_graph_search_overlap.py`
Graph-search integration tests:
- Neighbor overlap with search results
- Primary page tree generation from graph
- Graph enrichment value for relevance
- Graph traversal performance (P95 < 250ms for neighbors)
- Combined search + graph enrichment timing

### 5. `test_trace_sse.py` (Enhanced)
SSE trace streaming validation:
- Event sequence validation (start → search → confidence → synthesis)
- Payload field validation for each step type
- Timing constraints (< 5s for full trace)
- Confidence value range validation [0, 1]
- Graph event payload validation

## Environment Configuration

Set these environment variables before running:

```bash
# API endpoint
export API_BASE="http://localhost:8000"       # or your APIM/App Service URL

# Test queries
export TEST_SPECIFIC_QUERY="Graph Enrichment Skill"
export TEST_AMBIGUOUS_QUERY="architecture"
export TEST_SPACE="your-space-key"            # optional; unset if single-space

# Azure Search (for direct service tests)
export AZURE_SEARCH_ENDPOINT="https://<your-search>.search.windows.net"
export AZURE_SEARCH_INDEX="confluence-pages"
export AZURE_SEARCH_KEY="<admin-or-query-key>"

# Cosmos DB Gremlin (for direct service tests)
export COSMOS_GRAPH_DB_ENDPOINT="<cosmos-account>.gremlin.cosmos.azure.com"
export COSMOS_GRAPH_DB_DATABASE="confluence-graph"
export COSMOS_GRAPH_DB_COLLECTION="page-relationships"
export COSMOS_GRAPH_DB_KEY="<cosmos-primary-key>"

# Azure OpenAI (for reranking)
export AZURE_OPENAI_ENDPOINT="https://<your-openai>.openai.azure.com"
export AZURE_OPENAI_API_KEY="<openai-key>"
```

## Running the Tests

### Run all integration tests:
```bash
pytest tests/integration/ -v -s -m integration
```

### Run specific test suites:
```bash
# API retrieval tests
pytest tests/integration/test_retrieval_api.py -v -s

# Reranking tests
pytest tests/integration/test_rerank_toggle.py -v -s

# Direct service validation
pytest tests/integration/test_direct_services.py -v -s

# Graph-search overlap
pytest tests/integration/test_graph_search_overlap.py -v -s

# SSE trace tests
pytest tests/functional/test_trace_sse.py -v -s
```

### Run with specific markers:
```bash
# Only tests that require both search and graph
pytest tests/integration/ -v -s -k "search and graph"
```

## Pass/Fail Criteria

### ✅ Pass Criteria:
- `/ask` (specific): Returns answer/proceed in < 8s with 1-3 sources and tree
- `/ask` (ambiguous): Returns clarify with ≤20 words OR proceed/answer with confidence ≥0.55
- SSE trace: Complete event sequence in < 5s
- Search+Graph overlap: ≥5% for known queries
- Rerank: Order differs only when `rerank=true`
- P95 latency: < 8s for API calls, < 250ms for graph neighbors

### ❌ Fail Criteria:
- Missing required fields in API responses
- Confidence values outside [0, 1] range
- P95 latency exceeding thresholds
- No overlap between search and graph for well-connected documents
- Reranking changes order when `rerank=false`

## Key Features

1. **No Mocks**: All tests use real Azure services
2. **Environment-Driven**: Configurable via environment variables
3. **Graceful Skipping**: Tests skip when credentials not available
4. **Performance Metrics**: Reports actual latencies and percentiles
5. **Comprehensive Validation**: Schema, payload, and timing validation

## Debugging Tips

1. **Enable verbose output**: Use `-v -s` flags with pytest
2. **Check credentials**: Ensure all environment variables are set
3. **Network issues**: Tests may fail due to network latency; adjust timeouts if needed
4. **Service availability**: Ensure Azure services are running and accessible
5. **Index content**: Some tests require specific content in the search index

## CI/CD Integration

Add to your CI pipeline:

```yaml
- name: Run Retrieval + Graph Integration Tests
  env:
    API_BASE: ${{ secrets.API_BASE }}
    AZURE_SEARCH_ENDPOINT: ${{ secrets.AZURE_SEARCH_ENDPOINT }}
    AZURE_SEARCH_KEY: ${{ secrets.AZURE_SEARCH_KEY }}
    # ... other env vars
  run: |
    pytest tests/integration/ -v --tb=short \
      --junit-xml=test-results/retrieval-graph.xml \
      -m integration
```

## Notes

- Tests are designed to work with the existing Confluence Q&A system
- Graph tool implementation (`graph_tool.py`) is used for graph operations
- All tests validate real data flow through the system
- Performance thresholds may need adjustment based on infrastructure
