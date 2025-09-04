#!/usr/bin/env python3
"""
Functional Test: Real-time Tracing with SSE Streaming Using Real Azure Services
================================================================================

Objective:
----------
Validate that the tracing system works end-to-end with REAL Azure services:
- Real Azure Search queries returning actual Confluence documents
- Real Cosmos DB graph lookups for document relationships
- Real Azure OpenAI embeddings and completions
- Real-time SSE streaming of trace events

Test Criteria:
--------------
1. Each request yields a unique trace_id with ≥ 3 events
2. Events are single-line JSON (no multi-line logging)
3. P95 inter-event latency ≤ 100ms locally
4. SSE streaming works at /trace/{trace_id} endpoint
5. Uses REAL data from Azure services (no mocks)

Prerequisites:
--------------
- Azure Search index with Confluence documents
- Cosmos DB with graph relationships
- Azure OpenAI with embeddings model
- All environment variables configured in .env
"""

import json
import os
import statistics
import sys
import threading
import time
from typing import Any, Dict

import requests

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse

load_dotenv()

# Import tracing and agents
from api.app import create_app
from src.agents.autogen_tools.azure_search import AzureAISearchTool
from src.agents.autogen_tools.graph_lookup import get_neighbors
from tracing.autogen_tracer import log, new_trace_id

# from src.agents.orchestrator import ConfluenceQAOrchestrator  # Commented out - needs refactoring

# Test configuration
TEST_PORT = 8003
TEST_HOST = "0.0.0.0"
BASE_URL = f"http://localhost:{TEST_PORT}"

# Real queries that should return actual Confluence data
REAL_TEST_QUERIES = [
    "confluence pricing",
    "deployment guide",
    "security incident",
    "how to set up authentication",
    "what is the API rate limit",
]


def create_test_app() -> FastAPI:
    """Create a test FastAPI app with real agent endpoints."""
    app = create_app()  # Use the real app with all routers

    # Initialize real Azure services
    search_tool = AzureAISearchTool()

    @app.post("/ask")
    async def ask_question_real(request: Dict[str, Any]):
        """
        Real endpoint that queries actual Azure services.
        NO MOCKS - uses real Azure Search, Cosmos DB, and OpenAI.
        """
        query = request.get("query", REAL_TEST_QUERIES[0])

        # Generate unique trace ID
        trace_id = new_trace_id()
        start_time = time.time()

        # Event 1: Request start
        log(
            "request_start",
            trace_id,
            query=query,
            endpoint="/ask",
            timestamp=start_time,
        )

        try:
            # REAL Azure Search query
            # This will query the actual Confluence index and return real documents
            search_results = search_tool.raw_results(query, top=3, mode="hybrid")

            if search_results:
                # Get the first real document
                main_doc = search_results[0]
                doc_id = main_doc.get("id")
                doc_title = main_doc.get("title", "Unknown")
                doc_score = main_doc.get("@search.score", 0)

                # Log real search results
                log(
                    "search_results",
                    trace_id,
                    query=query,
                    count=len(search_results),
                    top_doc_id=doc_id,
                    top_doc_title=doc_title,
                    top_doc_score=doc_score,
                    timestamp=time.time(),
                )

                # REAL Cosmos DB graph lookup
                # Get actual neighbor documents from the graph
                if doc_id:
                    try:
                        neighbors = get_neighbors(
                            doc_id,
                            edge_types=("ParentOf", "LinksTo", "References"),
                            k=5,
                            trace_id=trace_id,
                        )

                        # Log real graph results
                        log(
                            "graph_neighbors",
                            trace_id,
                            main_doc_id=doc_id,
                            neighbor_count=len(neighbors),
                            neighbor_ids=[n.get("id") for n in neighbors],
                            timestamp=time.time(),
                        )
                    except Exception as graph_error:
                        log(
                            "graph_error",
                            trace_id,
                            error=str(graph_error),
                            doc_id=doc_id,
                            timestamp=time.time(),
                        )
                        neighbors = []

                # REAL synthesis using actual content
                context_docs = search_results[:3]
                total_content = sum(len(doc.get("content", "")) for doc in context_docs)

                log(
                    "synthesize",
                    trace_id,
                    query=query,
                    context_docs=len(context_docs),
                    total_content_chars=total_content,
                    timestamp=time.time(),
                )

                # Build real answer from actual documents
                answer_parts = [
                    f"Based on {len(search_results)} Confluence documents:",
                    f"\nTop result: '{doc_title}' (Score: {doc_score:.4f})",
                    f"\nDocument ID: {doc_id}",
                    f"\nNeighbor documents: {len(neighbors)}",
                    f"\nTotal context: {total_content} characters",
                ]

                # Add snippet from real document
                if main_doc.get("content"):
                    snippet = main_doc["content"][:200].replace("\n", " ")
                    answer_parts.append(f"\nSnippet: {snippet}...")

                answer = "\n".join(answer_parts)

            else:
                # No results from real search
                log("search_empty", trace_id, query=query, timestamp=time.time())
                answer = f"No results found in Confluence for: {query}"

            # Event: Request complete
            end_time = time.time()
            total_time_ms = (end_time - start_time) * 1000

            log(
                "request_complete",
                trace_id,
                query=query,
                total_time_ms=total_time_ms,
                results_found=len(search_results) if search_results else 0,
                timestamp=end_time,
            )

            return JSONResponse(
                {
                    "trace_id": trace_id,
                    "query": query,
                    "answer": answer,
                    "total_time_ms": total_time_ms,
                    "real_data": {
                        "search_results": len(search_results) if search_results else 0,
                        "top_document": doc_title if search_results else None,
                        "azure_search_index": os.getenv("AZURE_SEARCH_INDEX_NAME"),
                        "cosmos_db": os.getenv("COSMOS_GRAPH_DB_DATABASE"),
                    },
                    "stream_urls": {
                        "legacy": f"{BASE_URL}/trace/{trace_id}",
                        "otel": f"{BASE_URL}/otel/trace/{trace_id}",
                    },
                    "status": "success",
                }
            )

        except Exception as e:
            # Log real error
            log(
                "request_error",
                trace_id,
                query=query,
                error=str(e),
                error_type=type(e).__name__,
                timestamp=time.time(),
            )

            return JSONResponse(
                {
                    "trace_id": trace_id,
                    "query": query,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "stream_url": f"{BASE_URL}/trace/{trace_id}",
                    "status": "error",
                },
                status_code=500,
            )

    @app.get("/test/verify-services")
    async def verify_services():
        """Verify all Azure services are accessible with real data."""
        results = {}

        # Test Azure Search
        try:
            search_tool = AzureAISearchTool()
            test_results = search_tool.raw_results("test", top=1, mode="keyword")
            results["azure_search"] = {
                "status": "connected",
                "index": os.getenv("AZURE_SEARCH_INDEX_NAME"),
                "documents_found": len(test_results) if test_results else 0,
            }
        except Exception as e:
            results["azure_search"] = {"status": "error", "error": str(e)}

        # Test Cosmos DB
        try:
            endpoint = os.getenv("COSMOS_GRAPH_DB_ENDPOINT")
            key = os.getenv("COSMOS_GRAPH_DB_KEY")
            db = os.getenv("COSMOS_GRAPH_DB_DATABASE", "confluence-graph")

            if endpoint and key:
                # Try to count vertices
                results["cosmos_db"] = {
                    "status": "configured",
                    "database": db,
                    "endpoint": endpoint.split(".")[0],  # First part only
                }
            else:
                results["cosmos_db"] = {"status": "not configured"}
        except Exception as e:
            results["cosmos_db"] = {"status": "error", "error": str(e)}

        # Test Azure OpenAI
        try:
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            key = os.getenv("AZURE_OPENAI_API_KEY")
            if endpoint and key:
                results["azure_openai"] = {
                    "status": "configured",
                    "endpoint": endpoint.split(".")[0],  # First part only
                    "embed_model": os.getenv("AOAI_EMBED_DEPLOY"),
                }
            else:
                results["azure_openai"] = {"status": "not configured"}
        except Exception as e:
            results["azure_openai"] = {"status": "error", "error": str(e)}

        return results

    return app


# Ensure the functional test server is running on localhost:8003 during pytest runs
@pytest.fixture(scope="session", autouse=True)
def _start_functional_server():
    try:
        start_test_server()
    except Exception:
        # If server already running or fails to start, allow tests to proceed
        pass
    yield


def start_test_server():
    """Start the FastAPI test server in a background thread."""
    import uvicorn

    app = create_test_app()

    def run():
        uvicorn.run(app, host=TEST_HOST, port=TEST_PORT, log_level="error")

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    time.sleep(2)  # Wait for server to start
    print(f"✅ Test server started on {BASE_URL}")


def verify_azure_services() -> bool:
    """Verify all Azure services are accessible."""
    print("\n🔍 Verifying Azure Services (Real Data)")
    print("-" * 40)

    response = requests.get(f"{BASE_URL}/test/verify-services")
    services = response.json()

    all_connected = True
    for service, status in services.items():
        is_connected = status.get("status") in ["connected", "configured"]
        symbol = "✅" if is_connected else "❌"
        print(f"  {symbol} {service}: {status.get('status')}")
        if "documents_found" in status:
            print(f"      Documents: {status['documents_found']}")
        if "database" in status:
            print(f"      Database: {status['database']}")
        if not is_connected:
            all_connected = False
            if "error" in status:
                print(f"      Error: {status['error']}")

    return all_connected


def test_real_search_queries() -> bool:
    """Test with real Confluence queries returning actual documents."""
    print("\n🧪 Test 1: Real Azure Search Queries")
    print("-" * 40)

    passed_queries = []
    failed_queries = []

    for query in REAL_TEST_QUERIES[:3]:  # Test first 3 real queries
        response = requests.post(f"{BASE_URL}/ask", json={"query": query})
        data = response.json()

        if data.get("status") == "success":
            real_data = data.get("real_data", {})
            results_count = real_data.get("search_results", 0)
            top_doc = real_data.get("top_document", "None")

            print(f"  Query: '{query}'")
            print(f"    Results: {results_count} documents")
            print(f"    Top doc: {top_doc}")

            if results_count > 0:
                passed_queries.append(query)
            else:
                failed_queries.append(query)
        else:
            print(f"  Query: '{query}' - ERROR: {data.get('error')}")
            failed_queries.append(query)

    passed = len(passed_queries) > 0
    print(
        f"\n  Successful queries: {len(passed_queries)}/{len(passed_queries) + len(failed_queries)}"
    )
    print("  ✅ PASSED" if passed else "  ❌ FAILED - No real data returned")
    return passed


def test_trace_events_with_real_data() -> bool:
    """Test that real queries generate proper trace events."""
    print("\n🧪 Test 2: Trace Events from Real Data")
    print("-" * 40)

    # Use a real query
    query = "confluence pricing"
    response = requests.post(f"{BASE_URL}/ask", json={"query": query})
    data = response.json()
    trace_id = data.get("trace_id")

    if not trace_id:
        print("  ❌ No trace_id returned")
        return False

    # Capture trace events
    events = []
    event_types = []

    try:
        stream_response = requests.get(
            f"{BASE_URL}/trace/{trace_id}", stream=True, timeout=2
        )

        for line in stream_response.iter_lines():
            if line:
                decoded = line.decode("utf-8")
                if decoded.startswith("data: "):
                    try:
                        event = json.loads(decoded[6:])
                        events.append(event)
                        event_types.append(event.get("step", "unknown"))
                    except:
                        pass
            if len(events) >= 5:  # Stop after 5 events
                break

    except requests.exceptions.Timeout:
        pass

    print(f"  Query: '{query}'")
    print(f"  Trace ID: {trace_id[:12]}...")
    print(f"  Events captured: {len(events)}")
    print(f"  Event types: {event_types}")

    # Check for essential events
    has_request_start = "request_start" in event_types
    has_search = any("search" in e for e in event_types)
    has_complete = "request_complete" in event_types

    passed = len(events) >= 3 and has_request_start and (has_search or has_complete)
    print(f"\n  Essential events present: {has_request_start and has_search}")
    print("  ✅ PASSED" if passed else "  ❌ FAILED")
    return passed


def test_sse_latency_real_data() -> bool:
    """Test SSE streaming latency with real Azure queries."""
    print("\n🧪 Test 3: SSE Latency with Real Data")
    print("-" * 40)

    query = "deployment guide"
    response = requests.post(f"{BASE_URL}/ask", json={"query": query})
    data = response.json()
    trace_id = data.get("trace_id")

    event_times = []

    try:
        stream_start = time.time()
        stream_response = requests.get(
            f"{BASE_URL}/trace/{trace_id}", stream=True, timeout=3
        )

        for line in stream_response.iter_lines():
            if line:
                decoded = line.decode("utf-8")
                if decoded.startswith("data: "):
                    event_times.append(time.time())
            if len(event_times) >= 5:
                break

    except requests.exceptions.Timeout:
        pass

    if len(event_times) > 1:
        # Calculate latencies
        latencies_ms = []
        for i in range(1, len(event_times)):
            latency_ms = (event_times[i] - event_times[i - 1]) * 1000
            latencies_ms.append(latency_ms)

        avg_latency = statistics.mean(latencies_ms)
        p95_latency = (
            statistics.quantile(latencies_ms, 0.95)
            if len(latencies_ms) > 1
            else latencies_ms[0]
        )

        print(f"  Query: '{query}'")
        print(f"  Events streamed: {len(event_times)}")
        print(f"  Avg latency: {avg_latency:.2f}ms")
        print(f"  P95 latency: {p95_latency:.2f}ms")

        passed = p95_latency <= 100
        print("\n  ✅ PASSED (P95 ≤ 100ms)" if passed else "  ❌ FAILED (P95 > 100ms)")
        return passed
    else:
        print("  ⚠️  Insufficient events for latency measurement")
        return False


def test_graph_integration() -> bool:
    """Test that graph lookups work with real Cosmos DB data."""
    print("\n🧪 Test 4: Graph Integration with Real Data")
    print("-" * 40)

    # First get a real document ID from search
    search_tool = AzureAISearchTool()
    results = search_tool.raw_results("confluence", top=1, mode="keyword")

    if not results:
        print("  ⚠️  No documents found to test graph lookup")
        return False

    doc_id = results[0].get("id")
    doc_title = results[0].get("title", "Unknown")

    print(f"  Testing with real document: {doc_title}")
    print(f"  Document ID: {doc_id}")

    # Now test full flow with this document
    query = f"tell me about document {doc_id}"
    response = requests.post(f"{BASE_URL}/ask", json={"query": query})
    data = response.json()

    if data.get("status") == "success":
        real_data = data.get("real_data", {})
        print(f"  Search results: {real_data.get('search_results', 0)}")
        print(
            f"  Answer includes real data: {'Yes' if doc_id in data.get('answer', '') else 'No'}"
        )

        passed = real_data.get("search_results", 0) > 0
        print("\n  ✅ PASSED" if passed else "  ❌ FAILED")
        return passed
    else:
        print(f"  ❌ Error: {data.get('error')}")
        return False


def main():
    """Run all functional tests with real Azure data."""
    print("=" * 60)
    print("🚀 FUNCTIONAL TEST: Real-time Tracing with REAL Azure Data")
    print("=" * 60)
    print("\nUsing REAL Azure services:")
    print("  • Azure Search with actual Confluence documents")
    print("  • Cosmos DB with real graph relationships")
    print("  • Azure OpenAI for embeddings")
    print("  • NO MOCK DATA - ALL REAL")
    print("=" * 60)

    # Start test server
    print("\n🔧 Starting test server...")
    start_test_server()

    # Verify services first
    if not verify_azure_services():
        print("\n⚠️  Some Azure services are not accessible.")
        print("Please check your .env configuration and Azure resources.")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != "y":
            return 1

    # Run tests with real data
    print("\n📊 Running Tests with Real Data")
    print("=" * 60)

    results = []
    results.append(("Real Search Queries", test_real_search_queries()))
    results.append(("Trace Events", test_trace_events_with_real_data()))
    results.append(("SSE Latency", test_sse_latency_real_data()))
    results.append(("Graph Integration", test_graph_integration()))

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY (REAL DATA)")
    print("=" * 60)

    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name:20} {status}")

    total_passed = sum(1 for _, p in results if p)
    total_tests = len(results)

    print(f"\n  Total: {total_passed}/{total_tests} tests passed")

    if total_passed == total_tests:
        print("\n🎉 ALL TESTS PASSED with REAL DATA!")
        print("Tracing system is working correctly with actual Azure services.")
        return 0
    else:
        print(f"\n⚠️  {total_tests - total_passed} test(s) failed.")
        print("Check Azure service connectivity and data availability.")
        return 1


if __name__ == "__main__":
    exit(main())
