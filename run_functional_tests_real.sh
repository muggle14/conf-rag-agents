#!/bin/bash

# Run functional tests with real Azure services
# This script ensures all required environment variables are set
# and runs the tests against actual Azure resources

echo "========================================================================"
echo "🧪 CONFLUENCE Q&A API - FUNCTIONAL TESTS WITH REAL AZURE SERVICES"
echo "========================================================================"

# Check required environment variables
REQUIRED_VARS=(
    "AZURE_SEARCH_ENDPOINT"
    "AZURE_SEARCH_KEY"
    "AZURE_SEARCH_INDEX_NAME"
    "AZURE_OPENAI_ENDPOINT"
    "AZURE_OPENAI_API_KEY"
)

MISSING_VARS=()
for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        MISSING_VARS+=($var)
    fi
done

if [ ${#MISSING_VARS[@]} -gt 0 ]; then
    echo "❌ Missing required environment variables:"
    for var in "${MISSING_VARS[@]}"; do
        echo "   - $var"
    done
    echo ""
    echo "Please set these variables before running tests."
    exit 1
fi

echo "✅ All required Azure environment variables are set"
echo ""

# Set test configuration
export API_BASE="${API_BASE:-http://localhost:8000}"
export TEST_AMBIGUOUS_QUERY="${TEST_AMBIGUOUS_QUERY:-architecture}"
export TEST_SPECIFIC_QUERY="${TEST_SPECIFIC_QUERY:-Graph Enrichment Skill}"

echo "📋 Test Configuration:"
echo "   API_BASE: $API_BASE"
echo "   TEST_AMBIGUOUS_QUERY: $TEST_AMBIGUOUS_QUERY"
echo "   TEST_SPECIFIC_QUERY: $TEST_SPECIFIC_QUERY"
if [ -n "$TEST_SPACE" ]; then
    echo "   TEST_SPACE: $TEST_SPACE"
fi
echo ""

# Start the API server in background if not already running
echo "🚀 Checking API server..."
if ! curl -s "$API_BASE/health" > /dev/null 2>&1; then
    echo "   Starting API server on port 8000..."
    python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 &
    SERVER_PID=$!

    # Wait for server to start
    for i in {1..30}; do
        if curl -s "$API_BASE/health" > /dev/null 2>&1; then
            echo "   ✅ Server is ready!"
            break
        fi
        sleep 1
    done
else
    echo "   ✅ Server is already running"
fi

echo ""
echo "========================================================================"
echo "📋 Running functional tests with real Azure data..."
echo "------------------------------------------------------------------------"

# Run the consolidated functional tests
python -m pytest tests/functional/test_functional_complete.py -v --tb=short

TEST_RESULT=$?

# Also run the updated trace SSE tests
echo ""
echo "📋 Running SSE trace tests..."
echo "------------------------------------------------------------------------"
python -m pytest tests/functional/test_trace_sse.py -v --tb=short

SSE_RESULT=$?

# Clean up server if we started it
if [ -n "$SERVER_PID" ]; then
    echo ""
    echo "🛑 Stopping server..."
    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
fi

echo ""
echo "========================================================================"

if [ $TEST_RESULT -eq 0 ] && [ $SSE_RESULT -eq 0 ]; then
    echo "✅ All functional tests passed with real Azure services!"
else
    echo "❌ Some tests failed. Please check the output above."
    exit 1
fi

echo "========================================================================"
