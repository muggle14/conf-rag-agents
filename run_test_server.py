#!/usr/bin/env python3
"""Run the test server for functional testing."""

import sys

sys.path.append(".")

import uvicorn

from tests.test_tracing_functional import create_test_app

if __name__ == "__main__":
    app = create_test_app()
    print("🚀 Starting test server on port 8003...")
    print("   Access at: http://localhost:8003")
    print("   Test endpoint: POST http://localhost:8003/ask")
    print("   Verify services: GET http://localhost:8003/test/verify-services")
    print("\n   Press Ctrl+C to stop\n")

    uvicorn.run(app, host="0.0.0.0", port=8003, log_level="info")
