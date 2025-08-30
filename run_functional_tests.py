#!/usr/bin/env python3
"""
Run functional tests for the Confluence Q&A API.

This script:
1. Starts the API server
2. Runs the functional tests
3. Provides a summary of results
"""

import os
import subprocess
import sys
import time


def start_api_server():
    """Start the API server in a separate thread."""
    print("🚀 Starting API server on port 8000...")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "api.app:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return process


def wait_for_server(port=8000, timeout=30):
    """Wait for the server to be ready."""
    import httpx

    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = httpx.get(f"http://localhost:{port}/health")
            if response.status_code == 200:
                print("✅ Server is ready!")
                return True
        except:
            pass
        time.sleep(1)

    return False


def run_tests():
    """Run the functional tests."""
    print("\n📋 Running functional tests...")
    print("-" * 60)

    # Set environment variables for testing
    os.environ["API_BASE"] = "http://localhost:8000"
    os.environ["TEST_AMBIGUOUS_QUERY"] = "architecture"
    os.environ["TEST_SPECIFIC_QUERY"] = "Graph Enrichment Skill"

    # Run pytest on functional tests
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/functional/", "-v", "--tb=short"],
        check=False,
        capture_output=False,
    )

    return result.returncode


def main():
    """Main test runner."""
    print("=" * 60)
    print("🧪 Confluence Q&A API Functional Test Runner")
    print("=" * 60)

    # Start the server
    server_process = start_api_server()

    try:
        # Wait for server to be ready
        if not wait_for_server():
            print("❌ Server failed to start!")
            return 1

        # Run the tests
        test_result = run_tests()

        # Print summary
        print("\n" + "=" * 60)
        if test_result == 0:
            print("✅ All tests passed!")
        else:
            print("❌ Some tests failed. Check the output above.")
        print("=" * 60)

        return test_result

    finally:
        # Clean up: terminate the server
        print("\n🛑 Stopping server...")
        server_process.terminate()
        time.sleep(1)
        if server_process.poll() is None:
            server_process.kill()
        print("👋 Server stopped.")


if __name__ == "__main__":
    sys.exit(main())
