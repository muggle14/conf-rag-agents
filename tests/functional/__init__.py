"""
Functional tests for the Confluence Q&A API.

These tests verify the end-to-end functionality of the API endpoints,
including /ask, SSE tracing, and space filtering.
"""

# Ensure optional SSE dependency is present; skip functional tests otherwise
import pytest

pytest.importorskip(
    "sseclient",
    reason="Optional dependency for SSE-based functional tests (install via .[test])",
)
