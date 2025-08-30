"""
Shared pytest fixtures and configuration.
"""

import os

import pytest
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@pytest.fixture(scope="session")
def azure_credentials_available():
    """Check if Azure credentials are available."""
    return all(
        [
            os.getenv("AZURE_SEARCH_ENDPOINT"),
            os.getenv("AZURE_SEARCH_KEY"),
            os.getenv("AZURE_OPENAI_ENDPOINT"),
            os.getenv("AZURE_OPENAI_API_KEY"),
        ]
    )


@pytest.fixture
def mock_search_results():
    """Provide mock search results for testing."""
    return [
        {
            "id": "1474624",
            "title": "🔐 Security Incident Process",
            "content": "If Dynatrace shows an alert that could mean a security risk...",
            "url": "https://confluence.example.com/pages/1474624",
            "@search.score": 2.5,
            "parent_page_id": "1376582",
            "children_ids": [],
            "adjacent_ids": ["1474615"],
            "graph_centrality_score": 0.097178,
        },
        {
            "id": "98408",
            "title": "Getting started in Confluence",
            "content": "Welcome to Confluence! You can use Confluence to collaborate...",
            "url": "https://confluence.example.com/pages/98408",
            "@search.score": 2.1,
            "parent_page_id": None,
            "children_ids": ["1474615", "1474594"],
            "adjacent_ids": [],
            "graph_centrality_score": 0.15,
        },
    ]


@pytest.fixture
def mock_embedding():
    """Provide a mock embedding vector."""
    return [0.1] * 1536  # Standard embedding size for text-embedding-ada-002
