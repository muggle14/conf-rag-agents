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


# Functional test fixtures for API testing
@pytest.fixture(scope="session")
def api_base():
    """Base URL for API testing."""
    return os.getenv("API_BASE", "http://localhost:8000").rstrip("/")


@pytest.fixture(scope="session")
def ambiguous_query():
    """Query that should trigger clarification."""
    return os.getenv("TEST_AMBIGUOUS_QUERY", "architecture")


@pytest.fixture(scope="session")
def specific_query():
    """Query that should return direct answer."""
    # Pick a page you know exists in your Confluence index
    return os.getenv("TEST_SPECIFIC_QUERY", "Graph Enrichment Skill")


@pytest.fixture(scope="session")
def test_space():
    """Optional Confluence space filter."""
    # Optional: constrain to a single space if your index supports it
    return os.getenv("TEST_SPACE", None)


# Synthesis testing fixtures
@pytest.fixture(scope="session")
def answer_min_words():
    """Minimum word count for synthesized answers."""
    return int(os.getenv("ANSWER_MIN_WORDS", "150"))


@pytest.fixture(scope="session")
def answer_max_words():
    """Maximum word count for synthesized answers."""
    return int(os.getenv("ANSWER_MAX_WORDS", "300"))


@pytest.fixture(scope="session")
def allowed_domains():
    """Allowed domains for source citations."""
    domains = os.getenv("ALLOWED_SOURCE_DOMAINS", "")
    return [d.strip().lower() for d in domains.split(",") if d.strip()]
