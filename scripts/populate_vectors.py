#!/usr/bin/env python3
"""
Script to populate vector embeddings for existing documents in Azure Search index.
This script reads documents without vectors and generates embeddings using Azure OpenAI.
"""

import os
import time
from typing import Any, Dict, List

import requests
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from dotenv import load_dotenv
from openai import AzureOpenAI
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configuration
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AOAI_EMBED_DEPLOY = os.getenv("AOAI_EMBED_DEPLOY", "text-embedding-ada-002")
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "confluence-graph-embeddings-v2")

# Initialize clients
search_client = SearchClient(
    AZURE_SEARCH_ENDPOINT, INDEX_NAME, AzureKeyCredential(AZURE_SEARCH_KEY)
)

openai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2023-05-15",
)


def generate_embedding(text: str) -> List[float]:
    """Generate embedding for text using Azure OpenAI."""
    try:
        response = openai_client.embeddings.create(input=text, model=AOAI_EMBED_DEPLOY)
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None


def get_documents_without_vectors(batch_size: int = 50) -> List[Dict[str, Any]]:
    """Retrieve documents that don't have vector embeddings."""
    documents = []

    # Search for all documents
    results = search_client.search(
        search_text="*", select=["id", "title", "content"], top=batch_size
    )

    for doc in results:
        documents.append(doc)

    return documents


def update_document_vectors(documents: List[Dict[str, Any]]) -> int:
    """Update documents with vector embeddings."""
    updated_count = 0

    # Prepare batch update
    batch = []

    for doc in tqdm(documents, desc="Generating embeddings"):
        doc_id = doc.get("id")
        title = doc.get("title", "")
        content = doc.get("content", "")

        if not content and not title:
            print(f"Skipping document {doc_id}: No content or title")
            continue

        # Generate embeddings
        content_vector = None
        title_vector = None

        if content:
            content_vector = generate_embedding(content)
            if not content_vector:
                print(f"Failed to generate content vector for document {doc_id}")
                continue

        if title:
            title_vector = generate_embedding(title)
            if not title_vector:
                print(f"Failed to generate title vector for document {doc_id}")
                continue

        # Prepare update
        update_doc = {"@search.action": "merge", "id": doc_id}

        if content_vector:
            update_doc["content_vector"] = content_vector
        if title_vector:
            update_doc["title_vector"] = title_vector

        batch.append(update_doc)

        # Upload batch when it reaches 100 documents or at the end
        if len(batch) >= 100:
            if upload_batch(batch):
                updated_count += len(batch)
            batch = []

    # Upload remaining documents
    if batch:
        if upload_batch(batch):
            updated_count += len(batch)

    return updated_count


def upload_batch(batch: List[Dict[str, Any]]) -> bool:
    """Upload a batch of documents to Azure Search."""
    url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{INDEX_NAME}/docs/index?api-version=2023-11-01"
    headers = {"api-key": AZURE_SEARCH_KEY, "Content-Type": "application/json"}

    payload = {"value": batch}

    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            print(f"✅ Successfully uploaded batch of {len(batch)} documents")
            return True
        else:
            print(
                f"❌ Failed to upload batch: {response.status_code} - {response.text}"
            )
            return False
    except Exception as e:
        print(f"❌ Error uploading batch: {e}")
        return False


def verify_vectors():
    """Verify that vectors have been populated."""
    # Test vector search
    print("\n🔍 Testing vector search...")

    test_queries = ["security", "confluence", "documentation"]

    for query in test_queries:
        print(f"\nQuery: '{query}'")

        # Generate query embedding
        query_vector = generate_embedding(query)
        if not query_vector:
            print("Failed to generate query embedding")
            continue

        # Search using REST API
        url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{INDEX_NAME}/docs/search?api-version=2023-11-01"
        headers = {"api-key": AZURE_SEARCH_KEY, "Content-Type": "application/json"}

        payload = {
            "count": True,
            "select": "id,title",
            "top": 3,
            "vectorQueries": [
                {
                    "kind": "vector",
                    "vector": query_vector,
                    "fields": "content_vector",
                    "k": 3,
                }
            ],
        }

        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            results = response.json()
            count = results.get("@odata.count", 0)
            print(f"  Found {count} results")
            for i, result in enumerate(results.get("value", [])[:3], 1):
                print(
                    f"  {i}. {result.get('title', 'No title')} (Score: {result.get('@search.score', 0):.4f})"
                )
        else:
            print(f"  Error: {response.status_code}")


def main():
    """Main function to populate vectors for all documents."""
    print("🚀 Starting vector population process...")
    print(f"Index: {INDEX_NAME}")
    print(f"Embedding model: {AOAI_EMBED_DEPLOY}")

    # Get documents without vectors
    print("\n📄 Fetching documents...")
    documents = get_documents_without_vectors(batch_size=100)
    print(f"Found {len(documents)} documents to process")

    if not documents:
        print("No documents found to process")
        return

    # Update documents with vectors
    print("\n🔄 Generating and uploading embeddings...")
    updated_count = update_document_vectors(documents)
    print(f"\n✅ Updated {updated_count} documents with embeddings")

    # Wait for indexing
    print("\n⏳ Waiting for indexing to complete...")
    time.sleep(10)

    # Verify vectors
    verify_vectors()

    print("\n✨ Vector population complete!")


if __name__ == "__main__":
    main()
# .venv/bin/activate && python scripts/populate_vectors.py
