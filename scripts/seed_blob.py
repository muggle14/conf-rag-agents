#!/usr/bin/env python
"""
Seed Azure Blob Storage with test files for smoke testing.

This script creates test blobs in the raw and processed containers
for testing blob storage functionality.

Required env-vars (put them in `.env`)
------------------------------------
STORAGE_ACCOUNT             stgragconf
STORAGE_KEY                 <storage-key>
RAW_CONTAINER              raw
PROC_CONTAINER             processed
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime

from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_blob_service():
    """Create blob service client."""
    account = os.getenv("STORAGE_ACCOUNT")
    key = os.getenv("STORAGE_KEY")

    if not account or not key:
        sys.exit("Error: STORAGE_ACCOUNT and STORAGE_KEY must be set")

    connection_string = f"DefaultEndpointsProtocol=https;AccountName={account};AccountKey={key};EndpointSuffix=core.windows.net"
    return BlobServiceClient.from_connection_string(connection_string)


def seed_blobs():
    """Upload test blobs to storage containers."""
    service = get_blob_service()
    raw_container = os.getenv("RAW_CONTAINER", "raw")
    proc_container = os.getenv("PROC_CONTAINER", "processed")

    print("🌱 Seeding Azure Blob Storage...")

    # Ensure containers exist
    for container_name in [raw_container, proc_container]:
        try:
            container = service.get_container_client(container_name)
            if not container.exists():
                container.create_container()
                print(f"✅ Created container: {container_name}")
            else:
                print(f"✅ Container exists: {container_name}")
        except Exception as e:
            print(f"❌ Failed with container {container_name}: {e}")

    # Sample Confluence export data
    raw_pages = [
        {
            "id": "4001",
            "title": "API Documentation",
            "body": {
                "storage": {
                    "value": "<h1>API Documentation</h1><p>Our REST API provides comprehensive access to all system features.</p>"
                }
            },
            "url": "https://confluence.example.com/pages/4001",
            "created": "2024-01-15T10:00:00Z",
        },
        {
            "id": "4002",
            "title": "Integration Guide",
            "body": {
                "storage": {
                    "value": "<h1>Integration Guide</h1><p>Follow these steps to integrate with third-party services.</p>"
                }
            },
            "url": "https://confluence.example.com/pages/4002",
            "created": "2024-01-16T11:30:00Z",
        },
        {
            "id": "4003",
            "title": "Security Best Practices",
            "body": {
                "storage": {
                    "value": "<h1>Security Best Practices</h1><p>Essential security guidelines for production deployments.</p>"
                }
            },
            "url": "https://confluence.example.com/pages/4003",
            "created": "2024-01-17T09:15:00Z",
        },
    ]

    # Upload raw pages
    raw_client = service.get_container_client(raw_container)
    for page in raw_pages:
        blob_name = f"confluence/pages/{page['id']}.json"
        blob_client = raw_client.get_blob_client(blob_name)

        try:
            blob_client.upload_blob(
                json.dumps(page, indent=2),
                overwrite=True,
                metadata={
                    "source": "confluence",
                    "page_id": page["id"],
                    "uploaded_at": datetime.utcnow().isoformat(),
                },
            )
            print(f"✅ Uploaded raw blob: {blob_name}")
        except Exception as e:
            print(f"❌ Failed to upload {blob_name}: {e}")

    # Create processed versions
    proc_client = service.get_container_client(proc_container)
    for page in raw_pages:
        # Simulate processed data
        processed = {
            "id": page["id"],
            "title": page["title"],
            "content": page["body"]["storage"]["value"]
            .replace("<h1>", "")
            .replace("</h1>", "")
            .replace("<p>", "")
            .replace("</p>", ""),
            "url": page["url"],
            "parent_page_id": None,
            "children_ids": [],
            "adjacent_ids": [],
            "graph_centrality_score": 0.5,
            "processed_at": datetime.utcnow().isoformat(),
            "embeddings_generated": False,
        }

        blob_name = f"confluence/processed/{page['id']}.json"
        blob_client = proc_client.get_blob_client(blob_name)

        try:
            blob_client.upload_blob(
                json.dumps(processed, indent=2),
                overwrite=True,
                metadata={
                    "source": "confluence",
                    "page_id": page["id"],
                    "processed_at": datetime.utcnow().isoformat(),
                },
            )
            print(f"✅ Uploaded processed blob: {blob_name}")
        except Exception as e:
            print(f"❌ Failed to upload {blob_name}: {e}")

    # Verify uploads
    print("\n📊 Verifying blob storage...")

    for container_name in [raw_container, proc_container]:
        container = service.get_container_client(container_name)
        blobs = list(container.list_blobs(name_starts_with="confluence/"))
        print(f"\n{container_name} container:")
        print(f"  Total blobs: {len(blobs)}")
        for blob in blobs[:5]:  # Show first 5
            print(f"  - {blob.name} ({blob.size} bytes)")

    print("\n✨ Blob storage seeding complete!")
    return True


if __name__ == "__main__":
    success = seed_blobs()
    sys.exit(0 if success else 1)
