# blob_client.py
"""
Lightweight Azure Blob Storage client for accessing Confluence page content.
"""

import json
import logging
import os
from typing import Optional

from azure.identity import DefaultAzureCredential

try:
    from azure.storage.blob import BlobServiceClient

    BLOB_STORAGE_AVAILABLE = True
except ImportError:
    BLOB_STORAGE_AVAILABLE = False
    BlobServiceClient = None

logger = logging.getLogger(__name__)


class BlobStorageClient:
    """Lightweight client for accessing Confluence content in Azure Blob Storage."""

    def __init__(self):
        """Initialize blob storage client."""
        if not BLOB_STORAGE_AVAILABLE:
            logger.warning(
                "azure-storage-blob not installed - blob storage unavailable"
            )
            self.blob_service = None
            self.raw_container = None
            self.processed_container = None
            return

        try:
            credential = DefaultAzureCredential()
            storage_account = os.environ.get("STORAGE_ACCOUNT")

            if not storage_account:
                logger.warning(
                    "STORAGE_ACCOUNT not configured - blob storage unavailable"
                )
                self.blob_service = None
                self.raw_container = None
                self.processed_container = None
                return

            self.blob_service = BlobServiceClient(
                account_url=f"https://{storage_account}.blob.core.windows.net",
                credential=credential,
            )

            # Container clients
            self.raw_container = self.blob_service.get_container_client(
                "raw-confluence"
            )
            self.processed_container = self.blob_service.get_container_client(
                "processed-confluence"
            )

            logger.info(f"Connected to blob storage: {storage_account}")

        except Exception as e:
            logger.error(f"Failed to initialize blob storage: {e}")
            self.blob_service = None

    def get_page_content(self, page_id: str) -> Optional[str]:
        """
        Retrieve page content from Azure Blob Storage.

        Args:
            page_id: Confluence page ID

        Returns:
            Page content as string, or None if not found
        """
        if not self.blob_service:
            return None

        blob_name = f"{page_id}.json"

        # Try processed container first
        try:
            blob_client = self.processed_container.get_blob_client(blob_name)
            content = blob_client.download_blob().readall().decode("utf-8")
            logger.debug(f"Found page {page_id} in processed container")
            return content
        except Exception:
            pass

        # Fall back to raw container
        try:
            blob_client = self.raw_container.get_blob_client(blob_name)
            content = blob_client.download_blob().readall().decode("utf-8")
            logger.debug(f"Found page {page_id} in raw container")
            return content
        except Exception as e:
            logger.warning(f"Page {page_id} not found in blob storage: {e}")
            return None

    def get_parsed_page_content(self, page_id: str) -> Optional[dict]:
        """
        Retrieve and parse page content from blob storage.

        Args:
            page_id: Confluence page ID

        Returns:
            Parsed page data as dictionary, or None if not found
        """
        content = self.get_page_content(page_id)

        if not content:
            return None

        try:
            page_data = json.loads(content)
            return page_data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse page {page_id}: {e}")
            return None

    def get_page_body(self, page_id: str) -> Optional[str]:
        """
        Retrieve just the page body content.

        Args:
            page_id: Confluence page ID

        Returns:
            Page body HTML content, or None if not found
        """
        page_data = self.get_parsed_page_content(page_id)

        if not page_data:
            return None

        # Navigate the Confluence page structure
        try:
            body = page_data.get("body", {}).get("storage", {}).get("value", "")
            return body
        except (KeyError, TypeError):
            # Try alternative structure
            return page_data.get("content", "")
