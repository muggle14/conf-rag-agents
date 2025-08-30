#!/usr/bin/env python
"""
Seed the Azure AI Search index `confluence-graph-embeddings-v2`
with a few Confluence page JSON files.

➤  Usage
    python scripts/seed_search_index.py \
           --dir ./sample_pages \
           --top 3   # uploads any 3 files it finds (default = *all*)

Required env-vars  (put them in `.env` and `source .env`)
--------------------------------------------------------
AZURE_SEARCH_ENDPOINT      https://<svc>.search.windows.net
AZURE_SEARCH_KEY           <admin-key>
AZURE_OPENAI_ENDPOINT      https://<openai>.openai.azure.com
AZURE_OPENAI_API_KEY       <openai-key>
AOAI_EMBED_DEPLOY          text-embedding-ada-002   # or your deployment name
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys
from typing import Any, Dict, List

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from bs4 import BeautifulSoup  # pip install beautifulsoup4
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables
load_dotenv()


# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #
def plain_text_from_html(html: str) -> str:
    """Strip HTML to plain text (Confluence storage format)."""
    return BeautifulSoup(html or "", "html.parser").get_text(separator=" ", strip=True)


def extract_doc(page: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalises a *raw* or *processed* Confluence page JSON into
    the schema expected by the search index.
    """
    # ---- IDs ----
    pid = str(page.get("id") or page.get("page_id") or random.randint(1, 1_000_000))
    title = page.get("title") or page.get("name") or "Untitled Page"

    # ---- Page body ----
    if "content" in page:
        body = page["content"]
    else:  # raw export has body.storage.value (HTML)
        body = plain_text_from_html(
            page.get("body", {}).get("storage", {}).get("value", "")
        )

    # ---- Relationships (optional) ----
    parent_id = page.get("parent_page_id") or page.get("ancestors", [{}])[-1].get("id")
    children = page.get("children_ids") or [
        str(c.get("id")) for c in page.get("children", {}).get("page", [])
    ]
    siblings = page.get("adjacent_ids") or []  # you can compute these later

    return dict(
        id=pid,
        title=title,
        content=body,
        url=page.get("url") or f"/wiki/pages/{pid}",
        parent_page_id=parent_id,
        children_ids=children,
        adjacent_ids=siblings,
        graph_centrality_score=page.get("graph_centrality_score", 0.0),
    )


def get_embedding(client: AzureOpenAI, text: str) -> List[float]:
    """Call Azure OpenAI once per page – cache outside this demo if needed."""
    resp = client.embeddings.create(
        input=text[:8_000],  # keep tokens reasonable
        model=os.getenv("AOAI_EMBED_DEPLOY", "text-embedding-ada-002"),
    )
    return resp.data[0].embedding


# --------------------------------------------------------------------------- #
#  Main seeding routine                                                       #
# --------------------------------------------------------------------------- #
def main(dirpath: pathlib.Path, top: int | None):
    # ----- clients -----
    search = SearchClient(
        os.environ["AZURE_SEARCH_ENDPOINT"],
        index_name="confluence-graph-embeddings-v2",
        credential=AzureKeyCredential(os.environ["AZURE_SEARCH_KEY"]),
    )
    aoai = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version="2023-05-15",
    )

    # ----- load files -----
    json_files = list(dirpath.glob("*.json"))
    if not json_files:
        sys.exit(f"No *.json files found under {dirpath}")

    if top:
        json_files = random.sample(json_files, min(top, len(json_files)))

    docs = []
    for fp in json_files:
        print(f"Processing: {fp.name}")
        page_raw = json.loads(fp.read_text("utf-8"))
        doc = extract_doc(page_raw)

        # Generate embeddings for both content and title
        doc["content_vector"] = get_embedding(aoai, doc["content"])
        doc["title_vector"] = get_embedding(aoai, doc["title"])

        docs.append(doc)
        print(f"  ✓ {doc['title']} (ID: {doc['id']})")

    # ----- upload -----
    print(f"\nUploading {len(docs)} documents...")
    result = search.upload_documents(documents=docs)
    succeeded = sum([r.succeeded for r in result])
    failed = len(result) - succeeded

    print(f"\n✅ Uploaded {succeeded}/{len(docs)} documents")
    if failed > 0:
        print(f"❌ Failed: {failed}")
        for r in result:
            if not r.succeeded:
                print(f"  - {r.key}: {r.error_message}")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Seed Azure AI Search with sample Confluence pages"
    )
    p.add_argument(
        "--dir",
        required=True,
        type=pathlib.Path,
        help="Directory that contains *.json Confluence pages",
    )
    p.add_argument(
        "--top",
        type=int,
        default=None,
        help="Only upload this many random files (for quick tests)",
    )
    args = p.parse_args()
    main(args.dir, args.top)
