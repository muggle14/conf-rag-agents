#!/usr/bin/env python
"""
Seed the Cosmos DB Gremlin graph with test vertices and edges.

This script creates test vertices and edges for smoke testing the graph_feedback
and graph_lookup modules.

Required env-vars (put them in `.env`)
------------------------------------
COSMOS_ENDPOINT    https://your-cosmos.gremlin.cosmos.azure.com:443/
COSMOS_KEY         <cosmos-key>
COSMOS_DATABASE    confluence-graph
COSMOS_CONTAINER   knowledge-graph
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from gremlin_python.driver import client, serializer
from gremlin_python.process.traversal import T

# Load environment variables
load_dotenv()


def get_client():
    """Create Gremlin client."""
    endpoint = os.getenv("COSMOS_ENDPOINT")
    key = os.getenv("COSMOS_KEY")
    db = os.getenv("COSMOS_DATABASE", "confluence-graph")
    collection = os.getenv("COSMOS_CONTAINER", "knowledge-graph")

    if not endpoint or not key:
        sys.exit("Error: COSMOS_ENDPOINT and COSMOS_KEY must be set")

    # Extract hostname from the endpoint URL
    # If endpoint is like https://cosmos-rag-conf.gremlin.cosmos.azure.com:443/
    # We need just cosmos-rag-conf.gremlin.cosmos.azure.com
    if endpoint.startswith("https://"):
        endpoint = endpoint.replace("https://", "")
    if endpoint.endswith(":443/"):
        endpoint = endpoint.replace(":443/", "")
    elif endpoint.endswith("/"):
        endpoint = endpoint.rstrip("/")

    return client.Client(
        f"wss://{endpoint}:443/",
        "g",
        username=f"/dbs/{db}/colls/{collection}",
        password=key,
        message_serializer=serializer.GraphSONSerializersV2d0(),
    )


def seed_graph():
    """Create test vertices and edges."""
    g = get_client()

    print("🌱 Seeding Cosmos DB Gremlin graph...")

    # Clear existing test data (optional - be careful in production!)
    # Uncomment if you want to start fresh each time
    # g.submitAsync("g.V().has('test_data', true).drop()").result()

    # Create vertices
    vertices = [
        {
            "id": "A",
            "label": "question",
            "text": "What is the pricing structure?",
            "pageId": "A",  # Required partition key for Cosmos DB
            "test_data": True,
        },
        {
            "id": "B",
            "label": "question",
            "text": "How much does the enterprise plan cost?",
            "pageId": "B",
            "test_data": True,
        },
        {
            "id": "C",
            "label": "question",
            "text": "What features are included in professional plan?",
            "pageId": "C",
            "test_data": True,
        },
    ]

    # Create vertices
    for v in vertices:
        query = """
        g.V(vid).fold().coalesce(
            unfold(),
            addV(vlabel)
                .property(T.id, vid)
                .property('pageId', pageId)
                .property('text', vtext)
                .property('test_data', test_data)
                .property('created_at', ctime)
        )
        """

        try:
            g.submitAsync(
                query,
                bindings={
                    "vid": v["id"],
                    "vlabel": v["label"],
                    "pageId": v["pageId"],
                    "vtext": v["text"],
                    "test_data": v["test_data"],
                    "ctime": "2024-01-01T00:00:00Z",
                    "T": T,
                },
            ).result()
            print(f"✅ Created vertex: {v['id']}")
        except Exception as e:
            print(f"❌ Failed to create vertex {v['id']}: {e}")

    # Create edges
    edges = [
        ("A", "B", "DependsOn", 1.0),
        ("A", "C", "DependsOn", 0.8),
        ("B", "C", "References", 0.5),
    ]

    for src, dst, label, weight in edges:
        query = """
        g.V(src).coalesce(
            __.outE(elabel).where(inV().hasId(dst)),
            addE(elabel).to(V(dst))
                .property('weight', weight)
                .property('created_at', ctime)
        )
        """

        try:
            g.submitAsync(
                query,
                bindings={
                    "src": src,
                    "dst": dst,
                    "elabel": label,
                    "weight": weight,
                    "ctime": "2024-01-01T00:00:00Z",
                },
            ).result()
            print(f"✅ Created edge: {src} --{label}--> {dst}")
        except Exception as e:
            print(f"❌ Failed to create edge {src}->{dst}: {e}")

    # Verify the graph
    print("\n📊 Verifying graph state...")

    # Count vertices
    count_result = g.submitAsync("g.V().count()").result().all().result()
    vertex_count = count_result[0] if count_result else 0
    print(f"Total vertices: {vertex_count}")

    # Count edges
    edge_count_result = g.submitAsync("g.E().count()").result().all().result()
    edge_count = edge_count_result[0] if edge_count_result else 0
    print(f"Total edges: {edge_count}")

    # Test neighbor query for vertex A
    neighbors_query = """
    g.V('A').bothE('DependsOn', 'References').otherV().project('id', 'text').by(id).by('text')
    """
    neighbors = g.submitAsync(neighbors_query).result().all().result()
    print(f"\nNeighbors of vertex 'A': {len(neighbors)}")
    for n in neighbors:
        print(f"  - {n}")

    print("\n✨ Graph seeding complete!")
    return vertex_count >= 2  # Success if we have at least 2 vertices


if __name__ == "__main__":
    success = seed_graph()
    sys.exit(0 if success else 1)
