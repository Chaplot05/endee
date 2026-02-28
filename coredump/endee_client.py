"""
Endee Vector Database REST API Client
Complete wrapper for Endee's HTTP API with batching, error handling, and health checks.
"""

import requests
import os
import uuid
import time

ENDEE_URL = os.getenv("ENDEE_URL", "http://localhost:8080")


def create_index(name: str, dimension: int = 384):
    """Create a new vector index in Endee.
    
    Args:
        name: Unique name for the index
        dimension: Vector dimensionality (768 for CodeBERT)
    
    Returns:
        dict: API response with index creation status
    """
    payload = {
        "name": name,
        "dimension": dimension,
        "metric": "cosine"
    }
    try:
        r = requests.post(f"{ENDEE_URL}/api/v1/index/create", json=payload, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def insert_vectors(index_name: str, vectors: list):
    """Insert vectors into an Endee index in batches of 100.
    
    Args:
        index_name: Target index name
        vectors: List of dicts with keys: id (str), vector (list[float]), metadata (dict)
    
    Returns:
        list: API responses for each batch
    """
    batch_size = 100
    results = []
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        try:
            r = requests.post(
                f"{ENDEE_URL}/api/v1/index/{index_name}/insert",
                json={"vectors": batch},
                timeout=30
            )
            r.raise_for_status()
            results.append(r.json())
        except requests.exceptions.RequestException as e:
            results.append({"error": str(e), "batch_start": i})
    return results


def search(index_name: str, query_vector: list, top_k: int = 10):
    """Search for similar vectors in an Endee index.
    
    Args:
        index_name: Index to search in
        query_vector: Query embedding vector
        top_k: Number of nearest neighbors to return
    
    Returns:
        dict: Search results with scores and metadata
    """
    try:
        r = requests.post(
            f"{ENDEE_URL}/api/v1/index/{index_name}/search",
            json={"vector": query_vector, "top_k": top_k},
            timeout=15
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def list_indexes():
    """List all indexes in Endee.
    
    Returns:
        dict: API response with list of index names and metadata
    """
    try:
        r = requests.get(f"{ENDEE_URL}/api/v1/index/list", timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def delete_index(name: str):
    """Delete an index from Endee.
    
    Args:
        name: Index name to delete
    
    Returns:
        dict: API response confirming deletion
    """
    try:
        r = requests.delete(f"{ENDEE_URL}/api/v1/index/{name}", timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def ping():
    """Check if Endee server is running and reachable.
    
    Returns:
        bool: True if Endee is online, False otherwise
    """
    try:
        r = requests.get(f"{ENDEE_URL}/api/v1/index/list", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def generate_vector_id():
    """Generate a unique vector ID.
    
    Returns:
        str: UUID string for use as vector identifier
    """
    return str(uuid.uuid4())
