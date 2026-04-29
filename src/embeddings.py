"""
CLIP embedding functions for scene graph nodes and task strings.

Implements v3 specification with task/alpha-aware caching.
Text-only embeddings (no image crops available in 3RScan).
"""

import clip
import torch
import numpy as np
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Optional


CACHE_DIR = Path("data/scenes/cache")


def _cache_key(scene_id: str, task_strings: List[str], alpha: float) -> str:
    """
    Generate cache key that includes scene_id, sorted tasks, and alpha.
    Changing ANY of these invalidates the cache.
    
    Args:
        scene_id: Scene identifier
        task_strings: List of task descriptions
        alpha: Null task threshold
        
    Returns:
        16-character hex string cache key
    """
    payload = {
        "scene": scene_id,
        "tasks": sorted(task_strings),
        "alpha": round(alpha, 4)
    }
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def embed_nodes(
    scene_graph,
    clip_model,
    task_strings: List[str],
    alpha: float,
    device: str = 'cpu'
) -> Dict[str, np.ndarray]:
    """
    Compute unit-normalized CLIP text embeddings for all scene nodes.
    Uses text-only embeddings (no image crops available in 3RScan).
    
    Cache key includes scene_id, task_strings, and alpha.
    Changing any of these creates a new cache file.
    
    Args:
        scene_graph: SceneGraph instance
        clip_model: CLIP model (from clip.load())
        task_strings: List of task descriptions (for cache key)
        alpha: Null task threshold (for cache key)
        device: Device to run on ('cpu' or 'cuda')
        
    Returns:
        Dictionary mapping node_id (str) -> embedding (512-d unit vector)
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(scene_graph.scene_id, task_strings, alpha)
    cache_path = CACHE_DIR / f"{scene_graph.scene_id}_{key}.npy"
    id_path = CACHE_DIR / f"{scene_graph.scene_id}_{key}_ids.json"
    
    # Check cache
    if cache_path.exists() and id_path.exists():
        print(f"[Cache] Loading embeddings from {cache_path}")
        emb_array = np.load(str(cache_path))
        node_ids = json.loads(id_path.read_text())
        return dict(zip(node_ids, emb_array))
    
    print(f"[Embed] Computing text embeddings for {scene_graph.num_nodes()} nodes...")
    
    node_ids = [n["id"] for n in scene_graph.nodes]
    labels = [n["label"] for n in scene_graph.nodes]
    
    # Batch tokenize and encode
    with torch.no_grad():
        tokens = clip.tokenize(labels).to(device)  # shape: (N, 77)
        embs = clip_model.encode_text(tokens).float()  # shape: (N, 512)
        embs = embs / embs.norm(dim=-1, keepdim=True)  # unit normalize
        embs_np = embs.cpu().numpy()  # (N, 512)
    
    embeddings = dict(zip(node_ids, embs_np))
    
    # Save cache
    np.save(str(cache_path), embs_np)
    id_path.write_text(json.dumps(node_ids))
    print(f"[Cache] Saved to {cache_path}")
    
    return embeddings


def embed_tasks(
    task_strings: List[str],
    clip_model,
    device: str = 'cpu'
) -> np.ndarray:
    """
    Compute CLIP text embeddings for task strings.
    
    Args:
        task_strings: List of task descriptions (e.g., ["pick up the mug", "sit on a chair"])
        clip_model: CLIP model (from clip.load())
        device: Device to run on ('cpu' or 'cuda')
        
    Returns:
        Numpy array of shape (num_tasks, 512) with unit-normalized embeddings
    """
    if not task_strings:
        raise ValueError("task_strings cannot be empty")
    
    # Tokenize all tasks
    text_tokens = clip.tokenize(task_strings).to(device)
    
    # Encode with CLIP
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens).float()
        
        # Normalize each row to unit vector
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Convert to numpy
        embeddings = text_features.cpu().numpy()  # Shape: (num_tasks, 512)
    
    return embeddings


def embed_single(
    text: str,
    clip_model,
    device: str = 'cpu'
) -> np.ndarray:
    """
    Convenience wrapper to embed a single text string.
    
    Args:
        text: Single text string to embed
        clip_model: CLIP model (from clip.load())
        device: Device to run on ('cpu' or 'cuda')
        
    Returns:
        Numpy array of shape (512,) with unit-normalized embedding
    """
    result = embed_tasks([text], clip_model, device)
    return result[0]


def clear_cache(scene_id: str):
    """
    Clear all cached embeddings for a specific scene.
    
    Args:
        scene_id: Scene identifier
    """
    cache_files = list(CACHE_DIR.glob(f"{scene_id}_*.npy"))
    id_files = list(CACHE_DIR.glob(f"{scene_id}_*_ids.json"))
    
    for f in cache_files + id_files:
        f.unlink()
        print(f"Cleared cache: {f}")
    
    if not cache_files and not id_files:
        print(f"No cache found for scene: {scene_id}")


def verify_embeddings(embeddings: Dict[str, np.ndarray], tolerance: float = 1e-5) -> bool:
    """
    Verify that all embeddings are unit-normalized.
    
    Args:
        embeddings: Dictionary of node_id -> embedding
        tolerance: Tolerance for norm check (default 1e-5)
        
    Returns:
        True if all embeddings are valid, False otherwise
    """
    for node_id, embedding in embeddings.items():
        norm = np.linalg.norm(embedding)
        if not np.isclose(norm, 1.0, atol=tolerance):
            print(f"❌ Node {node_id} has norm {norm:.6f} (expected 1.0)")
            return False
    
    print(f"✓ All {len(embeddings)} embeddings are unit-normalized")
    return True
