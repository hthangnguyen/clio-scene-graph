"""
Clio-Prim baseline: threshold-based filtering without clustering.

Simple baseline that retains nodes whose max cosine similarity to any task >= threshold.
No merging, no information-theoretic optimization.
"""

import numpy as np
from typing import Set


def run_threshold_baseline(node_embeddings: dict,
                            task_embeddings: np.ndarray,
                            threshold: float = 0.3) -> Set[str]:
    """
    Clio-Prim baseline: retain nodes whose max task similarity >= threshold.
    
    For each node:
      max_sim = max cosine_similarity(node_embedding, task_embedding) over all tasks
      IF max_sim >= threshold: retain node as singleton cluster
      ELSE: discard (do not include in output)
    
    Returns:
        Set of retained node_ids (individual nodes, no merging)
    
    Metric note: compare len(retained) vs sum(cluster sizes for non-null AIB clusters).
    Both count individual nodes — this is the fair comparison.
    """
    retained = set()
    for node_id, emb in node_embeddings.items():
        sims = task_embeddings @ emb  # shape (m,)
        if sims.max() >= threshold:
            retained.add(node_id)
    return retained
