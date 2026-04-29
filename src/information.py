"""
Information-theoretic functions for Agglomerative Information Bottleneck.

Implements all formulas from Clio paper (Maggio et al., 2024):
- Formula 1: Task relevance vector θ(x_i)
- Formula 2: Conditional distribution p(y|x_i) with binary invariant
- Formula 3: Merge weight d_ij
- Formula 4: Fractional information loss δ(k)
"""

import numpy as np
from typing import List, Dict


def compute_theta(node_emb: np.ndarray,
                  task_embs: np.ndarray,
                  alpha: float) -> np.ndarray:
    """
    Compute task relevance vector θ(x_i).
    
    Formula 1:
      θ(x_i)[0]   = alpha                                   (null task score)
      θ(x_i)[j]   = dot(f_xi, f_tj)   for j = 1, ..., m   (cosine similarity)
    
    Args:
        node_emb: CLIP embedding of node i, shape (512,), unit vector
        task_embs: CLIP embeddings of tasks, shape (m, 512), unit vectors
        alpha: Null task threshold (typically 0.3)
        
    Returns:
        θ(x_i) of shape (m+1,)
        Index 0: alpha (null task)
        Indices 1..m: cosine similarities to tasks
    """
    task_sims = task_embs @ node_emb  # shape (m,) — cosine similarity
    theta = np.concatenate([[alpha], task_sims])  # shape (m+1,)
    return theta


def compute_p_y_given_x(theta: np.ndarray,
                         alpha: float,
                         l: int = 1) -> np.ndarray:
    """
    Compute conditional distribution p(y|x_i).
    
    Formula 2 with binary invariant:
      p(y|x_i)[0] is EITHER 1.0 (irrelevant) OR 0.0 (relevant) — never fractional
    
    Args:
        theta: Task relevance vector from compute_theta(), shape (m+1,)
        alpha: Null task threshold
        l: Number of top tasks to keep (default 1, fixed per v3 spec)
        
    Returns:
        p(y|x_i) of shape (m+1,) summing to 1.0
        Index 0 = null task. Indices 1..m = task probabilities.
    """
    task_sims = theta[1:]  # shape (m,)
    m = len(task_sims)
    
    # Check if node is task-irrelevant
    if task_sims.max() < alpha:
        # Node is TASK-IRRELEVANT
        p = np.zeros(m + 1)
        p[0] = 1.0  # All mass on null task
        return p
    
    # Node is TASK-RELEVANT — softmax over top-l tasks only
    top_indices = np.argsort(task_sims)[-l:]  # indices of top-l tasks
    scores = np.full(m, -np.inf)
    scores[top_indices] = task_sims[top_indices]
    
    # Numerically stable softmax
    finite = scores[scores > -np.inf]
    if len(finite) > 0:
        scores_shifted = scores - finite.max()
    else:
        scores_shifted = scores
    
    exp_scores = np.exp(scores_shifted)
    exp_scores[scores == -np.inf] = 0.0
    task_probs = exp_scores / exp_scores.sum()  # shape (m,)
    
    # Assemble final vector
    p = np.zeros(m + 1)
    p[0] = 0.0  # Null task: 0 (node IS relevant)
    p[1:] = task_probs
    
    # Invariant check
    if abs(p.sum() - 1.0) > 1e-5:
        print(f"WARNING: p(y|x) sum = {p.sum():.6f} — renormalizing")
        p /= p.sum()
    
    return p


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """
    Compute KL divergence KL(p || q).
    
    Handles p[k]=0 by treating 0*log(0)=0.
    
    Args:
        p: Probability distribution, shape (n,)
        q: Probability distribution, shape (n,)
        eps: Small constant to avoid log(0)
        
    Returns:
        KL(p || q) as float
    """
    result = 0.0
    for pk, qk in zip(p, q):
        if pk > 0:
            result += pk * np.log(pk / (qk + eps))
    return float(result)


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute Jensen-Shannon divergence.
    
    JS(p, q) = 0.5 * KL(p || m) + 0.5 * KL(q || m)
    where m = 0.5 * (p + q)
    
    Symmetric. Output in [0, log(2)].
    
    Args:
        p: Probability distribution, shape (n,)
        q: Probability distribution, shape (n,)
        
    Returns:
        JS(p, q) as float
    """
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def cluster_distribution(cluster_ids: List[str],
                          p_y_given_x: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Compute mean p(y|x) over all nodes in a cluster.
    
    For cluster C with nodes {x_1, ..., x_n}:
      p(y|C) = (1/n) * sum_i p(y|x_i)
    Then renormalize to sum=1.0
    
    Args:
        cluster_ids: List of node IDs in cluster
        p_y_given_x: Dict mapping node_id -> p(y|x) vector
        
    Returns:
        p(y|C) of shape (m+1,), normalized to sum=1.0
    """
    stack = np.array([p_y_given_x[nid] for nid in cluster_ids])
    mean = stack.mean(axis=0)
    total = mean.sum()
    if total > 0:
        mean /= total
    return mean


def mutual_information(clusters: List[List[str]],
                        p_y_given_x: Dict[str, np.ndarray],
                        total_nodes: int) -> float:
    """
    Compute mutual information I(X; Y) for a clustering state.
    
    Formula 4:
      I(X; Y) = sum over clusters C:
                  p(C) * sum over tasks j:
                    p(y=j|C) * log(p(y=j|C) / p(y=j))
    
    where p(C) = |C| / total_nodes
          p(y=j) = (1/K) * sum_c p(y=j|c)  (uniform weight over clusters)
    
    Args:
        clusters: List of node_id lists (partition of all nodes)
        p_y_given_x: Dict mapping node_id -> p(y|x) vector
        total_nodes: Total nodes in scene (for computing p(C))
        
    Returns:
        I(X; Y) as float (always >= 0)
    """
    cluster_dists = [cluster_distribution(c, p_y_given_x) for c in clusters]
    K = len(clusters)
    
    # Marginal p(y) — uniform weight over clusters
    p_y = np.array(cluster_dists).mean(axis=0)  # shape (m+1,)
    
    I = 0.0
    for c, p_c_given_x in zip(clusters, cluster_dists):
        p_c = len(c) / total_nodes
        for j in range(len(p_y)):
            if p_c_given_x[j] > 0 and p_y[j] > 0:
                I += p_c * p_c_given_x[j] * np.log(p_c_given_x[j] / p_y[j])
    
    return float(max(I, 0.0))  # clamp to 0 — can be slightly negative due to float errors


def merge_weight(cluster_i: List[str],
                 cluster_j: List[str],
                 p_y_given_x: Dict[str, np.ndarray],
                 total_nodes: int) -> float:
    """
    Compute merge weight d_ij for two clusters.
    
    Formula 3:
      d_ij = (p(x_i) + p(x_j)) * JS(p(y|x_i), p(y|x_j))
    
    where p(x_i) = |cluster_i| / total_nodes
    
    Args:
        cluster_i: List of node IDs in cluster i
        cluster_j: List of node IDs in cluster j
        p_y_given_x: Dict mapping node_id -> p(y|x) vector
        total_nodes: Total nodes in scene
        
    Returns:
        d_ij as float
    """
    p_i = len(cluster_i) / total_nodes
    p_j = len(cluster_j) / total_nodes
    dist_i = cluster_distribution(cluster_i, p_y_given_x)
    dist_j = cluster_distribution(cluster_j, p_y_given_x)
    return (p_i + p_j) * js_divergence(dist_i, dist_j)
