"""
Agglomerative Information Bottleneck clustering.

Implements the AIB loop from Clio paper with:
- Pre-filtering of task-irrelevant nodes
- Lazy updates for merge weights
- Runtime limit enforcement
- Partition guarantee
"""

import time
import numpy as np
from typing import List, Tuple
from src.information import (
    cluster_distribution, mutual_information, merge_weight
)


def run_aib(scene_graph,
            p_y_given_x: dict,
            tau: float = 0.1,
            max_runtime_seconds: float = 60.0) -> Tuple[List[List[str]], List[float]]:
    """
    Run Agglomerative Information Bottleneck clustering.
    
    Algorithm:
    1. Pre-filter: separate irrelevant nodes (p[0]==1.0) from clustering
    2. Initialize: each relevant node is its own cluster
    3. Build adjacency: from scene graph edges, restricted to relevant nodes
    4. Main loop: greedily merge adjacent clusters with minimum d_ij
    5. Stop when: δ(k) > tau or no more adjacent pairs
    
    Returns:
        clusters: List of node_id lists (partition of ALL nodes)
        delta_values: List of δ at each merge step (for plotting)
    """
    print(f"[AIB] Starting with tau={tau}, max_runtime={max_runtime_seconds}s")
    print(f"[AIB] l=1 (top-1 task selection — fixed)")
    start = time.time()
    
    all_node_ids = [n["id"] for n in scene_graph.nodes]
    total_nodes = len(all_node_ids)
    
    # STEP 1: Pre-filter — separate irrelevant nodes from clustering
    relevant_ids = []
    irrelevant_ids = []
    for nid in all_node_ids:
        if p_y_given_x[nid][0] == 1.0:
            irrelevant_ids.append(nid)
        else:
            relevant_ids.append(nid)
    
    n_relevant = len(relevant_ids)
    print(f"[AIB] Relevant nodes: {n_relevant}, Irrelevant (skipped): {len(irrelevant_ids)}")
    
    if n_relevant == 0:
        print("[AIB] WARNING: No task-relevant nodes found. Returning all as singletons.")
        return [[nid] for nid in all_node_ids], []
    
    if n_relevant > 150:
        print(f"[AIB] WARNING: {n_relevant} relevant nodes. Runtime may approach limit.")
    
    # STEP 2: Initialize clusters (relevant nodes only)
    # cluster_id -> list of node_ids
    next_cid = 0
    clusters = {}
    node_to_cluster = {}
    for nid in relevant_ids:
        clusters[next_cid] = [nid]
        node_to_cluster[nid] = next_cid
        next_cid += 1
    
    # STEP 3: Build adjacency (from scene_graph edges, restricted to relevant nodes)
    # adjacent_pairs: set of frozensets {cid_a, cid_b}
    adjacent_pairs = set()
    for (a, b, _) in scene_graph.edges:
        ca = node_to_cluster.get(a)
        cb = node_to_cluster.get(b)
        if ca is not None and cb is not None and ca != cb:
            adjacent_pairs.add(frozenset([ca, cb]))
    
    if len(adjacent_pairs) == 0:
        print("[AIB] WARNING: No adjacent pairs among relevant nodes. No merges possible.")
        result_clusters = [[nid] for nid in relevant_ids] + [[nid] for nid in irrelevant_ids]
        return result_clusters, []
    
    # STEP 4: Precompute cluster distributions and MI
    cluster_dists = {cid: cluster_distribution(nids, p_y_given_x)
                     for cid, nids in clusters.items()}
    I_0 = mutual_information(list(clusters.values()), p_y_given_x, total_nodes)
    
    if I_0 == 0:
        print("[AIB] WARNING: I_0 = 0, all nodes equally irrelevant. No merging.")
        result_clusters = [[nid] for nid in relevant_ids] + [[nid] for nid in irrelevant_ids]
        return result_clusters, []
    
    # STEP 5: Initialize d_cache (lazy — only adjacent pairs)
    d_cache = {}
    for pair in adjacent_pairs:
        ca, cb = sorted(pair)
        d_cache[(ca, cb)] = merge_weight(clusters[ca], clusters[cb], p_y_given_x, total_nodes)
    
    I_prev = I_0
    delta_values = []
    step = 0
    
    # STEP 6: Main AIB loop
    while adjacent_pairs and d_cache:
        if time.time() - start > max_runtime_seconds:
            print(f"[AIB] Runtime limit reached at step {step}. Stopping.")
            break
        
        # Find minimum-weight adjacent pair
        best_pair_key = min(d_cache, key=d_cache.get)
        ca, cb = best_pair_key
        
        # Tentative merge: compute new distribution and MI
        merged_ids = clusters[ca] + clusters[cb]
        merged_dist = cluster_distribution(merged_ids, p_y_given_x)
        
        temp_clusters = {cid: nids for cid, nids in clusters.items()
                         if cid != ca and cid != cb}
        temp_clusters[next_cid] = merged_ids
        temp_dists = {cid: cluster_dists[cid] for cid in temp_clusters
                      if cid != next_cid}
        temp_dists[next_cid] = merged_dist
        
        I_new = mutual_information(list(temp_clusters.values()), p_y_given_x, total_nodes)
        delta = (I_prev - I_new) / I_0
        
        if delta > tau:
            print(f"[AIB] Step {step}: delta={delta:.4f} > tau={tau}. Stopping.")
            break
        
        # Commit merge
        cid_new = next_cid
        next_cid += 1
        clusters = temp_clusters
        cluster_dists = temp_dists
        
        # Update node_to_cluster mapping
        for nid in merged_ids:
            node_to_cluster[nid] = cid_new
        
        # Update adjacency: replace ca and cb with cid_new
        new_adjacent = set()
        for pair in adjacent_pairs:
            if ca in pair or cb in pair:
                other = (pair - {ca, cb})
                if other:
                    other_cid = list(other)[0]
                    if other_cid != cid_new and other_cid in clusters:
                        new_pair = frozenset([cid_new, other_cid])
                        new_adjacent.add(new_pair)
            else:
                new_adjacent.add(pair)
        adjacent_pairs = new_adjacent
        
        # LAZY UPDATE: recompute d_cache only for pairs involving cid_new
        d_cache = {k: v for k, v in d_cache.items()
                   if ca not in k and cb not in k}
        for pair in adjacent_pairs:
            if cid_new in pair:
                other_cid = list(pair - {cid_new})[0]
                key = tuple(sorted([cid_new, other_cid]))
                if key not in d_cache:
                    d_cache[key] = merge_weight(
                        clusters[cid_new], clusters[other_cid], p_y_given_x, total_nodes
                    )
        
        I_prev = I_new
        delta_values.append(delta)
        step += 1
    
    elapsed = time.time() - start
    print(f"[AIB] Done: {step} merges, {len(clusters)} clusters in {elapsed:.1f}s")
    
    # Reassemble: relevant clusters + irrelevant singletons
    result_clusters = list(clusters.values()) + [[nid] for nid in irrelevant_ids]
    return result_clusters, delta_values
