#!/usr/bin/env python3
"""
Phase 5: Ablation Studies

Compare AIB vs Clio-Prim baseline across:
- Scenes: both real 3RScan scenes
- Task sets: 4 sets with increasing specificity
- Tau values: [0.05, 0.1, 0.2, 0.3, 0.5]

Metrics:
- Compression ratio: (initial_nodes - final_clusters) / initial_nodes
- Task coverage: fraction of task-relevant nodes retained
- Node retention: fraction of all nodes retained

Output:
- experiments/results/ablation_table.csv
- experiments/results/compression_plot.png
- experiments/results/coverage_plot.png
"""

import sys
import os
import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import clip

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scene_graph import SceneGraph
from src.embeddings import embed_nodes, embed_tasks
from src.information import compute_theta, compute_p_y_given_x
from src.clustering import run_aib
from src.baseline import run_threshold_baseline


# Task sets with increasing specificity
TASK_SETS = {
    "generic": [
        "find an object",
        "locate something",
    ],
    "furniture": [
        "find a chair",
        "find a table",
        "find a sofa",
    ],
    "specific": [
        "pick up the mug",
        "sit on the chair",
        "put the book on the table",
    ],
    "complex": [
        "find a comfortable place to sit",
        "locate something to drink from",
        "find a surface to work on",
        "identify an object to read",
    ],
}

TAU_VALUES = [0.05, 0.1, 0.2, 0.3, 0.5]
ALPHA = 0.3
MAX_RUNTIME = 60.0


def get_scene_files() -> List[str]:
    """Find all 3RScan scene files."""
    # Try data/scenes/ first
    data_dir = Path(__file__).parent.parent / "data" / "scenes"
    scene_files = sorted(data_dir.glob("*/semseg.v2.json"))
    
    # If not found, try data/ directly
    if not scene_files:
        data_dir = Path(__file__).parent.parent / "data"
        scene_files = sorted(data_dir.glob("*/semseg.v2.json"))
    
    if not scene_files:
        # Fall back to synthetic
        print("[Ablation] No real scenes found. Using synthetic scenes.")
        return None
    return [str(f) for f in scene_files[:2]]  # Use first 2 scenes


def compute_metrics(
    scene_graph: SceneGraph,
    p_y_given_x: Dict[str, np.ndarray],
    aib_clusters: List[List[str]],
    baseline_retained: set,
) -> Dict[str, float]:
    """
    Compute metrics for a single run.
    
    Returns:
        Dict with keys:
        - aib_compression: (n_nodes - n_clusters) / n_nodes
        - aib_coverage: n_task_relevant_retained / n_task_relevant_total
        - aib_retention: n_nodes_retained / n_nodes
        - baseline_compression: (n_nodes - n_retained) / n_nodes
        - baseline_coverage: n_task_relevant_retained / n_task_relevant_total
        - baseline_retention: n_retained / n_nodes
    """
    n_nodes = scene_graph.num_nodes()
    
    # Count task-relevant nodes
    n_task_relevant = sum(1 for nid in p_y_given_x if p_y_given_x[nid][0] == 0.0)
    
    # AIB metrics
    aib_retained = set()
    for cluster in aib_clusters:
        aib_retained.update(cluster)
    
    aib_n_clusters = len(aib_clusters)
    aib_compression = (n_nodes - aib_n_clusters) / n_nodes if n_nodes > 0 else 0.0
    
    aib_task_relevant_retained = sum(
        1 for nid in aib_retained if p_y_given_x[nid][0] == 0.0
    )
    aib_coverage = aib_task_relevant_retained / n_task_relevant if n_task_relevant > 0 else 0.0
    aib_retention = len(aib_retained) / n_nodes if n_nodes > 0 else 0.0
    
    # Baseline metrics
    baseline_n_retained = len(baseline_retained)
    baseline_compression = (n_nodes - baseline_n_retained) / n_nodes if n_nodes > 0 else 0.0
    
    baseline_task_relevant_retained = sum(
        1 for nid in baseline_retained if p_y_given_x[nid][0] == 0.0
    )
    baseline_coverage = baseline_task_relevant_retained / n_task_relevant if n_task_relevant > 0 else 0.0
    baseline_retention = baseline_n_retained / n_nodes if n_nodes > 0 else 0.0
    
    return {
        "aib_compression": aib_compression,
        "aib_coverage": aib_coverage,
        "aib_retention": aib_retention,
        "aib_n_clusters": aib_n_clusters,
        "baseline_compression": baseline_compression,
        "baseline_coverage": baseline_coverage,
        "baseline_retention": baseline_n_retained,
    }


def run_ablation():
    """Run full ablation study."""
    print("=" * 70)
    print("PHASE 5: ABLATION STUDIES")
    print("=" * 70)
    
    # Load CLIP model
    print("\n[Setup] Loading CLIP model...")
    device = "cpu"
    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()
    
    # Get scene files
    scene_files = get_scene_files()
    if scene_files is None:
        print("[Ablation] Using synthetic scenes for demonstration.")
        scene_files = []
        for i, seed in enumerate([42, 123]):
            sg = SceneGraph.make_synthetic(n_nodes=30 + i*10, seed=seed)
            scene_files.append(sg)
    else:
        scene_files = [SceneGraph(f) for f in scene_files]
    
    results = []
    
    # Main ablation loop
    total_runs = len(scene_files) * len(TASK_SETS) * len(TAU_VALUES)
    run_count = 0
    
    for scene_idx, sg in enumerate(scene_files):
        if isinstance(sg, str):
            sg = SceneGraph(sg)
        
        scene_id = sg.scene_id
        n_nodes = sg.num_nodes()
        
        print(f"\n{'='*70}")
        print(f"Scene {scene_idx + 1}/{len(scene_files)}: {scene_id} ({n_nodes} nodes)")
        print(f"{'='*70}")
        
        for task_set_name, task_strings in TASK_SETS.items():
            print(f"\n  Task set: {task_set_name} ({len(task_strings)} tasks)")
            
            # Embed nodes once per task set (using actual task strings for cache key)
            print(f"  [Embed] Computing node embeddings...")
            node_embs = embed_nodes(sg, model, task_strings, ALPHA)
            
            # Embed tasks
            task_embs = embed_tasks(task_strings, model)
            
            # Compute p(y|x) for all nodes
            p_y_given_x = {}
            for node_id, node_emb in node_embs.items():
                theta = compute_theta(node_emb, task_embs, ALPHA)
                p_y_given_x[node_id] = compute_p_y_given_x(theta, ALPHA, l=1)
            
            # Baseline (once per task set)
            baseline_retained = run_threshold_baseline(node_embs, task_embs, threshold=ALPHA)
            
            for tau in TAU_VALUES:
                run_count += 1
                print(f"    [{run_count}/{total_runs}] tau={tau:.2f}...", end=" ", flush=True)
                
                # Run AIB
                aib_clusters, delta_values = run_aib(
                    sg, p_y_given_x, tau=tau, max_runtime_seconds=MAX_RUNTIME
                )
                
                # Compute metrics
                metrics = compute_metrics(sg, p_y_given_x, aib_clusters, baseline_retained)
                
                result = {
                    "scene_id": scene_id,
                    "n_nodes": n_nodes,
                    "task_set": task_set_name,
                    "n_tasks": len(task_strings),
                    "tau": tau,
                    "aib_compression": metrics["aib_compression"],
                    "aib_coverage": metrics["aib_coverage"],
                    "aib_retention": metrics["aib_retention"],
                    "aib_n_clusters": metrics["aib_n_clusters"],
                    "baseline_compression": metrics["baseline_compression"],
                    "baseline_coverage": metrics["baseline_coverage"],
                    "baseline_retention": metrics["baseline_retention"],
                }
                results.append(result)
                
                print(f"✓ (AIB: {metrics['aib_n_clusters']} clusters, "
                      f"coverage={metrics['aib_coverage']:.2f})")
    
    # Save results to CSV
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / "ablation_table.csv"
    print(f"\n[Output] Saving results to {csv_path}...")
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    print(f"✓ Saved {len(results)} results")
    
    # Generate plots
    print(f"\n[Plots] Generating visualizations...")
    generate_plots(results, output_dir)
    
    print("\n" + "=" * 70)
    print("ABLATION COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {output_dir}")
    print(f"  - ablation_table.csv")
    print(f"  - compression_plot.png")
    print(f"  - coverage_plot.png")


def generate_plots(results: List[Dict], output_dir: Path):
    """Generate comparison plots."""
    
    # Plot 1: Compression vs Tau
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Group by task set
    task_sets = sorted(set(r["task_set"] for r in results))
    colors = plt.cm.Set2(np.linspace(0, 1, len(task_sets)))
    
    for task_set, color in zip(task_sets, colors):
        subset = [r for r in results if r["task_set"] == task_set]
        taus = sorted(set(r["tau"] for r in subset))
        
        aib_comp = [np.mean([r["aib_compression"] for r in subset if r["tau"] == t]) for t in taus]
        baseline_comp = [subset[0]["baseline_compression"]] * len(taus)  # constant
        
        axes[0].plot(taus, aib_comp, marker="o", label=f"{task_set} (AIB)", color=color)
        axes[0].axhline(baseline_comp[0], linestyle="--", color=color, alpha=0.5, label=f"{task_set} (Baseline)")
    
    axes[0].set_xlabel("Tau (information loss threshold)")
    axes[0].set_ylabel("Compression ratio")
    axes[0].set_title("Compression: AIB vs Baseline")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Coverage vs Tau
    for task_set, color in zip(task_sets, colors):
        subset = [r for r in results if r["task_set"] == task_set]
        taus = sorted(set(r["tau"] for r in subset))
        
        aib_cov = [np.mean([r["aib_coverage"] for r in subset if r["tau"] == t]) for t in taus]
        baseline_cov = [subset[0]["baseline_coverage"]] * len(taus)  # constant
        
        axes[1].plot(taus, aib_cov, marker="o", label=f"{task_set} (AIB)", color=color)
        axes[1].axhline(baseline_cov[0], linestyle="--", color=color, alpha=0.5, label=f"{task_set} (Baseline)")
    
    axes[1].set_xlabel("Tau (information loss threshold)")
    axes[1].set_ylabel("Task coverage")
    axes[1].set_title("Coverage: AIB vs Baseline")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "compression_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ compression_plot.png")
    
    # Plot 3: Coverage plot (separate)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for task_set, color in zip(task_sets, colors):
        subset = [r for r in results if r["task_set"] == task_set]
        taus = sorted(set(r["tau"] for r in subset))
        
        aib_cov = [np.mean([r["aib_coverage"] for r in subset if r["tau"] == t]) for t in taus]
        baseline_cov = [subset[0]["baseline_coverage"]] * len(taus)
        
        ax.plot(taus, aib_cov, marker="o", linewidth=2, label=f"{task_set} (AIB)", color=color)
        ax.plot(taus, baseline_cov, marker="s", linestyle="--", linewidth=2, 
                label=f"{task_set} (Baseline)", color=color, alpha=0.6)
    
    ax.set_xlabel("Tau (information loss threshold)", fontsize=12)
    ax.set_ylabel("Task coverage (fraction of relevant nodes retained)", fontsize=12)
    ax.set_title("Task Coverage: AIB vs Clio-Prim Baseline", fontsize=14)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_dir / "coverage_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ coverage_plot.png")


if __name__ == "__main__":
    run_ablation()
