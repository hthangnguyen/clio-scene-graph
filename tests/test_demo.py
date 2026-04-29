"""
Quick test of demo functionality without launching Gradio.

Tests that all components work together correctly.
Run with: python tests/test_demo.py
"""

import sys
from pathlib import Path
import numpy as np
import clip

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scene_graph import SceneGraph
from src.embeddings import embed_nodes, embed_tasks
from src.information import compute_theta, compute_p_y_given_x
from src.clustering import run_aib
from src.baseline import run_threshold_baseline


def test_demo_pipeline():
    """Test the complete demo pipeline."""
    print("\n" + "="*60)
    print("Testing Demo Pipeline")
    print("="*60)
    
    # Load CLIP
    print("\n1. Loading CLIP model...")
    model, _ = clip.load("ViT-B/32", device="cpu")
    print("✓ CLIP loaded")
    
    # Create synthetic scene
    print("\n2. Creating synthetic scene...")
    sg = SceneGraph.make_synthetic(30, seed=42)
    print(f"✓ Scene: {sg.scene_id}, {sg.num_nodes()} nodes, {len(sg.edges)} edges")
    
    # Parse tasks
    print("\n3. Parsing tasks...")
    tasks = ["pick up the mug", "sit on a chair", "open the door"]
    print(f"✓ Tasks: {tasks}")
    
    # Embed
    print("\n4. Computing embeddings...")
    task_embs = embed_tasks(tasks, model)
    node_embs = embed_nodes(sg, model, tasks, alpha=0.3)
    print(f"✓ Task embeddings: {task_embs.shape}")
    print(f"✓ Node embeddings: {len(node_embs)} nodes")
    
    # Compute p(y|x)
    print("\n5. Computing task relevance...")
    p_y_given_x = {}
    for node in sg.nodes:
        theta = compute_theta(node_embs[node["id"]], task_embs, alpha=0.3)
        p_y_given_x[node["id"]] = compute_p_y_given_x(theta, alpha=0.3, l=1)
    
    n_relevant = sum(1 for p in p_y_given_x.values() if p[0] == 0.0)
    n_irrelevant = sum(1 for p in p_y_given_x.values() if p[0] == 1.0)
    print(f"✓ Relevant nodes: {n_relevant}, Irrelevant: {n_irrelevant}")
    
    # Run AIB
    print("\n6. Running AIB clustering...")
    clusters, deltas = run_aib(sg, p_y_given_x, tau=0.1, max_runtime_seconds=60.0)
    print(f"✓ Clusters: {len(clusters)}")
    print(f"✓ Merges: {len(deltas)}")
    
    # Verify partition
    all_nodes = set()
    for cluster in clusters:
        all_nodes.update(cluster)
    assert len(all_nodes) == sg.num_nodes(), "Partition violated!"
    print(f"✓ Partition valid: all {sg.num_nodes()} nodes returned")
    
    # Run baseline
    print("\n7. Running baseline...")
    retained = run_threshold_baseline(node_embs, task_embs, threshold=0.3)
    print(f"✓ Baseline retained: {len(retained)} nodes")
    
    # Compare
    aib_count = sum(len(c) for c in clusters if not all(p_y_given_x[nid][0] == 1.0 for nid in c))
    print(f"\n8. Comparison:")
    print(f"✓ AIB retained: {aib_count} nodes")
    print(f"✓ Baseline retained: {len(retained)} nodes")
    print(f"✓ AIB more compact: {aib_count < len(retained)}")
    
    print("\n" + "="*60)
    print("✅ Demo pipeline test PASSED!")
    print("="*60)
    print("\nReady to launch: python demo/app.py")


if __name__ == "__main__":
    try:
        test_demo_pipeline()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
