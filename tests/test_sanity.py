"""
Sanity tests for Task-Driven Scene Graph Explorer (v3).

Tests Phase 1 (SceneGraph) and Phase 2 (Embeddings).
Run with: python tests/test_sanity.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import clip

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scene_graph import SceneGraph
from src.embeddings import embed_nodes, embed_tasks, embed_single, verify_embeddings
from src.information import (
    compute_theta, compute_p_y_given_x, js_divergence, kl_divergence,
    cluster_distribution, mutual_information, merge_weight
)


def test_scene_graph_loading():
    """Test that SceneGraph can load 3RScan data with background filtering."""
    print("\n" + "="*60)
    print("Test: Scene Graph Loading")
    print("="*60)
    
    # Find available scenes
    data_dir = Path(__file__).parent.parent / "data"
    scene_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    
    assert len(scene_dirs) > 0, "No scene directories found in data/"
    
    # Test loading first scene
    scene_dir = scene_dirs[0]
    json_path = scene_dir / "semseg.v2.json"
    
    assert json_path.exists(), f"semseg.v2.json not found in {scene_dir}"
    
    # Load scene graph
    sg = SceneGraph(str(json_path))
    
    # Verify basic properties
    assert len(sg.nodes) > 0, "Scene graph has no nodes"
    assert all('id' in node for node in sg.nodes), "Not all nodes have 'id'"
    assert all('label' in node for node in sg.nodes), "Not all nodes have 'label'"
    assert all('position' in node for node in sg.nodes), "Not all nodes have 'position'"
    assert all('extent' in node for node in sg.nodes), "Not all nodes have 'extent'"
    
    # Verify no background labels
    labels = [node['label'] for node in sg.nodes]
    background = {"wall", "floor", "ceiling", "floor mat", "unknown"}
    assert not any(l in background for l in labels), "Background labels not filtered"
    
    print(f"✓ Loaded scene: {sg.scene_id}")
    print(f"✓ Nodes: {len(sg.nodes)}")
    print(f"✓ Edges: {len(sg.edges)}")
    print(f"✓ No background labels found")
    
    # Test summary
    summary = sg.summary()
    assert "Scene:" in summary
    assert "Nodes:" in summary
    print(f"\n{summary}")


def test_neighbors():
    """Test that get_neighbors returns valid results."""
    print("\n" + "="*60)
    print("Test: Neighbor Queries")
    print("="*60)
    
    data_dir = Path(__file__).parent.parent / "data"
    scene_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    scene_dir = scene_dirs[0]
    json_path = scene_dir / "semseg.v2.json"
    
    sg = SceneGraph(str(json_path))
    
    # Test get_neighbors for first node
    if len(sg.nodes) > 0:
        first_node_id = sg.nodes[0]['id']
        neighbors = sg.get_neighbors(first_node_id)
        
        print(f"✓ Node {first_node_id} ({sg.get_node(first_node_id)['label']}) has {len(neighbors)} neighbors")
        
        # Verify neighbors are valid node IDs
        for neighbor_id in neighbors:
            neighbor = sg.get_node(neighbor_id)
            assert neighbor is not None, f"Neighbor {neighbor_id} not found"


def test_synthetic_scene():
    """Test synthetic scene generation."""
    print("\n" + "="*60)
    print("Test: Synthetic Scene Generation")
    print("="*60)
    
    # Generate synthetic scenes
    sg_small = SceneGraph.make_synthetic(30, seed=42)
    sg_large = SceneGraph.make_synthetic(80, seed=7)
    
    # Verify determinism
    sg_small_2 = SceneGraph.make_synthetic(30, seed=42)
    assert sg_small.num_nodes() == sg_small_2.num_nodes()
    
    # Verify no isolated nodes
    for sg in [sg_small, sg_large]:
        isolated = [n for n in sg.nodes if len(sg.get_neighbors(n['id'])) == 0]
        assert len(isolated) == 0, f"Found {len(isolated)} isolated nodes"
    
    print(f"✓ Synthetic small: {sg_small.num_nodes()} nodes, {len(sg_small.edges)} edges")
    print(f"✓ Synthetic large: {sg_large.num_nodes()} nodes, {len(sg_large.edges)} edges")
    print(f"✓ No isolated nodes")


def test_visualization():
    """Test networkx graph visualization."""
    print("\n" + "="*60)
    print("Test: Graph Visualization")
    print("="*60)
    
    # Use synthetic scene for guaranteed success
    sg = SceneGraph.make_synthetic(30, seed=0)
    
    # Create networkx graph
    G = nx.Graph()
    
    # Add nodes
    for node in sg.nodes:
        G.add_node(node['id'], label=node['label'])
    
    # Add edges
    for node_id_a, node_id_b, relation in sg.edges:
        G.add_edge(node_id_a, node_id_b)
    
    # Check graph properties
    assert G.number_of_nodes() == len(sg.nodes)
    assert G.number_of_edges() == len(sg.edges)
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    
    # Use spring layout
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightblue', alpha=0.7)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    
    # Draw labels
    labels = {node['id']: node['label'] for node in sg.nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    plt.title(f"Scene Graph: {sg.scene_id}\n{len(sg.nodes)} nodes, {len(sg.edges)} edges")
    plt.axis('off')
    plt.tight_layout()
    
    # Save visualization
    output_dir = Path(__file__).parent.parent / "experiments" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "scene_preview.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to: {output_path}")
    
    plt.close()


def test_embeddings():
    """Test CLIP embeddings with task/alpha-aware caching."""
    print("\n" + "="*60)
    print("Test: CLIP Embeddings")
    print("="*60)
    
    # Load CLIP
    device = 'cpu'
    print(f"Loading CLIP model (device: {device})...")
    model, _ = clip.load('ViT-B/32', device=device)
    print("✓ CLIP model loaded")
    
    # Use synthetic scene
    sg = SceneGraph.make_synthetic(30, seed=0)
    task_strings = ["pick up the mug", "sit on a chair"]
    alpha = 0.3
    
    # Embed nodes (first run - no cache)
    print(f"\nEmbedding {sg.num_nodes()} nodes...")
    embeddings = embed_nodes(sg, model, task_strings, alpha, device=device)
    
    # Verify results
    print(f"✓ Total embeddings: {len(embeddings)}")
    assert len(embeddings) == len(sg.nodes), "Mismatch in number of embeddings"
    
    # Check first embedding
    first_node_id = sg.nodes[0]['id']
    first_embedding = embeddings[first_node_id]
    print(f"✓ First node: {first_node_id} ({sg.nodes[0]['label']})")
    print(f"✓ Embedding shape: {first_embedding.shape}")
    print(f"✓ Embedding norm: {np.linalg.norm(first_embedding):.6f}")
    
    assert first_embedding.shape == (512,), f"Wrong shape: {first_embedding.shape}"
    
    # Verify all embeddings are unit-normalized
    assert verify_embeddings(embeddings, tolerance=1e-5), "Embeddings not unit-normalized"
    
    # Test cache loading (second run)
    print(f"\nTesting cache loading...")
    embeddings_cached = embed_nodes(sg, model, task_strings, alpha, device=device)
    
    # Verify cached embeddings match
    for node_id in embeddings.keys():
        assert np.allclose(embeddings[node_id], embeddings_cached[node_id]), \
            f"Cached embedding mismatch for node {node_id}"
    
    print("✓ Cache loading works correctly")
    
    # Test task embeddings
    print(f"\nTesting task embeddings...")
    task_embs = embed_tasks(task_strings, model, device=device)
    print(f"✓ Task embeddings shape: {task_embs.shape}")
    assert task_embs.shape == (2, 512), f"Wrong shape: {task_embs.shape}"
    
    # Test single embedding
    single_emb = embed_single("open the door", model, device=device)
    print(f"✓ Single embedding shape: {single_emb.shape}")
    assert single_emb.shape == (512,), f"Wrong shape: {single_emb.shape}"


def test_information_functions():
    """Test information-theoretic functions."""
    print("\n" + "="*60)
    print("Test: Information Functions")
    print("="*60)
    
    # Test JS divergence
    p = np.array([0.7, 0.2, 0.1])
    q = np.array([0.1, 0.6, 0.3])
    
    js_pq = js_divergence(p, q)
    js_qp = js_divergence(q, p)
    assert abs(js_pq - js_qp) < 1e-9, "JS divergence not symmetric"
    print(f"✓ JS divergence symmetric: {js_pq:.6f}")
    
    # Test JS identity
    js_pp = js_divergence(p, p)
    assert js_pp < 1e-6, f"JS(p, p) should be ~0, got {js_pp}"
    print(f"✓ JS(p, p) = {js_pp:.6f}")
    
    # Test p(y|x) binary invariant
    theta = np.array([0.3, 0.8, 0.4])  # alpha=0.3, 2 tasks
    p_y = compute_p_y_given_x(theta, alpha=0.3, l=1)
    assert p_y[0] == 0.0, "Relevant node should have p[0]=0"
    assert abs(p_y.sum() - 1.0) < 1e-5, "p(y|x) doesn't sum to 1"
    print(f"✓ p(y|x) binary invariant: p[0]={p_y[0]}, sum={p_y.sum():.6f}")
    
    # Test irrelevant node
    theta_irr = np.array([0.3, 0.1, 0.2])  # max task sim 0.2 < alpha 0.3
    p_y_irr = compute_p_y_given_x(theta_irr, alpha=0.3, l=1)
    assert p_y_irr[0] == 1.0, "Irrelevant node should have p[0]=1"
    assert np.allclose(p_y_irr[1:], 0.0), "Irrelevant node should have 0 task mass"
    print(f"✓ Irrelevant node: p[0]={p_y_irr[0]}")
    
    print("\n✓ All information functions passed")


if __name__ == "__main__":
    print("Running Phase 1-2 sanity tests...\n")
    
    try:
        test_scene_graph_loading()
        test_neighbors()
        test_synthetic_scene()
        test_visualization()
        test_embeddings()
        test_information_functions()
        
        print("\n" + "="*60)
        print("✅ ALL SANITY TESTS PASSED!")
        print("="*60)
        print("\nReady for Phase 3 (Information Bottleneck core).")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
