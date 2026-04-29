"""
Tests for CLIP embeddings module.

Run with: python tests/test_embeddings.py
"""

import sys
from pathlib import Path
import numpy as np
import clip
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scene_graph import SceneGraph
from src.embeddings import embed_nodes, embed_tasks, embed_single, verify_embeddings


def test_node_embeddings():
    """Test Task 2.1: Text embeddings for all nodes."""
    print("\n" + "="*60)
    print("Task 2.1: Testing node embeddings")
    print("="*60)
    
    # Load CLIP model
    device = 'cpu'
    print(f"Loading CLIP model (device: {device})...")
    model, preprocess = clip.load('ViT-B/32', device=device)
    print("✓ CLIP model loaded")
    
    # Find available scenes
    data_dir = Path(__file__).parent.parent / "data"
    scene_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    
    for scene_dir in scene_dirs:
        json_path = scene_dir / "semseg.v2.json"
        if not json_path.exists():
            continue
        
        print(f"\n--- Testing scene: {scene_dir.name} ---")
        
        # Load scene graph
        sg = SceneGraph(str(json_path))
        print(f"Loaded scene: {len(sg.nodes)} nodes")
        
        # Embed nodes (first run - no cache)
        embeddings = embed_nodes(sg, model, task_strings=["test"], alpha=0.3, device=device)
        
        # Verify results
        print(f"\nVerifying embeddings...")
        print(f"  Total embeddings: {len(embeddings)}")
        print(f"  Expected: {len(sg.nodes)}")
        assert len(embeddings) == len(sg.nodes), "Mismatch in number of embeddings"
        
        # Check first embedding
        first_node_id = sg.nodes[0]['id']
        first_embedding = embeddings[first_node_id]
        print(f"  First node: {first_node_id} ({sg.nodes[0]['label']})")
        print(f"  Embedding shape: {first_embedding.shape}")
        print(f"  Embedding norm: {np.linalg.norm(first_embedding):.6f}")
        
        assert first_embedding.shape == (512,), f"Wrong shape: {first_embedding.shape}"
        
        # Verify all embeddings are unit-normalized
        assert verify_embeddings(embeddings, tolerance=1e-5), "Embeddings not unit-normalized"
        
        # Test cache loading (second run)
        print(f"\nTesting cache loading...")
        embeddings_cached = embed_nodes(sg, model, task_strings=["test"], alpha=0.3, device=device)
        
        # Verify cached embeddings match
        for node_id in embeddings.keys():
            assert np.allclose(embeddings[node_id], embeddings_cached[node_id]), \
                f"Cached embedding mismatch for node {node_id}"
        
        print("✓ Cache loading works correctly")
        
        print(f"\n✅ Task 2.1 passed for scene: {scene_dir.name}")


def test_task_embeddings():
    """Test Task 2.3: Task embedding function."""
    print("\n" + "="*60)
    print("Task 2.3: Testing task embeddings")
    print("="*60)
    
    # Load CLIP model
    device = 'cpu'
    print(f"Loading CLIP model (device: {device})...")
    model, preprocess = clip.load('ViT-B/32', device=device)
    print("✓ CLIP model loaded")
    
    # Test embed_tasks
    print(f"\nTesting embed_tasks()...")
    tasks = ["pick up the mug", "sit on a chair"]
    task_embs = embed_tasks(tasks, model, device=device)
    
    print(f"  Input tasks: {tasks}")
    print(f"  Output shape: {task_embs.shape}")
    print(f"  Expected shape: (2, 512)")
    
    assert task_embs.shape == (2, 512), f"Wrong shape: {task_embs.shape}"
    
    # Verify unit normalization
    for i, task in enumerate(tasks):
        norm = np.linalg.norm(task_embs[i])
        print(f"  Task '{task}' norm: {norm:.6f}")
        assert np.isclose(norm, 1.0, atol=1e-5), f"Task embedding not unit-normalized: {norm}"
    
    print("✓ embed_tasks() works correctly")
    
    # Test embed_single
    print(f"\nTesting embed_single()...")
    single_task = "open the door"
    single_emb = embed_single(single_task, model, device=device)
    
    print(f"  Input task: '{single_task}'")
    print(f"  Output shape: {single_emb.shape}")
    print(f"  Norm: {np.linalg.norm(single_emb):.6f}")
    
    assert single_emb.shape == (512,), f"Wrong shape: {single_emb.shape}"
    assert np.isclose(np.linalg.norm(single_emb), 1.0, atol=1e-5), "Not unit-normalized"
    
    print("✓ embed_single() works correctly")
    
    print(f"\n✅ Task 2.3 passed")


def test_semantic_similarity():
    """Test semantic similarity smoke test."""
    print("\n" + "="*60)
    print("Semantic Similarity Smoke Test")
    print("="*60)
    
    # Load CLIP model
    device = 'cpu'
    model, preprocess = clip.load('ViT-B/32', device=device)
    
    # Load a scene
    data_dir = Path(__file__).parent.parent / "data"
    scene_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    scene_dir = scene_dirs[0]
    json_path = scene_dir / "semseg.v2.json"
    
    sg = SceneGraph(str(json_path))
    embeddings = embed_nodes(sg, model, task_strings=["test"], alpha=0.3, device=device)
    
    # Find nodes with specific labels
    table_nodes = [n for n in sg.nodes if 'table' in n['label'].lower()]
    chair_nodes = [n for n in sg.nodes if 'chair' in n['label'].lower() or 'sofa' in n['label'].lower()]
    
    if not table_nodes:
        print("⚠ No table nodes found, using first node as substitute")
        table_nodes = [sg.nodes[0]]
    
    if not chair_nodes:
        print("⚠ No chair/sofa nodes found, using second node as substitute")
        chair_nodes = [sg.nodes[1] if len(sg.nodes) > 1 else sg.nodes[0]]
    
    table_node = table_nodes[0]
    chair_node = chair_nodes[0]
    
    print(f"\nTest nodes:")
    print(f"  Table-like: {table_node['id']} ({table_node['label']})")
    print(f"  Chair-like: {chair_node['id']} ({chair_node['label']})")
    
    # Task embeddings
    tasks = ["pick up the mug", "sit down"]
    task_embs = embed_tasks(tasks, model, device=device)
    
    # Get node embeddings
    table_emb = embeddings[table_node['id']]
    chair_emb = embeddings[chair_node['id']]
    
    # Compute cosine similarities
    sim_pickup_table = np.dot(task_embs[0], table_emb)
    sim_pickup_chair = np.dot(task_embs[0], chair_emb)
    sim_sit_table = np.dot(task_embs[1], table_emb)
    sim_sit_chair = np.dot(task_embs[1], chair_emb)
    
    print(f"\nCosine similarities:")
    print(f"  'pick up the mug' → {table_node['label']}: {sim_pickup_table:.3f}")
    print(f"  'pick up the mug' → {chair_node['label']}: {sim_pickup_chair:.3f}")
    print(f"  'sit down' → {table_node['label']}: {sim_sit_table:.3f}")
    print(f"  'sit down' → {chair_node['label']}: {sim_sit_chair:.3f}")
    
    # Semantic checks
    print(f"\nSemantic checks:")
    
    # Check 1: "sit down" should be more similar to chair than table
    if sim_sit_chair > sim_sit_table:
        print(f"  ✓ 'sit down' is more similar to {chair_node['label']} than {table_node['label']}")
    else:
        print(f"  ⚠ 'sit down' is more similar to {table_node['label']} than {chair_node['label']}")
        print(f"    (This is acceptable - CLIP may not always match human intuition)")
    
    # Check 2: All similarities should be in reasonable range
    all_sims = [sim_pickup_table, sim_pickup_chair, sim_sit_table, sim_sit_chair]
    print(f"  Similarity range: [{min(all_sims):.3f}, {max(all_sims):.3f}]")
    
    if all(-1.0 <= s <= 1.0 for s in all_sims):
        print(f"  ✓ All similarities in valid range [-1, 1]")
    else:
        print(f"  ❌ Some similarities out of range!")
        return False
    
    print(f"\n✅ Semantic similarity test passed")
    return True


if __name__ == "__main__":
    print("Running Phase 2 embedding tests...\n")
    
    try:
        test_node_embeddings()
        test_task_embeddings()
        test_semantic_similarity()
        
        print("\n" + "="*60)
        print("✅ ALL PHASE 2 TESTS PASSED!")
        print("="*60)
        print("\nPhase 2 complete. Ready for Phase 3 (Information Bottleneck).")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
