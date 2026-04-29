#!/usr/bin/env python3
"""
Task-Driven Scene Graph Explorer - Flask Web UI

A simple web interface for exploring AIB clustering on scene graphs.
Run with: python demo/app.py
Then open http://127.0.0.1:5000 in your browser
"""

import sys
from pathlib import Path
import json
import io
import base64
from datetime import datetime

import numpy as np
import clip
import torch
import matplotlib.pyplot as plt
import networkx as nx
from flask import Flask, render_template_string, request, jsonify

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scene_graph import SceneGraph
from src.embeddings import embed_nodes, embed_tasks
from src.information import compute_theta, compute_p_y_given_x
from src.clustering import run_aib
from src.baseline import run_threshold_baseline


# Initialize Flask app
app = Flask(__name__)

# Load CLIP model once at startup
print("Loading CLIP model...")
device = 'cpu'
clip_model, _ = clip.load('ViT-B/32', device=device)
clip_model.eval()
print("✓ CLIP model loaded.")

# Find available scenes
data_dir = Path(__file__).parent.parent / "data"
scene_files = sorted(data_dir.glob("*/semseg.v2.json"))
scene_choices = []

for scene_file in scene_files:
    scene_id = scene_file.parent.name
    scene_choices.append(scene_id)

# Add synthetic scenes
scene_choices.append("[synthetic-30]")
scene_choices.append("[synthetic-80]")

print(f"\nAvailable scenes: {len(scene_choices)}")
for sc in scene_choices:
    print(f"  - {sc}")


def load_scene(scene_choice):
    """Load a scene (real or synthetic)."""
    if scene_choice.startswith("[synthetic"):
        n_nodes = int(scene_choice.split("-")[1].rstrip("]"))
        return SceneGraph.make_synthetic(n_nodes=n_nodes, seed=42)
    else:
        scene_file = data_dir / scene_choice / "semseg.v2.json"
        return SceneGraph(str(scene_file))


def render_graph_image(sg, p_y_given_x, clusters):
    """Render scene graph as PNG image."""
    G = nx.Graph()
    
    # Add nodes with cluster info
    node_to_cluster = {}
    for cluster_idx, cluster in enumerate(clusters):
        for node_id in cluster:
            node_to_cluster[node_id] = cluster_idx
            node = sg.get_node(node_id)
            is_irrelevant = p_y_given_x[node_id][0] == 1.0
            G.add_node(node_id, label=node['label'], cluster=cluster_idx, irrelevant=is_irrelevant)
    
    # Add edges
    for a, b, _ in sg.edges:
        if a in node_to_cluster and b in node_to_cluster:
            G.add_edge(a, b)
    
    # Create layout
    pos = {n["id"]: (n["position"][0], n["position"][1]) for n in sg.nodes}
    
    # Color nodes by cluster
    colors = []
    for node_id in G.nodes():
        if G.nodes[node_id]['irrelevant']:
            colors.append('#CCCCCC')  # Gray for irrelevant
        else:
            cluster_idx = G.nodes[node_id]['cluster']
            # Use a color palette
            palette = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2']
            colors.append(palette[cluster_idx % len(palette)])
    
    # Draw
    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#CCCCCC', width=0.5, alpha=0.5)
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=300, ax=ax)
    
    labels = {n: G.nodes[n]['label'][:8] for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax)
    
    ax.set_title(f"Scene: {sg.scene_id} | {len(clusters)} clusters")
    ax.axis('off')
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    plt.close()
    
    return f"data:image/png;base64,{img_base64}"


# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Task-Driven Scene Graph Explorer</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 8px; margin-bottom: 30px; }
        header h1 { font-size: 28px; margin-bottom: 10px; }
        header p { opacity: 0.9; }
        .main { display: grid; grid-template-columns: 1fr 2fr; gap: 20px; }
        .controls { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .controls h2 { font-size: 18px; margin-bottom: 15px; color: #333; }
        .form-group { margin-bottom: 15px; }
        .form-group label { display: block; font-weight: 500; margin-bottom: 5px; color: #555; font-size: 14px; }
        .form-group input, .form-group select, .form-group textarea { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; font-size: 14px; }
        .form-group textarea { resize: vertical; min-height: 80px; font-family: monospace; }
        .slider-group { margin-bottom: 15px; }
        .slider-group label { display: block; font-weight: 500; margin-bottom: 5px; color: #555; font-size: 14px; }
        .slider-group input[type="range"] { width: 100%; }
        .slider-value { display: inline-block; background: #f0f0f0; padding: 2px 8px; border-radius: 3px; font-size: 12px; margin-left: 10px; }
        button { width: 100%; padding: 12px; background: #667eea; color: white; border: none; border-radius: 4px; font-size: 16px; font-weight: 600; cursor: pointer; margin-top: 20px; }
        button:hover { background: #5568d3; }
        button:active { transform: scale(0.98); }
        .results { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .results h2 { font-size: 18px; margin-bottom: 15px; color: #333; }
        .graph-container { margin-bottom: 20px; }
        .graph-container img { max-width: 100%; height: auto; border-radius: 4px; }
        .stats { background: #f9f9f9; padding: 15px; border-radius: 4px; border-left: 4px solid #667eea; font-family: monospace; font-size: 13px; white-space: pre-wrap; }
        .loading { display: none; text-align: center; padding: 20px; }
        .loading.active { display: block; }
        .spinner { border: 3px solid #f3f3f3; border-top: 3px solid #667eea; border-radius: 50%; width: 30px; height: 30px; animation: spin 1s linear infinite; margin: 0 auto 10px; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .error { background: #fee; color: #c33; padding: 15px; border-radius: 4px; margin-bottom: 15px; border-left: 4px solid #c33; }
        .success { background: #efe; color: #3c3; padding: 15px; border-radius: 4px; margin-bottom: 15px; border-left: 4px solid #3c3; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎯 Task-Driven Scene Graph Explorer</h1>
            <p>Agglomerative Information Bottleneck clustering for task-driven scene understanding</p>
        </header>
        
        <div class="main">
            <div class="controls">
                <h2>⚙️ Controls</h2>
                <form id="clusterForm">
                    <div class="form-group">
                        <label for="scene">Scene</label>
                        <select id="scene" name="scene">
                            {% for choice in scene_choices %}
                            <option value="{{ choice }}">{{ choice }}</option>
                            {% endfor %}
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="tasks">Tasks (comma-separated)</label>
                        <textarea id="tasks" name="tasks" placeholder="pick up the mug, find a place to sit, open the door">pick up the mug, find a place to sit</textarea>
                    </div>
                    
                    <div class="slider-group">
                        <label for="tau">Compression τ: <span class="slider-value" id="tauValue">0.10</span></label>
                        <input type="range" id="tau" name="tau" min="0" max="1" step="0.05" value="0.1">
                    </div>
                    
                    <div class="slider-group">
                        <label for="alpha">Null task threshold α: <span class="slider-value" id="alphaValue">0.30</span></label>
                        <input type="range" id="alpha" name="alpha" min="0.1" max="0.9" step="0.05" value="0.3">
                    </div>
                    
                    <button type="submit">🚀 Run Clustering</button>
                </form>
            </div>
            
            <div class="results">
                <h2>📊 Results</h2>
                <div id="message"></div>
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p>Running clustering...</p>
                </div>
                <div id="resultsContent" style="display: none;">
                    <div class="graph-container">
                        <img id="graphImage" src="" alt="Scene graph">
                    </div>
                    <div class="stats" id="stats"></div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Update slider values
        document.getElementById('tau').addEventListener('input', (e) => {
            document.getElementById('tauValue').textContent = parseFloat(e.target.value).toFixed(2);
        });
        document.getElementById('alpha').addEventListener('input', (e) => {
            document.getElementById('alphaValue').textContent = parseFloat(e.target.value).toFixed(2);
        });
        
        // Handle form submission
        document.getElementById('clusterForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const scene = document.getElementById('scene').value;
            const tasks = document.getElementById('tasks').value.split(',').map(t => t.trim()).filter(t => t);
            const tau = parseFloat(document.getElementById('tau').value);
            const alpha = parseFloat(document.getElementById('alpha').value);
            
            if (tasks.length === 0) {
                showMessage('Please enter at least one task', 'error');
                return;
            }
            
            document.getElementById('loading').classList.add('active');
            document.getElementById('resultsContent').style.display = 'none';
            document.getElementById('message').innerHTML = '';
            
            try {
                const response = await fetch('/cluster', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ scene, tasks, tau, alpha })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    showMessage(data.error, 'error');
                } else {
                    document.getElementById('graphImage').src = data.graph;
                    document.getElementById('stats').textContent = data.stats;
                    document.getElementById('resultsContent').style.display = 'block';
                    showMessage('✓ Clustering complete!', 'success');
                }
            } catch (err) {
                showMessage('Error: ' + err.message, 'error');
            } finally {
                document.getElementById('loading').classList.remove('active');
            }
        });
        
        function showMessage(msg, type) {
            const msgDiv = document.getElementById('message');
            msgDiv.innerHTML = `<div class="${type}">${msg}</div>`;
        }
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    """Render main page."""
    return render_template_string(HTML_TEMPLATE, scene_choices=scene_choices)


@app.route('/cluster', methods=['POST'])
def cluster():
    """Run clustering and return results."""
    try:
        data = request.json
        scene_choice = data.get('scene')
        task_strings = data.get('tasks', [])
        tau = float(data.get('tau', 0.1))
        alpha = float(data.get('alpha', 0.3))
        
        if not scene_choice or not task_strings:
            return jsonify({'error': 'Missing scene or tasks'}), 400
        
        # Load scene
        sg = load_scene(scene_choice)
        
        # Embed nodes
        node_embs = embed_nodes(sg, clip_model, task_strings, alpha, device=device)
        
        # Embed tasks
        task_embs = embed_tasks(task_strings, clip_model, device=device)
        
        # Compute p(y|x) for all nodes
        p_y_given_x = {}
        for node_id, node_emb in node_embs.items():
            theta = compute_theta(node_emb, task_embs, alpha)
            p_y_given_x[node_id] = compute_p_y_given_x(theta, alpha, l=1)
        
        # Run AIB
        aib_clusters, _ = run_aib(sg, p_y_given_x, tau=tau, max_runtime_seconds=60.0)
        
        # Run baseline
        baseline_retained = run_threshold_baseline(node_embs, task_embs, threshold=alpha)
        
        # Render graph
        graph_img = render_graph_image(sg, p_y_given_x, aib_clusters)
        
        # Compute statistics
        n_nodes = sg.num_nodes()
        n_task_relevant = sum(1 for nid in p_y_given_x if p_y_given_x[nid][0] == 0.0)
        aib_retained = set()
        for cluster in aib_clusters:
            aib_retained.update(cluster)
        
        aib_compression = (n_nodes - len(aib_clusters)) / n_nodes if n_nodes > 0 else 0.0
        aib_coverage = sum(1 for nid in aib_retained if p_y_given_x[nid][0] == 0.0) / n_task_relevant if n_task_relevant > 0 else 0.0
        baseline_compression = (n_nodes - len(baseline_retained)) / n_nodes if n_nodes > 0 else 0.0
        
        stats = f"""Scene: {sg.scene_id}
Nodes: {n_nodes}
Task-relevant: {n_task_relevant}
Tasks: {', '.join(task_strings)}

AIB Results:
  Clusters: {len(aib_clusters)}
  Compression: {aib_compression:.1%}
  Coverage: {aib_coverage:.1%}
  Retained nodes: {len(aib_retained)}

Baseline (threshold={alpha:.2f}):
  Retained nodes: {len(baseline_retained)}
  Compression: {baseline_compression:.1%}

Parameters:
  τ (information loss threshold): {tau:.2f}
  α (null task threshold): {alpha:.2f}
  l (top-l tasks): 1"""
        
        return jsonify({
            'graph': graph_img,
            'stats': stats
        })
    
    except Exception as e:
        import traceback
        return jsonify({'error': f"Error: {str(e)}\n\n{traceback.format_exc()}"}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Starting Task-Driven Scene Graph Explorer")
    print("="*60)
    print("📱 Open http://127.0.0.1:5000 in your browser")
    print("="*60 + "\n")
    
    app.run(debug=False, host='127.0.0.1', port=5000)
