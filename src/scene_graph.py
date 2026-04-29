"""
SceneGraph class for loading and querying 3RScan scene data.

Handles the semseg.v2.json format from 3RScan dataset.
Implements v3 specification with background filtering and synthetic generation.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional


# Background labels to filter out (structural/non-interactive elements)
BACKGROUND_LABELS = {
    "wall", "floor", "ceiling", "floor mat", "unknown",
    "object", "otherstructure", "otherfurniture", "otherprop"
}


class SceneGraph:
    """
    Scene graph representation for 3RScan data.
    
    The 3RScan semseg.v2.json format contains:
    - segGroups: array of objects with objectId, label, obb (oriented bounding box)
    - No explicit edges - we construct spatial relationships based on proximity
    
    Background labels are filtered out before building the scene graph.
    """
    
    def __init__(self, json_path: str, distance_threshold: float = 1.5):
        """
        Load a scene graph from 3RScan semseg.v2.json file.
        
        Args:
            json_path: Path to semseg.v2.json file
            distance_threshold: Maximum distance (meters) for spatial adjacency
        """
        self.json_path = Path(json_path)
        self.scene_id = self.json_path.parent.name
        
        # Load JSON data
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Parse nodes from segGroups with background filtering
        self._nodes = []
        self._node_dict = {}
        n_filtered = 0
        
        if 'segGroups' not in data:
            raise ValueError(f"Invalid 3RScan format: 'segGroups' key not found in {json_path}")
        
        for seg_group in data['segGroups']:
            label = seg_group.get('label', 'unknown').strip().lower()
            
            # Filter background labels
            if label in BACKGROUND_LABELS:
                n_filtered += 1
                continue
            
            node = self._parse_node(seg_group, label)
            self._nodes.append(node)
            self._node_dict[node['id']] = node
        
        print(f"[Parser] Scene {self.scene_id}: {len(self._nodes)} objects kept, {n_filtered} background filtered")
        
        # Build spatial adjacency based on Euclidean distance
        self._adjacency = {}
        self._build_spatial_adjacency(distance_threshold=distance_threshold)
    
    def _parse_node(self, seg_group: dict, label: str) -> dict:
        """
        Parse a segGroup into a node dictionary.
        
        Args:
            seg_group: Raw segGroup dict from JSON
            label: Already-processed label (lowercase, stripped)
            
        Returns:
            Node dict with standardized keys: id, label, position, extent
        """
        # Extract centroid and extents from OBB
        obb = seg_group.get('obb', {})
        centroid = obb.get('centroid', [0.0, 0.0, 0.0])
        axes_lengths = obb.get('axesLengths', [0.0, 0.0, 0.0])
        
        node = {
            'id': str(seg_group.get('objectId', seg_group.get('id'))),
            'label': label,
            'position': centroid,  # [x, y, z] in meters
            'extent': axes_lengths,  # [lx, ly, lz] in meters
        }
        
        return node
    
    def _build_spatial_adjacency(self, distance_threshold: float = 1.5):
        """
        Build adjacency list based on Euclidean distance between centroids.
        
        Args:
            distance_threshold: Maximum distance (meters) to consider nodes adjacent
        """
        self._adjacency = {node['id']: [] for node in self._nodes}
        
        # Compute pairwise distances
        for i, node_i in enumerate(self._nodes):
            pos_i = np.array(node_i['position'])
            
            for j, node_j in enumerate(self._nodes):
                if i >= j:  # Skip self and already computed pairs
                    continue
                
                pos_j = np.array(node_j['position'])
                distance = np.linalg.norm(pos_i - pos_j)
                
                if distance <= distance_threshold:
                    # Add bidirectional edge
                    self._adjacency[node_i['id']].append(node_j['id'])
                    self._adjacency[node_j['id']].append(node_i['id'])
    
    @property
    def nodes(self) -> List[dict]:
        """Get all nodes in the scene graph."""
        return self._nodes
    
    @property
    def edges(self) -> List[Tuple[str, str, str]]:
        """
        Get all edges as (node_id_a, node_id_b, relation_label) tuples.
        
        For 3RScan, relation is always 'near' (spatial proximity).
        """
        edges = []
        seen = set()
        
        for node_id, neighbors in self._adjacency.items():
            for neighbor_id in neighbors:
                # Avoid duplicates (undirected graph)
                edge_key = tuple(sorted([node_id, neighbor_id]))
                if edge_key not in seen:
                    edges.append((node_id, neighbor_id, 'near'))
                    seen.add(edge_key)
        
        return edges
    
    def get_neighbors(self, node_id: str) -> List[str]:
        """
        Get neighbor node IDs for a given node.
        
        Args:
            node_id: ID of the node
            
        Returns:
            List of neighbor node IDs
        """
        return self._adjacency.get(str(node_id), [])
    
    def get_node(self, node_id: str) -> Optional[dict]:
        """
        Get node data by ID.
        
        Args:
            node_id: ID of the node
            
        Returns:
            Node dict or None if not found
        """
        return self._node_dict.get(str(node_id))
    
    def summary(self) -> str:
        """
        Generate a summary string of the scene graph.
        
        Returns:
            Summary string with node count, edge count, and unique labels
        """
        num_nodes = len(self._nodes)
        num_edges = len(self.edges)
        
        # Count unique labels
        labels = [node['label'] for node in self._nodes]
        unique_labels = sorted(set(labels))
        label_counts = {label: labels.count(label) for label in unique_labels}
        
        # Compute degree statistics
        degrees = [len(self._adjacency.get(node['id'], [])) for node in self._nodes]
        avg_degree = sum(degrees) / len(degrees) if degrees else 0
        min_degree = min(degrees) if degrees else 0
        max_degree = max(degrees) if degrees else 0
        
        summary = f"Scene: {self.scene_id}\n"
        summary += f"Nodes: {num_nodes}\n"
        summary += f"Edges: {num_edges}\n"
        summary += f"Unique labels: {len(unique_labels)}\n"
        summary += f"Avg degree: {avg_degree:.1f}, Min/Max: {min_degree}/{max_degree}\n"
        summary += "\nLabel distribution:\n"
        for label, count in sorted(label_counts.items(), key=lambda x: -x[1])[:10]:
            summary += f"  {label}: {count}\n"
        
        return summary
    
    def num_nodes(self) -> int:
        """Get total number of nodes."""
        return len(self._nodes)


    @staticmethod
    def make_synthetic(n_nodes: int = 30, seed: int = 42) -> "SceneGraph":
        """
        Generate a synthetic scene for development without needing the dataset.
        All Phases 2–4 can run on synthetic scenes before real data is ready.
        
        Args:
            n_nodes: Number of nodes to generate
            seed: Random seed for reproducibility
            
        Returns:
            SceneGraph instance with synthetic data
        """
        rng = np.random.default_rng(seed)
        LABELS = [
            "chair", "table", "cup", "mug", "sofa", "bed", "desk", "lamp",
            "book", "bottle", "plant", "monitor", "keyboard", "door", "window",
            "shelf", "box", "backpack", "phone", "remote"
        ]
        
        nodes = []
        for i in range(n_nodes):
            nodes.append({
                "id": f"node_{i:03d}",
                "label": rng.choice(LABELS),
                "position": rng.uniform([0, 0, 0], [10, 10, 3]).tolist(),
                "extent": rng.uniform([0.2, 0.2, 0.2], [1.0, 1.0, 1.5]).tolist()
            })
        
        # Build a SceneGraph without a file
        sg = SceneGraph.__new__(SceneGraph)
        sg.scene_id = f"synthetic_n{n_nodes}_s{seed}"
        sg.json_path = None
        sg._nodes = nodes
        sg._node_dict = {n["id"]: n for n in nodes}
        
        # Build edges with proximity threshold
        sg._adjacency = {}
        for node in nodes:
            sg._adjacency[node["id"]] = []
        
        positions = np.array([n["position"] for n in nodes])
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist <= 2.5:  # Synthetic threshold
                    sg._adjacency[nodes[i]["id"]].append(nodes[j]["id"])
                    sg._adjacency[nodes[j]["id"]].append(nodes[i]["id"])
        
        # Guarantee no isolated nodes: connect to nearest neighbor if isolated
        for i, node in enumerate(nodes):
            if len(sg._adjacency[node["id"]]) == 0:
                dists = np.linalg.norm(positions - positions[i], axis=1)
                dists[i] = np.inf
                nearest_idx = int(np.argmin(dists))
                nearest_id = nodes[nearest_idx]["id"]
                sg._adjacency[node["id"]].append(nearest_id)
                sg._adjacency[nearest_id].append(node["id"])
        
        return sg
