# core/layout_optimizer.py
import tensorflow as tf
import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import pandas as pd

class LayoutOptimizer:
    def __init__(self):
        self.customer_paths = []
        self.product_affinities = {}
        self.shelf_locations = {}
        self.heatmap_data = None
        
    def analyze_customer_flow(self, customer_paths, store_layout):
        self.customer_paths = customer_paths
        
        flow_analysis = {
            'bottlenecks': self._find_bottlenecks(customer_paths, store_layout),
            'popular_routes': self._find_popular_routes(customer_paths),
            'dead_zones': self._find_dead_zones(customer_paths, store_layout),
            'flow_efficiency': self._calculate_flow_efficiency(customer_paths)
        }
        
        return flow_analysis
    
    def _find_bottlenecks(self, paths, layout):
        position_counts = defaultdict(int)
        for path in paths:
            for x, y, _ in path:
                grid_x, grid_y = int(x // 10), int(y // 10)
                position_counts[(grid_x, grid_y)] += 1
        
        bottlenecks = []
        for (x, y), count in position_counts.items():
            if count > len(paths) * 0.3:
                bottlenecks.append({
                    'location': (x * 10, y * 10),
                    'congestion_level': count / len(paths),
                    'affected_customers': count
                })
        
        return sorted(bottlenecks, key=lambda x: x['congestion_level'], reverse=True)
    
    def _find_popular_routes(self, paths):
        if len(paths) < 2:
            return []
        
        route_similarities = []
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                similarity = self._path_similarity(paths[i], paths[j])
                route_similarities.append((i, j, similarity))
        
        route_similarities.sort(key=lambda x: x[2], reverse=True)
        
        popular_routes = []
        for i, j, sim in route_similarities[:5]:
            if sim > 0.7:
                popular_routes.append({
                    'path_1': paths[i],
                    'path_2': paths[j],
                    'similarity': sim
                })
        
        return popular_routes
    
    def _path_similarity(self, path1, path2):
        if len(path1) == 0 or len(path2) == 0:
            return 0.0
        
        min_len = min(len(path1), len(path2))
        path1_pts = np.array([(p[0], p[1]) for p in path1[:min_len]])
        path2_pts = np.array([(p[0], p[1]) for p in path2[:min_len]])
        
        distances = np.linalg.norm(path1_pts - path2_pts, axis=1)
        avg_distance = np.mean(distances)
        
        max_possible_distance = np.sqrt(640**2 + 480**2)
        similarity = 1.0 - (avg_distance / max_possible_distance)
        
        return max(0.0, similarity)
    
    def _find_dead_zones(self, paths, layout):
        all_positions = []
        for path in paths:
            for x, y, _ in path:
                all_positions.append((x, y))
        
        if not all_positions:
            return []
        
        position_array = np.array(all_positions)
        kmeans = KMeans(n_clusters=10, random_state=42)
        clusters = kmeans.fit_predict(position_array)
        
        cluster_counts = np.bincount(clusters)
        dead_zones = []
        
        layout_area = layout.shape[0] * layout.shape[1] if hasattr(layout, 'shape') else 10000
        
        for i, count in enumerate(cluster_counts):
            if count < len(paths) * 0.05:
                centroid = kmeans.cluster_centers_[i]
                dead_zones.append({
                    'location': (centroid[0], centroid[1]),
                    'visit_frequency': count / len(paths),
                    'area_utilization': count / layout_area
                })
        
        return dead_zones
    
    def _calculate_flow_efficiency(self, paths):
        if not paths:
            return 0.0
        
        efficiencies = []
        for path in paths:
            if len(path) > 1:
                total_distance = 0
                for i in range(1, len(path)):
                    x1, y1, _ = path[i-1]
                    x2, y2, _ = path[i]
                    distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    total_distance += distance
                
                straight_line_distance = np.sqrt(
                    (path[-1][0]-path[0][0])**2 + (path[-1][1]-path[0][1])**2
                )
                
                if straight_line_distance > 0:
                    efficiency = straight_line_distance / total_distance
                    efficiencies.append(efficiency)
        
        return np.mean(efficiencies) if efficiencies else 0.0
    
    def optimize_product_placement(self, product_affinities, current_layout, customer_heatmap):
        self.product_affinities = product_affinities
        self.heatmap_data = customer_heatmap
        
        G = self._build_affinity_graph(product_affinities)
        
        try:
            communities = nx.community.louvain_communities(G)
        except:
            communities = [set(G.nodes())]
        
        optimized_layout = current_layout.copy()
        
        shelf_centroids = self._calculate_shelf_centroids()
        
        community_placements = {}
        for i, community in enumerate(communities):
            community_products = list(community)
            
            if shelf_centroids:
                centroid_idx = i % len(shelf_centroids)
                best_shelf = shelf_centroids[centroid_idx]
                
                for product in community_products:
                    community_placements[product] = best_shelf
        
        for product, shelf_loc in community_placements.items():
            x, y = int(shelf_loc[0]), int(shelf_loc[1])
            if 0 <= x < optimized_layout.shape[1] and 0 <= y < optimized_layout.shape[0]:
                optimized_layout[y, x] = product
        
        return optimized_layout
    
    def _build_affinity_graph(self, affinities):
        G = nx.Graph()
        
        for product1, relationships in affinities.items():
            for product2, affinity in relationships.items():
                if affinity > 0.3:
                    G.add_edge(product1, product2, weight=affinity)
        
        return G
    
    def _calculate_shelf_centroids(self):
        if self.heatmap_data is None:
            return [(320, 240)]
        
        y_coords, x_coords = np.where(self.heatmap_data > np.percentile(self.heatmap_data, 70))
        
        if len(x_coords) == 0:
            return [(320, 240)]
        
        coords = np.column_stack((x_coords, y_coords))
        kmeans = KMeans(n_clusters=min(5, len(coords)), random_state=42)
        kmeans.fit(coords)
        
        return kmeans.cluster_centers_.tolist()
    
    def generate_layout_recommendations(self, flow_analysis, current_layout):
        recommendations = []
        
        for bottleneck in flow_analysis.get('bottlenecks', []):
            recommendations.append({
                'type': 'bottleneck_relief',
                'location': bottleneck['location'],
                'suggestion': f"Widen aisle or reposition displays near {bottleneck['location']}",
                'priority': 'high',
                'expected_impact': 'Reduce congestion by 30-50%'
            })
        
        for dead_zone in flow_analysis.get('dead_zones', []):
            recommendations.append({
                'type': 'dead_zone_activation',
                'location': dead_zone['location'],
                'suggestion': "Place promotional items or high-demand products in this area",
                'priority': 'medium',
                'expected_impact': 'Increase area utilization by 20-40%'
            })
        
        flow_efficiency = flow_analysis.get('flow_efficiency', 0)
        if flow_efficiency < 0.6:
            recommendations.append({
                'type': 'flow_optimization',
                'location': 'store_wide',
                'suggestion': "Reorganize product categories to create more direct paths",
                'priority': 'medium',
                'expected_impact': f'Improve flow efficiency from {flow_efficiency:.1%} to target 70%'
            })
        
        return recommendations
    
    def simulate_layout_change(self, new_layout, customer_paths):
        original_efficiency = self._calculate_flow_efficiency(customer_paths)
        
        simulated_paths = self._simulate_paths_with_new_layout(customer_paths, new_layout)
        
        new_efficiency = self._calculate_flow_efficiency(simulated_paths)
        
        improvement = new_efficiency - original_efficiency
        
        return {
            'original_efficiency': original_efficiency,
            'new_efficiency': new_efficiency,
            'improvement': improvement,
            'improvement_percentage': (improvement / original_efficiency * 100) if original_efficiency > 0 else 0
        }
    
    def _simulate_paths_with_new_layout(self, original_paths, new_layout):
        simulated_paths = []
        
        for path in original_paths:
            simulated_path = []
            for x, y, frame in path:
                grid_x, grid_y = int(x // 20), int(y // 20)
                
                if (0 <= grid_x < new_layout.shape[1] and 
                    0 <= grid_y < new_layout.shape[0]):
                    
                    product_at_location = new_layout[grid_y, grid_x]
                    
                    dwell_probability = self._get_dwell_probability(product_at_location)
                    
                    if np.random.random() < dwell_probability:
                        for _ in range(3):
                            simulated_path.append((x, y, frame))
                
                simulated_path.append((x, y, frame))
            
            simulated_paths.append(simulated_path)
        
        return simulated_paths
    
    def _get_dwell_probability(self, product_category):
        dwell_probabilities = {
            'beverages': 0.3,
            'snacks': 0.4,
            'dairy': 0.2,
            'produce': 0.5,
            'meat': 0.6,
            'bakery': 0.7,
            'frozen': 0.3,
            'household': 0.2
        }
        
        return dwell_probabilities.get(product_category, 0.2)