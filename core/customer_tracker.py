# core/customer_tracker.py
import tensorflow as tf
import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
from collections import defaultdict, deque
import torch

class CustomerTracker:
    def __init__(self, model_path='models/yolov8n.pt'):
        self.model = YOLO(model_path)
        self.track_history = defaultdict(lambda: deque(maxlen=30))
        self.customer_data = defaultdict(dict)
        self.current_tracks = {}
        
        self.areas_of_interest = {}
        self.dwell_times = defaultdict(list)
        self.movement_patterns = defaultdict(list)
        
    def process_frame(self, frame, frame_count):
        results = self.model.track(frame, persist=True, classes=[0])
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confidences = results[0].boxes.conf.cpu().tolist()
            
            for box, track_id, confidence in zip(boxes, track_ids, confidences):
                x, y, w, h = box
                
                self.track_history[track_id].append((float(x), float(y)))
                
                self.current_tracks[track_id] = {
                    'bbox': [float(x), float(y), float(w), float(h)],
                    'confidence': float(confidence),
                    'frame_count': frame_count
                }
                
                self._update_customer_behavior(track_id, x, y, frame_count)
        
        return self._analyze_customer_behavior(frame)
    
    def _update_customer_behavior(self, track_id, x, y, frame_count):
        if track_id not in self.customer_data:
            self.customer_data[track_id] = {
                'first_seen': frame_count,
                'last_seen': frame_count,
                'path': [],
                'area_visits': defaultdict(list),
                'total_dwell_time': 0
            }
        
        self.customer_data[track_id]['last_seen'] = frame_count
        self.customer_data[track_id]['path'].append((x, y, frame_count))
        
        current_area = self._get_current_area(x, y)
        if current_area:
            if track_id in self.customer_data:
                if not self.customer_data[track_id]['area_visits'][current_area]:
                    self.customer_data[track_id]['area_visits'][current_area].append(frame_count)
                else:
                    last_visit = self.customer_data[track_id]['area_visits'][current_area][-1]
                    if frame_count - last_visit > 30:
                        self.customer_data[track_id]['area_visits'][current_area].append(frame_count)
    
    def _get_current_area(self, x, y):
        for area_name, area_coords in self.areas_of_interest.items():
            if self._point_in_polygon(x, y, area_coords):
                return area_name
        return None
    
    def _point_in_polygon(self, x, y, poly):
        n = len(poly)
        inside = False
        p1x, p1y = poly[0]
        for i in range(n + 1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside
    
    def _analyze_customer_behavior(self, frame):
        analysis = {
            'customer_count': len(self.current_tracks),
            'active_tracks': list(self.current_tracks.keys()),
            'dwell_times': {},
            'movement_analysis': {},
            'heatmap_data': self._generate_heatmap()
        }
        
        for track_id, data in self.customer_data.items():
            dwell_time = data['last_seen'] - data['first_seen']
            analysis['dwell_times'][track_id] = dwell_time
            
            if len(data['path']) > 1:
                path_array = np.array([(p[0], p[1]) for p in data['path']])
                total_distance = np.sum(np.linalg.norm(np.diff(path_array, axis=0), axis=1))
                analysis['movement_analysis'][track_id] = {
                    'total_distance': total_distance,
                    'avg_speed': total_distance / len(data['path']),
                    'areas_visited': list(data['area_visits'].keys())
                }
        
        return analysis
    
    def _generate_heatmap(self):
        heatmap_data = np.zeros((480, 640))
        
        for track_id, track_points in self.track_history.items():
            for x, y in track_points:
                x_idx = min(int(x), 639)
                y_idx = min(int(y), 479)
                heatmap_data[y_idx, x_idx] += 1
        
        return heatmap_data
    
    def define_areas_of_interest(self, areas_dict):
        self.areas_of_interest = areas_dict
    
    def get_customer_insights(self):
        insights = {
            'total_customers_tracked': len(self.customer_data),
            'average_dwell_time': np.mean([data['last_seen'] - data['first_seen'] 
                                         for data in self.customer_data.values()]),
            'popular_areas': self._get_popular_areas(),
            'customer_paths': self._analyze_customer_paths(),
            'peak_traffic_times': self._analyze_traffic_patterns()
        }
        return insights
    
    def _get_popular_areas(self):
        area_counts = defaultdict(int)
        for data in self.customer_data.values():
            for area in data['area_visits']:
                area_counts[area] += 1
        return dict(sorted(area_counts.items(), key=lambda x: x[1], reverse=True))
    
    def _analyze_customer_paths(self):
        paths = []
        for track_id, data in self.customer_data.items():
            if len(data['path']) > 10:
                path = {
                    'track_id': track_id,
                    'length': len(data['path']),
                    'areas_visited': list(data['area_visits'].keys()),
                    'duration': data['last_seen'] - data['first_seen']
                }
                paths.append(path)
        return paths
    
    def _analyze_traffic_patterns(self):
        frame_counts = [data['first_seen'] for data in self.customer_data.values()]
        hist, bins = np.histogram(frame_counts, bins=10)
        peak_bin = bins[np.argmax(hist)]
        return {'peak_frame': peak_bin, 'customer_distribution': hist.tolist()}