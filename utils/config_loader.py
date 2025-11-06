# utils/config_loader.py
import yaml
import os
from datetime import datetime

class ConfigLoader:
    def __init__(self, config_path='config.yaml'):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self):
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            return self.get_default_config()
    
    def get_default_config(self):
        return {
            'camera': {
                'source': 0,
                'width': 1280,
                'height': 720,
                'fps': 30,
                'frame_skip': 5
            },
            'tracking': {
                'model_path': 'models/yolov8n.pt',
                'confidence_threshold': 0.5,
                'max_track_length': 30,
                'dwell_time_threshold': 300
            },
            'inventory': {
                'db_path': 'inventory.db',
                'restock_threshold': 20,
                'shelf_capacity': 50,
                'detection_confidence': 0.7
            },
            'recommendation': {
                'model_path': 'models/recommender.h5',
                'top_k_recommendations': 5,
                'sequence_length': 10,
                'embedding_dim': 64
            },
            'behavior_analysis': {
                'session_timeout_minutes': 30,
                'min_sessions_for_segmentation': 3,
                'anomaly_detection_threshold': 2.5
            },
            'api': {
                'host': '0.0.0.0',
                'port': 8000,
                'debug': False,
                'workers': 4
            },
            'dashboard': {
                'host': '0.0.0.0',
                'port': 5000,
                'debug': True
            },
            'storage': {
                'database_path': 'retail_data.db',
                'backup_interval_hours': 24,
                'max_log_files': 10
            }
        }
    
    def get(self, key, default=None):
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, {})
        return value if value != {} else default
    
    def update(self, key, value):
        keys = key.split('.')
        config_ref = self.config
        
        for k in keys[:-1]:
            if k not in config_ref:
                config_ref[k] = {}
            config_ref = config_ref[k]
        
        config_ref[keys[-1]] = value
    
    def save(self, config_path=None):
        if config_path is None:
            config_path = self.config_path
        
        with open(config_path, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False)
    
    def validate_config(self):
        required_sections = ['camera', 'tracking', 'inventory', 'api']
        
        for section in required_sections:
            if section not in self.config:
                return False, f"Missing required section: {section}"
        
        return True, "Configuration is valid"