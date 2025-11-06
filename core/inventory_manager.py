# core/inventory_manager.py
import tensorflow as tf
import cv2
import numpy as np
from ultralytics import YOLO
import json
import sqlite3
from datetime import datetime, timedelta
import pandas as pd

class ProductDetector:
    def __init__(self, model_path='models/product_detector.pt'):
        self.model = YOLO(model_path)
        self.product_categories = {
            0: 'beverages', 1: 'snacks', 2: 'dairy', 3: 'produce', 
            4: 'meat', 5: 'bakery', 6: 'frozen', 7: 'household'
        }
    
    def detect_products(self, image):
        results = self.model(image)
        detections = []
        
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                detection = {
                    'bbox': box.tolist(),
                    'confidence': float(conf),
                    'category_id': int(cls_id),
                    'category_name': self.product_categories.get(cls_id, 'unknown')
                }
                detections.append(detection)
        
        return detections

class InventoryManager:
    def __init__(self, db_path='inventory.db'):
        self.detector = ProductDetector()
        self.db_path = db_path
        self._init_database()
        
        self.shelf_configurations = {}
        self.restock_thresholds = {}
        self.sales_data = defaultdict(list)
    
    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                shelf_location TEXT,
                current_stock INTEGER DEFAULT 0,
                max_capacity INTEGER DEFAULT 100,
                restock_threshold INTEGER DEFAULT 20,
                last_restocked DATETIME,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS inventory_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id INTEGER,
                action TEXT,
                quantity INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (product_id) REFERENCES products (id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS shelf_analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                shelf_id TEXT,
                product_id INTEGER,
                occupancy_rate REAL,
                detection_confidence REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (product_id) REFERENCES products (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def analyze_shelf_image(self, image, shelf_id):
        detections = self.detector.detect_products(image)
        
        shelf_analysis = {
            'shelf_id': shelf_id,
            'timestamp': datetime.now().isoformat(),
            'total_detections': len(detections),
            'product_distribution': defaultdict(int),
            'occupancy_rate': 0.0,
            'restock_alerts': []
        }
        
        for detection in detections:
            category = detection['category_name']
            shelf_analysis['product_distribution'][category] += 1
        
        total_capacity = self._get_shelf_capacity(shelf_id)
        if total_capacity > 0:
            shelf_analysis['occupancy_rate'] = len(detections) / total_capacity
        
        self._log_shelf_analytics(shelf_id, shelf_analysis)
        self._check_restock_needs(shelf_id, shelf_analysis)
        
        return shelf_analysis
    
    def _get_shelf_capacity(self, shelf_id):
        return 50
    
    def _log_shelf_analytics(self, shelf_id, analysis):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for product_category, count in analysis['product_distribution'].items():
            cursor.execute('''
                INSERT INTO shelf_analytics (shelf_id, product_id, occupancy_rate, detection_confidence)
                VALUES (?, ?, ?, ?)
            ''', (shelf_id, self._get_product_id(product_category), 
                  analysis['occupancy_rate'], 0.8))
        
        conn.commit()
        conn.close()
    
    def _get_product_id(self, category):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id FROM products WHERE category = ? LIMIT 1', (category,))
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else 1
    
    def _check_restock_needs(self, shelf_id, analysis):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for category, count in analysis['product_distribution'].items():
            cursor.execute('''
                SELECT current_stock, restock_threshold FROM products 
                WHERE category = ? LIMIT 1
            ''', (category,))
            result = cursor.fetchone()
            
            if result and count < result[1]:
                analysis['restock_alerts'].append({
                    'category': category,
                    'current_count': count,
                    'threshold': result[1],
                    'urgency': 'high' if count < result[1] * 0.5 else 'medium'
                })
        
        conn.close()
    
    def update_inventory(self, product_id, quantity_change, action='sale'):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT current_stock FROM products WHERE id = ?', (product_id,))
        current_stock = cursor.fetchone()[0]
        
        new_stock = max(0, current_stock + quantity_change)
        
        cursor.execute('''
            UPDATE products SET current_stock = ? WHERE id = ?
        ''', (new_stock, product_id))
        
        cursor.execute('''
            INSERT INTO inventory_logs (product_id, action, quantity)
            VALUES (?, ?, ?)
        ''', (product_id, action, quantity_change))
        
        conn.commit()
        conn.close()
        
        return new_stock
    
    def get_inventory_status(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT p.name, p.category, p.current_stock, p.restock_threshold,
                   p.max_capacity, p.last_restocked,
                   COUNT(CASE WHEN il.action = 'sale' THEN 1 END) as sales_count
            FROM products p
            LEFT JOIN inventory_logs il ON p.id = il.product_id
            WHERE il.timestamp >= datetime('now', '-1 day')
            GROUP BY p.id
        ''')
        
        inventory_data = []
        for row in cursor.fetchall():
            item = {
                'name': row[0],
                'category': row[1],
                'current_stock': row[2],
                'restock_threshold': row[3],
                'max_capacity': row[4],
                'last_restocked': row[5],
                'daily_sales': row[6],
                'needs_restock': row[2] < row[3],
                'stock_ratio': row[2] / row[4] if row[4] > 0 else 0
            }
            inventory_data.append(item)
        
        conn.close()
        return inventory_data
    
    def predict_restock_needs(self, days_ahead=7):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT p.id, p.name, p.category, p.current_stock,
                   AVG(CASE WHEN il.action = 'sale' THEN il.quantity ELSE 0 END) as avg_daily_sales
            FROM products p
            LEFT JOIN inventory_logs il ON p.id = il.product_id
            WHERE il.timestamp >= datetime('now', '-30 days')
            GROUP BY p.id
        ''')
        
        predictions = []
        for row in cursor.fetchall():
            product_id, name, category, current_stock, avg_daily_sales = row
            avg_daily_sales = avg_daily_sales or 1
            
            days_until_empty = current_stock / avg_daily_sales if avg_daily_sales > 0 else float('inf')
            
            prediction = {
                'product_id': product_id,
                'name': name,
                'category': category,
                'current_stock': current_stock,
                'avg_daily_sales': avg_daily_sales,
                'days_until_empty': days_until_empty,
                'restock_urgency': 'critical' if days_until_empty < 2 else 
                                 'high' if days_until_empty < 5 else 
                                 'medium' if days_until_empty < 10 else 'low'
            }
            predictions.append(prediction)
        
        conn.close()
        return predictions
    
    def optimize_shelf_layout(self, customer_heatmap, current_layout):
        heatmap_flat = customer_heatmap.flatten()
        high_traffic_indices = np.argsort(heatmap_flat)[-10:]
        
        optimized_layout = current_layout.copy()
        
        high_demand_products = self._get_high_demand_products()
        
        for i, idx in enumerate(high_traffic_indices[:len(high_demand_products)]):
            row = idx // customer_heatmap.shape[1]
            col = idx % customer_heatmap.shape[1]
            
            if i < len(high_demand_products):
                product = high_demand_products[i]
                optimized_layout[row, col] = product['category']
        
        return optimized_layout
    
    def _get_high_demand_products(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT p.category, COUNT(il.id) as sales_count
            FROM products p
            JOIN inventory_logs il ON p.id = il.product_id
            WHERE il.timestamp >= datetime('now', '-7 days')
            GROUP BY p.category
            ORDER BY sales_count DESC
            LIMIT 5
        ''')
        
        high_demand = []
        for row in cursor.fetchall():
            high_demand.append({'category': row[0], 'sales_count': row[1]})
        
        conn.close()
        return high_demand