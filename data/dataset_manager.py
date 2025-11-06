# data/dataset_manager.py
import pandas as pd
import numpy as np
import json
import sqlite3
from datetime import datetime, timedelta
import os
from collections import defaultdict

class DatasetManager:
    def __init__(self, db_path='retail_data.db'):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS customer_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT,
                timestamp DATETIME,
                position_x REAL,
                position_y REAL,
                dwell_time REAL,
                area_visited TEXT,
                frame_count INTEGER
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS product_detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                shelf_id TEXT,
                timestamp DATETIME,
                product_category TEXT,
                confidence REAL,
                bbox TEXT,
                occupancy_rate REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sales_transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT,
                customer_id TEXT,
                timestamp DATETIME,
                total_amount REAL,
                product_count INTEGER,
                store_location TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS customer_behavior (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT,
                session_id TEXT,
                timestamp DATETIME,
                dwell_time REAL,
                products_viewed INTEGER,
                products_purchased INTEGER,
                total_value REAL,
                aisles_visited INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_customer_tracking_data(self, tracking_data):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for track_id, data in tracking_data.items():
            if 'path' in data and data['path']:
                for x, y, frame in data['path'][-1:]:
                    cursor.execute('''
                        INSERT INTO customer_tracking 
                        (customer_id, timestamp, position_x, position_y, dwell_time, area_visited, frame_count)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (str(track_id), datetime.now(), x, y, 
                          data.get('last_seen', 0) - data.get('first_seen', 0),
                          ','.join(data.get('area_visits', {}).keys()), frame))
        
        conn.commit()
        conn.close()
    
    def save_shelf_analysis_data(self, shelf_data):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for category, count in shelf_data.get('product_distribution', {}).items():
            cursor.execute('''
                INSERT INTO product_detections 
                (shelf_id, timestamp, product_category, confidence, bbox, occupancy_rate)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (shelf_data.get('shelf_id'), datetime.now(), category, 0.8, '{}', 
                  shelf_data.get('occupancy_rate', 0)))
        
        conn.commit()
        conn.close()
    
    def save_sales_data(self, transaction_data):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO sales_transactions 
            (transaction_id, customer_id, timestamp, total_amount, product_count, store_location)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (transaction_data.get('transaction_id'),
              transaction_data.get('customer_id'),
              transaction_data.get('timestamp'),
              transaction_data.get('total_amount', 0),
              len(transaction_data.get('products', [])),
              transaction_data.get('store_location', 'unknown')))
        
        conn.commit()
        conn.close()
    
    def get_customer_tracking_history(self, customer_id, hours_back=24):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        cursor.execute('''
            SELECT timestamp, position_x, position_y, dwell_time, area_visited
            FROM customer_tracking
            WHERE customer_id = ? AND timestamp >= ?
            ORDER BY timestamp
        ''', (customer_id, cutoff_time))
        
        history = []
        for row in cursor.fetchall():
            history.append({
                'timestamp': row[0],
                'position': (row[1], row[2]),
                'dwell_time': row[3],
                'areas_visited': row[4].split(',') if row[4] else []
            })
        
        conn.close()
        return history
    
    def get_shelf_occupancy_trends(self, shelf_id, days_back=7):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_time = datetime.now() - timedelta(days=days_back)
        
        cursor.execute('''
            SELECT timestamp, product_category, occupancy_rate
            FROM product_detections
            WHERE shelf_id = ? AND timestamp >= ?
            ORDER BY timestamp
        ''', (shelf_id, cutoff_time))
        
        trends = defaultdict(list)
        for row in cursor.fetchall():
            trends[row[1]].append({
                'timestamp': row[0],
                'occupancy_rate': row[2]
            })
        
        conn.close()
        return dict(trends)
    
    def get_sales_analytics(self, start_date, end_date):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                DATE(timestamp) as sale_date,
                COUNT(*) as transaction_count,
                SUM(total_amount) as daily_revenue,
                AVG(total_amount) as avg_transaction_value,
                COUNT(DISTINCT customer_id) as unique_customers
            FROM sales_transactions
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY DATE(timestamp)
            ORDER BY sale_date
        ''', (start_date, end_date))
        
        daily_analytics = []
        for row in cursor.fetchall():
            daily_analytics.append({
                'date': row[0],
                'transaction_count': row[1],
                'daily_revenue': row[2],
                'avg_transaction_value': row[3],
                'unique_customers': row[4]
            })
        
        cursor.execute('''
            SELECT 
                strftime('%H', timestamp) as hour,
                COUNT(*) as transaction_count,
                AVG(total_amount) as avg_value
            FROM sales_transactions
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY strftime('%H', timestamp)
            ORDER BY hour
        ''', (start_date, end_date))
        
        hourly_patterns = []
        for row in cursor.fetchall():
            hourly_patterns.append({
                'hour': int(row[0]),
                'transaction_count': row[1],
                'avg_transaction_value': row[2]
            })
        
        conn.close()
        
        return {
            'daily_analytics': daily_analytics,
            'hourly_patterns': hourly_patterns
        }
    
    def export_data_for_training(self, output_dir, data_types=None):
        if data_types is None:
            data_types = ['customer_tracking', 'product_detections', 'sales']
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        export_info = {}
        
        conn = sqlite3.connect(self.db_path)
        
        if 'customer_tracking' in data_types:
            df_tracking = pd.read_sql_query('SELECT * FROM customer_tracking', conn)
            tracking_file = os.path.join(output_dir, 'customer_tracking.csv')
            df_tracking.to_csv(tracking_file, index=False)
            export_info['customer_tracking'] = tracking_file
        
        if 'product_detections' in data_types:
            df_products = pd.read_sql_query('SELECT * FROM product_detections', conn)
            products_file = os.path.join(output_dir, 'product_detections.csv')
            df_products.to_csv(products_file, index=False)
            export_info['product_detections'] = products_file
        
        if 'sales' in data_types:
            df_sales = pd.read_sql_query('SELECT * FROM sales_transactions', conn)
            sales_file = os.path.join(output_dir, 'sales_transactions.csv')
            df_sales.to_csv(sales_file, index=False)
            export_info['sales'] = sales_file
        
        conn.close()
        
        metadata = {
            'export_timestamp': datetime.now().isoformat(),
            'data_types_exported': data_types,
            'file_locations': export_info
        }
        
        with open(os.path.join(output_dir, 'export_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return export_info
    
    def get_database_stats(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        tables = ['customer_tracking', 'product_detections', 'sales_transactions', 'customer_behavior']
        
        for table in tables:
            cursor.execute(f'SELECT COUNT(*) FROM {table}')
            count = cursor.fetchone()[0]
            
            cursor.execute(f'SELECT MIN(timestamp), MAX(timestamp) FROM {table}')
            time_range = cursor.fetchone()
            
            stats[table] = {
                'record_count': count,
                'date_range': {
                    'start': time_range[0],
                    'end': time_range[1]
                }
            }
        
        conn.close()
        return stats