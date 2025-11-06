# data/data_processor.py
import pandas as pd
import numpy as np
import cv2
from datetime import datetime, timedelta
import json
from collections import defaultdict

class DataProcessor:
    def __init__(self):
        self.customer_data = defaultdict(dict)
        self.product_data = {}
        self.sales_data = []
        
    def process_customer_tracking_data(self, tracking_data):
        processed_data = {
            'customer_count': len(tracking_data.get('active_tracks', [])),
            'dwell_times': tracking_data.get('dwell_times', {}),
            'movement_patterns': tracking_data.get('movement_analysis', {}),
            'heatmap': tracking_data.get('heatmap_data'),
            'timestamp': datetime.now().isoformat()
        }
        
        return processed_data
    
    def process_shelf_analysis_data(self, shelf_data):
        processed_data = {
            'shelf_id': shelf_data.get('shelf_id'),
            'occupancy_rate': shelf_data.get('occupancy_rate', 0),
            'product_distribution': dict(shelf_data.get('product_distribution', {})),
            'restock_alerts': shelf_data.get('restock_alerts', []),
            'detection_confidence': shelf_data.get('detection_confidence', 0),
            'timestamp': datetime.now().isoformat()
        }
        
        return processed_data
    
    def process_sales_transaction(self, transaction_data):
        processed_transaction = {
            'transaction_id': transaction_data.get('transaction_id'),
            'customer_id': transaction_data.get('customer_id'),
            'products': transaction_data.get('products', []),
            'total_amount': transaction_data.get('total_amount', 0),
            'timestamp': transaction_data.get('timestamp', datetime.now().isoformat()),
            'payment_method': transaction_data.get('payment_method', 'unknown'),
            'store_location': transaction_data.get('store_location', 'unknown')
        }
        
        self.sales_data.append(processed_transaction)
        
        for product in transaction_data.get('products', []):
            product_id = product.get('product_id')
            if product_id not in self.product_data:
                self.product_data[product_id] = {
                    'total_sold': 0,
                    'revenue': 0,
                    'transactions': 0
                }
            
            self.product_data[product_id]['total_sold'] += product.get('quantity', 0)
            self.product_data[product_id]['revenue'] += product.get('price', 0) * product.get('quantity', 0)
            self.product_data[product_id]['transactions'] += 1
        
        return processed_transaction
    
    def aggregate_customer_behavior(self, customer_id, time_period='daily'):
        customer_sessions = self.customer_data.get(customer_id, {})
        
        if not customer_sessions:
            return None
        
        aggregated_data = {
            'customer_id': customer_id,
            'time_period': time_period,
            'total_visits': len(customer_sessions),
            'avg_dwell_time': np.mean([s.get('dwell_time', 0) for s in customer_sessions.values()]),
            'total_spent': sum(s.get('total_spent', 0) for s in customer_sessions.values()),
            'preferred_categories': self._get_preferred_categories(customer_sessions),
            'visit_frequency': self._calculate_visit_frequency(customer_sessions),
            'last_visit': max(s.get('timestamp') for s in customer_sessions.values()) 
                        if customer_sessions else None
        }
        
        return aggregated_data
    
    def _get_preferred_categories(self, customer_sessions):
        category_counts = defaultdict(int)
        for session in customer_sessions.values():
            for category in session.get('categories_visited', []):
                category_counts[category] += 1
        
        return sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    def _calculate_visit_frequency(self, customer_sessions):
        if len(customer_sessions) < 2:
            return 'unknown'
        
        timestamps = [s.get('timestamp') for s in customer_sessions.values() 
                     if isinstance(s.get('timestamp'), datetime)]
        
        if len(timestamps) < 2:
            return 'unknown'
        
        timestamps.sort()
        time_diffs = [(timestamps[i] - timestamps[i-1]).days 
                     for i in range(1, len(timestamps))]
        
        avg_days_between_visits = np.mean(time_diffs)
        
        if avg_days_between_visits < 7:
            return 'weekly'
        elif avg_days_between_visits < 30:
            return 'monthly'
        else:
            return 'occasional'
    
    def generate_sales_report(self, start_date, end_date):
        report_data = {
            'period': f"{start_date} to {end_date}",
            'total_sales': 0,
            'total_transactions': 0,
            'average_transaction_value': 0,
            'top_products': [],
            'customer_metrics': {},
            'category_performance': defaultdict(float)
        }
        
        period_sales = [t for t in self.sales_data 
                       if start_date <= t.get('timestamp') <= end_date]
        
        if not period_sales:
            return report_data
        
        report_data['total_sales'] = sum(t.get('total_amount', 0) for t in period_sales)
        report_data['total_transactions'] = len(period_sales)
        report_data['average_transaction_value'] = report_data['total_sales'] / len(period_sales)
        
        product_sales = defaultdict(float)
        for transaction in period_sales:
            for product in transaction.get('products', []):
                product_id = product.get('product_id')
                product_sales[product_id] += product.get('price', 0) * product.get('quantity', 0)
        
        report_data['top_products'] = sorted(product_sales.items(), 
                                           key=lambda x: x[1], reverse=True)[:10]
        
        customer_visits = defaultdict(int)
        customer_spending = defaultdict(float)
        
        for transaction in period_sales:
            customer_id = transaction.get('customer_id')
            customer_visits[customer_id] += 1
            customer_spending[customer_id] += transaction.get('total_amount', 0)
        
        report_data['customer_metrics'] = {
            'unique_customers': len(customer_visits),
            'repeat_customers': sum(1 for visits in customer_visits.values() if visits > 1),
            'avg_visits_per_customer': np.mean(list(customer_visits.values())) if customer_visits else 0,
            'avg_spending_per_customer': np.mean(list(customer_spending.values())) if customer_spending else 0
        }
        
        return report_data
    
    def calculate_product_affinity(self, product_pairs):
        affinity_matrix = defaultdict(lambda: defaultdict(float))
        
        for transaction in self.sales_data:
            products = [p.get('product_id') for p in transaction.get('products', [])]
            
            for i, product1 in enumerate(products):
                for product2 in products[i+1:]:
                    affinity_matrix[product1][product2] += 1
                    affinity_matrix[product2][product1] += 1
        
        total_transactions = len(self.sales_data)
        if total_transactions > 0:
            for product1 in affinity_matrix:
                for product2 in affinity_matrix[product1]:
                    affinity_matrix[product1][product2] /= total_transactions
        
        return affinity_matrix
    
    def detect_seasonal_patterns(self, years_of_data=2):
        if len(self.sales_data) < 30:
            return {}
        
        sales_df = pd.DataFrame(self.sales_data)
        sales_df['timestamp'] = pd.to_datetime(sales_df['timestamp'])
        sales_df['month'] = sales_df['timestamp'].dt.month
        sales_df['day_of_week'] = sales_df['timestamp'].dt.dayofweek
        
        monthly_patterns = sales_df.groupby('month')['total_amount'].agg(['sum', 'count', 'mean']).to_dict()
        daily_patterns = sales_df.groupby('day_of_week')['total_amount'].agg(['sum', 'count', 'mean']).to_dict()
        
        return {
            'monthly_patterns': monthly_patterns,
            'daily_patterns': daily_patterns,
            'seasonal_peaks': self._find_seasonal_peaks(monthly_patterns),
            'weekly_peaks': self._find_weekly_peaks(daily_patterns)
        }
    
    def _find_seasonal_peaks(self, monthly_patterns):
        if not monthly_patterns.get('sum'):
            return []
        
        monthly_totals = monthly_patterns['sum']
        peak_months = sorted(monthly_totals.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return [{'month': month, 'revenue': revenue} for month, revenue in peak_months]
    
    def _find_weekly_peaks(self, daily_patterns):
        if not daily_patterns.get('sum'):
            return []
        
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_totals = daily_patterns['sum']
        
        peak_days = sorted(daily_totals.items(), key=lambda x: x[1], reverse=True)[:2]
        
        return [{'day': day_names[day], 'revenue': revenue} for day, revenue in peak_days]