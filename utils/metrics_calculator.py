# utils/metrics_calculator.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

class MetricsCalculator:
    def __init__(self):
        self.customer_metrics = {}
        self.inventory_metrics = {}
        self.sales_metrics = {}
    
    def calculate_customer_metrics(self, tracking_data, ground_truth=None):
        metrics = {}
        
        total_customers = len(tracking_data.get('active_tracks', []))
        avg_dwell_time = np.mean(list(tracking_data.get('dwell_times', {}).values())) \
                        if tracking_data.get('dwell_times') else 0
        
        movement_data = tracking_data.get('movement_analysis', {})
        if movement_data:
            total_distances = [data.get('total_distance', 0) for data in movement_data.values()]
            avg_movement_distance = np.mean(total_distances) if total_distances else 0
            movement_efficiency = np.mean([data.get('avg_speed', 0) for data in movement_data.values()])
        else:
            avg_movement_distance = 0
            movement_efficiency = 0
        
        metrics.update({
            'customer_count': total_customers,
            'average_dwell_time': avg_dwell_time,
            'average_movement_distance': avg_movement_distance,
            'movement_efficiency': movement_efficiency,
            'store_coverage': self._calculate_store_coverage(tracking_data.get('heatmap_data')),
            'bottleneck_score': self._calculate_bottleneck_score(tracking_data)
        })
        
        if ground_truth:
            detection_accuracy = self._calculate_detection_accuracy(tracking_data, ground_truth)
            metrics['detection_accuracy'] = detection_accuracy
        
        self.customer_metrics = metrics
        return metrics
    
    def _calculate_store_coverage(self, heatmap_data):
        if heatmap_data is None:
            return 0.0
        
        coverage_threshold = np.percentile(heatmap_data, 70)
        covered_area = np.sum(heatmap_data > coverage_threshold)
        total_area = heatmap_data.size
        
        return covered_area / total_area
    
    def _calculate_bottleneck_score(self, tracking_data):
        movement_data = tracking_data.get('movement_analysis', {})
        if not movement_data:
            return 0.0
        
        speeds = [data.get('avg_speed', 0) for data in movement_data.values()]
        speed_variance = np.var(speeds) if speeds else 0
        
        max_speed = max(speeds) if speeds else 1
        bottleneck_score = speed_variance / (max_speed ** 2) if max_speed > 0 else 0
        
        return min(bottleneck_score, 1.0)
    
    def _calculate_detection_accuracy(self, tracking_data, ground_truth):
        detected_ids = set(tracking_data.get('active_tracks', {}).keys())
        actual_ids = set(ground_truth.get('actual_tracks', {}).keys())
        
        true_positives = len(detected_ids.intersection(actual_ids))
        false_positives = len(detected_ids - actual_ids)
        false_negatives = len(actual_ids - detected_ids)
        
        if true_positives + false_positives > 0:
            precision = true_positives / (true_positives + false_positives)
        else:
            precision = 0.0
        
        if true_positives + false_negatives > 0:
            recall = true_positives / (true_positives + false_negatives)
        else:
            recall = 0.0
        
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': len(detected_ids.intersection(actual_ids)) / len(actual_ids) if actual_ids else 0.0
        }
    
    def calculate_inventory_metrics(self, inventory_data, sales_data):
        metrics = {}
        
        total_products = len(inventory_data)
        low_stock_items = sum(1 for item in inventory_data if item.get('needs_restock', False))
        out_of_stock_items = sum(1 for item in inventory_data if item.get('current_stock', 0) == 0)
        
        stock_ratios = [item.get('stock_ratio', 0) for item in inventory_data]
        avg_stock_ratio = np.mean(stock_ratios) if stock_ratios else 0
        
        occupancy_rates = [item.get('occupancy_rate', 0) for item in inventory_data 
                         if 'occupancy_rate' in item]
        avg_occupancy_rate = np.mean(occupancy_rates) if occupancy_rates else 0
        
        metrics.update({
            'total_products': total_products,
            'low_stock_items': low_stock_items,
            'out_of_stock_items': out_of_stock_items,
            'stockout_rate': out_of_stock_items / total_products if total_products > 0 else 0,
            'average_stock_ratio': avg_stock_ratio,
            'average_occupancy_rate': avg_occupancy_rate,
            'inventory_turnover': self._calculate_inventory_turnover(inventory_data, sales_data),
            'restock_efficiency': self._calculate_restock_efficiency(inventory_data)
        })
        
        self.inventory_metrics = metrics
        return metrics
    
    def _calculate_inventory_turnover(self, inventory_data, sales_data):
        total_cost_of_goods_sold = sum(item.get('daily_sales', 0) * item.get('cost_price', 1) 
                                      for item in inventory_data)
        average_inventory_value = sum(item.get('current_stock', 0) * item.get('cost_price', 1) 
                                     for item in inventory_data)
        
        if average_inventory_value > 0:
            return total_cost_of_goods_sold / average_inventory_value
        else:
            return 0.0
    
    def _calculate_restock_efficiency(self, inventory_data):
        timely_restocks = sum(1 for item in inventory_data 
                            if item.get('days_since_restock', 0) <= item.get('restock_cycle', 7))
        total_restocked_items = sum(1 for item in inventory_data 
                                  if item.get('last_restocked') is not None)
        
        if total_restocked_items > 0:
            return timely_restocks / total_restocked_items
        else:
            return 0.0
    
    def calculate_sales_metrics(self, sales_data, time_period='daily'):
        metrics = {}
        
        total_revenue = sum(item.get('total_amount', 0) for item in sales_data)
        total_transactions = len(sales_data)
        unique_customers = len(set(item.get('customer_id') for item in sales_data))
        
        transaction_values = [item.get('total_amount', 0) for item in sales_data]
        avg_transaction_value = np.mean(transaction_values) if transaction_values else 0
        
        product_counts = [item.get('product_count', 0) for item in sales_data]
        avg_products_per_transaction = np.mean(product_counts) if product_counts else 0
        
        metrics.update({
            'total_revenue': total_revenue,
            'total_transactions': total_transactions,
            'unique_customers': unique_customers,
            'average_transaction_value': avg_transaction_value,
            'average_products_per_transaction': avg_products_per_transaction,
            'conversion_rate': self._calculate_conversion_rate(sales_data),
            'customer_retention_rate': self._calculate_retention_rate(sales_data),
            'sales_growth': self._calculate_sales_growth(sales_data, time_period)
        })
        
        self.sales_metrics = metrics
        return metrics
    
    def _calculate_conversion_rate(self, sales_data):
        if not sales_data:
            return 0.0
        
        browsing_sessions = len(set(item.get('session_id') for item in sales_data 
                                  if item.get('session_id')))
        purchasing_sessions = len(set(item.get('session_id') for item in sales_data 
                                    if item.get('session_id') and item.get('total_amount', 0) > 0))
        
        if browsing_sessions > 0:
            return purchasing_sessions / browsing_sessions
        else:
            return 0.0
    
    def _calculate_retention_rate(self, sales_data):
        if len(sales_data) < 2:
            return 0.0
        
        customer_visits = {}
        for transaction in sales_data:
            customer_id = transaction.get('customer_id')
            timestamp = transaction.get('timestamp')
            
            if customer_id not in customer_visits:
                customer_visits[customer_id] = []
            
            customer_visits[customer_id].append(timestamp)
        
        returning_customers = 0
        for visits in customer_visits.values():
            if len(visits) >= 2:
                returning_customers += 1
        
        total_customers = len(customer_visits)
        
        if total_customers > 0:
            return returning_customers / total_customers
        else:
            return 0.0
    
    def _calculate_sales_growth(self, sales_data, time_period):
        if len(sales_data) < 2:
            return 0.0
        
        sales_by_period = {}
        for transaction in sales_data:
            timestamp = transaction.get('timestamp')
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            
            if time_period == 'daily':
                period_key = timestamp.date()
            elif time_period == 'weekly':
                period_key = timestamp.isocalendar()[1]
            else:
                period_key = timestamp.month
            
            if period_key not in sales_by_period:
                sales_by_period[period_key] = 0
            
            sales_by_period[period_key] += transaction.get('total_amount', 0)
        
        sorted_periods = sorted(sales_by_period.keys())
        if len(sorted_periods) < 2:
            return 0.0
        
        recent_sales = sales_by_period[sorted_periods[-1]]
        previous_sales = sales_by_period[sorted_periods[-2]]
        
        if previous_sales > 0:
            return (recent_sales - previous_sales) / previous_sales
        else:
            return 0.0 if recent_sales == 0 else 1.0
    
    def calculate_recommendation_metrics(self, recommendations, actual_purchases):
        if not recommendations or not actual_purchases:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'hit_rate': 0.0,
                'mean_reciprocal_rank': 0.0
            }
        
        recommended_products = [rec['product_name'] for rec in recommendations]
        purchased_products = [purchase['product_id'] for purchase in actual_purchases]
        
        true_positives = len(set(recommended_products).intersection(set(purchased_products)))
        
        precision = true_positives / len(recommended_products) if recommended_products else 0.0
        recall = true_positives / len(purchased_products) if purchased_products else 0.0
        
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        
        hit_rate = 1.0 if true_positives > 0 else 0.0
        
        reciprocal_ranks = []
        for purchased_product in purchased_products:
            if purchased_product in recommended_products:
                rank = recommended_products.index(purchased_product) + 1
                reciprocal_ranks.append(1.0 / rank)
        
        mean_reciprocal_rank = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'hit_rate': hit_rate,
            'mean_reciprocal_rank': mean_reciprocal_rank
        }
    
    def get_overall_performance_score(self, weights=None):
        if weights is None:
            weights = {
                'customer': 0.3,
                'inventory': 0.25,
                'sales': 0.3,
                'recommendation': 0.15
            }
        
        customer_score = self._normalize_metrics(self.customer_metrics)
        inventory_score = self._normalize_metrics(self.inventory_metrics)
        sales_score = self._normalize_metrics(self.sales_metrics)
        recommendation_score = self.customer_metrics.get('detection_accuracy', {}).get('f1_score', 0) \
                              if 'detection_accuracy' in self.customer_metrics else 0.5
        
        overall_score = (
            customer_score * weights['customer'] +
            inventory_score * weights['inventory'] +
            sales_score * weights['sales'] +
            recommendation_score * weights['recommendation']
        )
        
        return {
            'overall_score': overall_score,
            'component_scores': {
                'customer_analytics': customer_score,
                'inventory_management': inventory_score,
                'sales_performance': sales_score,
                'recommendation_quality': recommendation_score
            },
            'weights': weights
        }
    
    def _normalize_metrics(self, metrics):
        if not metrics:
            return 0.5
        
        scores = []
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if 'rate' in key or 'ratio' in key or 'score' in key:
                    scores.append(min(value, 1.0))
                elif 'time' in key:
                    normalized = min(value / 3600, 1.0)
                    scores.append(normalized)
                elif 'count' in key or 'items' in key:
                    normalized = min(value / 100, 1.0)
                    scores.append(normalized)
                else:
                    scores.append(min(value, 1.0))
        
        return np.mean(scores) if scores else 0.5