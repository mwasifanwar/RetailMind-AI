# core/behavior_analyzer.py
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import json
from collections import defaultdict

class BehaviorAnalyzer:
    def __init__(self):
        self.customer_sessions = defaultdict(list)
        self.behavior_patterns = defaultdict(dict)
        self.cluster_models = {}
        self.scaler = StandardScaler()
        
        self.session_timeout = timedelta(minutes=30)
    
    def add_customer_session(self, customer_id, session_data):
        session_data['timestamp'] = datetime.now()
        self.customer_sessions[customer_id].append(session_data)
        
        if len(self.customer_sessions[customer_id]) > 1000:
            self.customer_sessions[customer_id] = self.customer_sessions[customer_id][-500:]
    
    def analyze_customer_behavior(self, customer_id):
        sessions = self.customer_sessions.get(customer_id, [])
        
        if not sessions:
            return self._get_default_behavior_profile(customer_id)
        
        features = self._extract_behavior_features(sessions)
        
        behavior_profile = {
            'customer_id': customer_id,
            'total_sessions': len(sessions),
            'shopping_style': self._classify_shopping_style(features),
            'preferred_times': self._analyze_preferred_times(sessions),
            'browsing_pattern': self._analyze_browsing_pattern(sessions),
            'purchase_behavior': self._analyze_purchase_behavior(sessions),
            'engagement_level': self._calculate_engagement_level(sessions),
            'loyalty_score': self._calculate_loyalty_score(sessions)
        }
        
        self.behavior_patterns[customer_id] = behavior_profile
        return behavior_profile
    
    def _extract_behavior_features(self, sessions):
        features = []
        
        for session in sessions:
            feature_vector = [
                session.get('dwell_time', 0),
                session.get('products_viewed', 0),
                session.get('products_purchased', 0),
                session.get('total_value', 0),
                session.get('aisles_visited', 0),
                self._time_to_feature(session.get('timestamp'))
            ]
            features.append(feature_vector)
        
        return np.array(features) if features else np.array([])
    
    def _time_to_feature(self, timestamp):
        if not timestamp:
            return 0.0
        
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp)
            except:
                return 0.0
        
        hour = timestamp.hour
        minute = timestamp.minute
        
        return (hour * 60 + minute) / (24 * 60)
    
    def _classify_shopping_style(self, features):
        if len(features) == 0:
            return 'unknown'
        
        if len(features) < 3:
            return 'exploratory'
        
        try:
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(features)
            
            cluster_centers = kmeans.cluster_centers_
            
            avg_dwell_times = cluster_centers[:, 0]
            avg_products_viewed = cluster_centers[:, 1]
            avg_purchases = cluster_centers[:, 2]
            
            cluster_labels = []
            for i in range(len(cluster_centers)):
                if avg_purchases[i] > np.median(avg_purchases) and avg_dwell_times[i] < np.median(avg_dwell_times):
                    cluster_labels.append('efficient')
                elif avg_products_viewed[i] > np.median(avg_products_viewed) and avg_purchases[i] < np.median(avg_purchases):
                    cluster_labels.append('browser')
                else:
                    cluster_labels.append('balanced')
            
            most_common_cluster = np.bincount(clusters).argmax()
            return cluster_labels[most_common_cluster]
        
        except:
            return 'balanced'
    
    def _analyze_preferred_times(self, sessions):
        if not sessions:
            return {'peak_hours': [], 'preferred_days': []}
        
        hours = []
        days = []
        
        for session in sessions:
            timestamp = session.get('timestamp')
            if timestamp:
                if isinstance(timestamp, str):
                    try:
                        timestamp = datetime.fromisoformat(timestamp)
                    except:
                        continue
                
                hours.append(timestamp.hour)
                days.append(timestamp.weekday())
        
        if not hours:
            return {'peak_hours': [], 'preferred_days': []}
        
        hour_counts = np.bincount(hours, minlength=24)
        day_counts = np.bincount(days, minlength=7)
        
        peak_hours = np.argsort(hour_counts)[-3:][::-1].tolist()
        preferred_days = np.argsort(day_counts)[-2:][::-1].tolist()
        
        return {
            'peak_hours': peak_hours,
            'preferred_days': preferred_days,
            'hour_distribution': hour_counts.tolist(),
            'day_distribution': day_counts.tolist()
        }
    
    def _analyze_browsing_pattern(self, sessions):
        if not sessions:
            return {'pattern_type': 'unknown', 'depth': 0, 'variety': 0}
        
        total_products_viewed = sum(s.get('products_viewed', 0) for s in sessions)
        total_aisles_visited = sum(s.get('aisles_visited', 0) for s in sessions)
        total_dwell_time = sum(s.get('dwell_time', 0) for s in sessions)
        
        avg_products_per_session = total_products_viewed / len(sessions)
        avg_aisles_per_session = total_aisles_visited / len(sessions)
        avg_dwell_per_session = total_dwell_time / len(sessions)
        
        if avg_products_per_session > 20 and avg_aisles_per_session > 5:
            pattern_type = 'exploratory'
        elif avg_products_per_session < 10 and avg_aisles_per_session < 3:
            pattern_type = 'focused'
        else:
            pattern_type = 'balanced'
        
        return {
            'pattern_type': pattern_type,
            'depth': avg_products_per_session,
            'variety': avg_aisles_per_session,
            'time_intensity': avg_dwell_per_session
        }
    
    def _analyze_purchase_behavior(self, sessions):
        if not sessions:
            return {'conversion_rate': 0, 'avg_basket_size': 0, 'impulse_ratio': 0}
        
        sessions_with_purchase = [s for s in sessions if s.get('products_purchased', 0) > 0]
        total_purchases = sum(s.get('products_purchased', 0) for s in sessions)
        total_value = sum(s.get('total_value', 0) for s in sessions)
        
        conversion_rate = len(sessions_with_purchase) / len(sessions)
        avg_basket_size = total_value / len(sessions_with_purchase) if sessions_with_purchase else 0
        
        planned_purchases = sum(s.get('planned_purchases', 0) for s in sessions)
        impulse_ratio = 1 - (planned_purchases / total_purchases) if total_purchases > 0 else 0
        
        return {
            'conversion_rate': conversion_rate,
            'avg_basket_size': avg_basket_size,
            'impulse_ratio': impulse_ratio,
            'purchase_frequency': len(sessions_with_purchase) / max(1, (len(sessions) / 7))
        }
    
    def _calculate_engagement_level(self, sessions):
        if not sessions:
            return 'low'
        
        recent_sessions = [s for s in sessions 
                         if datetime.now() - s.get('timestamp', datetime.now()) < timedelta(days=30)]
        
        session_count = len(recent_sessions)
        avg_session_length = np.mean([s.get('dwell_time', 0) for s in recent_sessions])
        
        if session_count >= 8 and avg_session_length > 600:
            return 'high'
        elif session_count >= 4 and avg_session_length > 300:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_loyalty_score(self, sessions):
        if not sessions:
            return 0.0
        
        recent_sessions = [s for s in sessions 
                         if datetime.now() - s.get('timestamp', datetime.now()) < timedelta(days=90)]
        
        if not recent_sessions:
            return 0.0
        
        session_frequency = len(recent_sessions) / 90
        total_value = sum(s.get('total_value', 0) for s in recent_sessions)
        avg_session_value = total_value / len(recent_sessions)
        
        consistency_score = 1.0 - (np.std([s.get('total_value', 0) for s in recent_sessions]) / avg_session_value 
                                 if avg_session_value > 0 else 1.0)
        
        loyalty_score = (
            min(session_frequency * 10, 1.0) * 0.4 +
            min(avg_session_value / 100, 1.0) * 0.3 +
            consistency_score * 0.3
        )
        
        return min(loyalty_score, 1.0)
    
    def _get_default_behavior_profile(self, customer_id):
        return {
            'customer_id': customer_id,
            'total_sessions': 0,
            'shopping_style': 'unknown',
            'preferred_times': {'peak_hours': [], 'preferred_days': []},
            'browsing_pattern': {'pattern_type': 'unknown', 'depth': 0, 'variety': 0},
            'purchase_behavior': {'conversion_rate': 0, 'avg_basket_size': 0, 'impulse_ratio': 0},
            'engagement_level': 'low',
            'loyalty_score': 0.0
        }
    
    def segment_customers(self, min_sessions=3):
        customers_with_data = []
        feature_vectors = []
        
        for customer_id, sessions in self.customer_sessions.items():
            if len(sessions) >= min_sessions:
                profile = self.analyze_customer_behavior(customer_id)
                
                feature_vector = [
                    profile.get('loyalty_score', 0),
                    profile['purchase_behavior'].get('conversion_rate', 0),
                    profile['purchase_behavior'].get('avg_basket_size', 0),
                    profile['browsing_pattern'].get('depth', 0),
                    profile['browsing_pattern'].get('variety', 0)
                ]
                
                customers_with_data.append(customer_id)
                feature_vectors.append(feature_vector)
        
        if len(feature_vectors) < 5:
            return {}
        
        try:
            feature_array = np.array(feature_vectors)
            feature_array = self.scaler.fit_transform(feature_array)
            
            dbscan = DBSCAN(eps=0.5, min_samples=3)
            clusters = dbscan.fit_predict(feature_array)
            
            segments = defaultdict(list)
            for customer_id, cluster_id in zip(customers_with_data, clusters):
                segments[cluster_id].append(customer_id)
            
            segment_profiles = {}
            for cluster_id, customer_list in segments.items():
                if cluster_id != -1:
                    segment_features = feature_array[[clusters == cluster_id]]
                    segment_profile = {
                        'segment_id': int(cluster_id),
                        'customer_count': len(customer_list),
                        'avg_loyalty_score': np.mean([f[0] for f in segment_features]),
                        'avg_conversion_rate': np.mean([f[1] for f in segment_features]),
                        'avg_basket_size': np.mean([f[2] for f in segment_features]),
                        'customer_ids': customer_list
                    }
                    segment_profiles[cluster_id] = segment_profile
            
            return segment_profiles
        
        except Exception as e:
            print(f"Segmentation failed: {e}")
            return {}
    
    def predict_customer_value(self, customer_id, future_days=30):
        profile = self.analyze_customer_behavior(customer_id)
        sessions = self.customer_sessions.get(customer_id, [])
        
        if not sessions:
            return {
                'predicted_visits': 0,
                'predicted_value': 0,
                'confidence': 0.0
            }
        
        recent_sessions = [s for s in sessions 
                         if datetime.now() - s.get('timestamp', datetime.now()) < timedelta(days=90)]
        
        if not recent_sessions:
            return {
                'predicted_visits': 0,
                'predicted_value': 0,
                'confidence': 0.0
            }
        
        visit_frequency = len(recent_sessions) / 90
        avg_session_value = np.mean([s.get('total_value', 0) for s in recent_sessions])
        
        predicted_visits = visit_frequency * future_days
        predicted_value = predicted_visits * avg_session_value
        
        value_variance = np.var([s.get('total_value', 0) for s in recent_sessions])
        confidence = 1.0 - (value_variance / (avg_session_value ** 2)) if avg_session_value > 0 else 0.5
        
        return {
            'predicted_visits': predicted_visits,
            'predicted_value': predicted_value,
            'confidence': min(confidence, 1.0)
        }
    
    def detect_anomalous_behavior(self, customer_id, current_session):
        profile = self.analyze_customer_behavior(customer_id)
        historical_sessions = self.customer_sessions.get(customer_id, [])
        
        if len(historical_sessions) < 5:
            return {'is_anomalous': False, 'anomaly_score': 0.0, 'reasons': []}
        
        historical_dwell_times = [s.get('dwell_time', 0) for s in historical_sessions]
        historical_basket_sizes = [s.get('total_value', 0) for s in historical_sessions]
        
        current_dwell = current_session.get('dwell_time', 0)
        current_basket = current_session.get('total_value', 0)
        
        dwell_mean = np.mean(historical_dwell_times)
        dwell_std = np.std(historical_dwell_times)
        
        basket_mean = np.mean(historical_basket_sizes)
        basket_std = np.std(historical_basket_sizes)
        
        anomalies = []
        anomaly_score = 0.0
        
        if dwell_std > 0:
            dwell_zscore = abs(current_dwell - dwell_mean) / dwell_std
            if dwell_zscore > 2.5:
                anomalies.append('unusual_session_length')
                anomaly_score += dwell_zscore / 2.5
        
        if basket_std > 0:
            basket_zscore = abs(current_basket - basket_mean) / basket_std
            if basket_zscore > 2.5:
                anomalies.append('unusual_purchase_value')
                anomaly_score += basket_zscore / 2.5
        
        return {
            'is_anomalous': len(anomalies) > 0,
            'anomaly_score': anomaly_score / len(anomalies) if anomalies else 0.0,
            'reasons': anomalies
        }