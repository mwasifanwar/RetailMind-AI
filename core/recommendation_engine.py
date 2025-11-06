# core/recommendation_engine.py
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import json
from collections import defaultdict
import torch
import torch.nn as nn

class TransformerRecommendationModel(nn.Module):
    def __init__(self, num_products, embedding_dim=64, num_heads=4, num_layers=2):
        super(TransformerRecommendationModel, self).__init__()
        
        self.product_embedding = nn.Embedding(num_products, embedding_dim)
        self.position_embedding = nn.Embedding(100, embedding_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=256,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_products)
        )
    
    def forward(self, product_sequences):
        batch_size, seq_len = product_sequences.shape
        
        product_embeddings = self.product_embedding(product_sequences)
        
        positions = torch.arange(seq_len).expand(batch_size, seq_len).to(product_sequences.device)
        position_embeddings = self.position_embedding(positions)
        
        embeddings = product_embeddings + position_embeddings
        embeddings = embeddings.permute(1, 0, 2)
        
        transformer_output = self.transformer_encoder(embeddings)
        transformer_output = transformer_output.permute(1, 0, 2)
        
        last_output = transformer_output[:, -1, :]
        
        output = self.fc(last_output)
        
        return output

class RecommendationEngine:
    def __init__(self, model_path=None):
        self.product_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.model = None
        self.product_data = {}
        self.customer_profiles = defaultdict(dict)
        self.transaction_history = defaultdict(list)
        
        self.affinity_matrix = None
        self.popular_products = []
        
        if model_path:
            self.load_model(model_path)
    
    def train_recommendation_model(self, transaction_data, product_catalog):
        self._prepare_training_data(transaction_data, product_catalog)
        
        num_products = len(self.product_encoder.classes_)
        self.model = TransformerRecommendationModel(num_products)
        
        sequences = self._create_sequences(transaction_data)
        
        if len(sequences) > 0:
            sequences_tensor = torch.tensor(sequences, dtype=torch.long)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            
            self.model.train()
            for epoch in range(10):
                optimizer.zero_grad()
                outputs = self.model(sequences_tensor[:, :-1])
                loss = criterion(outputs, sequences_tensor[:, -1])
                loss.backward()
                optimizer.step()
    
    def _prepare_training_data(self, transaction_data, product_catalog):
        all_products = list(product_catalog.keys())
        self.product_encoder.fit(all_products)
        
        self.product_data = product_catalog
    
    def _create_sequences(self, transaction_data, sequence_length=5):
        sequences = []
        
        for customer_id, transactions in transaction_data.items():
            if len(transactions) >= sequence_length + 1:
                product_ids = [t['product_id'] for t in transactions]
                encoded_ids = self.product_encoder.transform(product_ids)
                
                for i in range(len(encoded_ids) - sequence_length):
                    sequence = encoded_ids[i:i + sequence_length + 1]
                    sequences.append(sequence)
        
        return np.array(sequences) if sequences else np.array([])
    
    def get_recommendations(self, customer_id, current_basket, top_k=5):
        if self.model is None:
            return self._get_fallback_recommendations(current_basket, top_k)
        
        customer_history = self.transaction_history.get(customer_id, [])
        
        if not customer_history and not current_basket:
            return self._get_popular_recommendations(top_k)
        
        sequence = self._build_sequence(customer_history, current_basket)
        
        if len(sequence) == 0:
            return self._get_fallback_recommendations(current_basket, top_k)
        
        self.model.eval()
        with torch.no_grad():
            sequence_tensor = torch.tensor([sequence], dtype=torch.long)
            predictions = self.model(sequence_tensor)
            top_preds = torch.topk(predictions, top_k, dim=1)
            
            recommended_ids = top_preds.indices[0].numpy()
            recommended_scores = top_preds.values[0].numpy()
        
        recommendations = []
        for product_id, score in zip(recommended_ids, recommended_scores):
            try:
                product_name = self.product_encoder.inverse_transform([product_id])[0]
                recommendations.append({
                    'product_id': product_id,
                    'product_name': product_name,
                    'confidence': float(score),
                    'reason': 'behavioral_pattern'
                })
            except:
                continue
        
        return recommendations
    
    def _build_sequence(self, history, current_basket, max_length=5):
        all_products = []
        
        for transaction in history[-3:]:
            all_products.append(transaction['product_id'])
        
        for product in current_basket:
            all_products.append(product)
        
        if len(all_products) == 0:
            return []
        
        try:
            encoded_products = self.product_encoder.transform(all_products)
            return encoded_products[-max_length:]
        except:
            return []
    
    def _get_fallback_recommendations(self, current_basket, top_k):
        if not current_basket:
            return self._get_popular_recommendations(top_k)
        
        basket_categories = set()
        for product in current_basket:
            if product in self.product_data:
                basket_categories.add(self.product_data[product]['category'])
        
        recommendations = []
        for category in basket_categories:
            category_products = self._get_products_by_category(category)
            for product in category_products[:2]:
                if product not in current_basket:
                    recommendations.append({
                        'product_id': product,
                        'product_name': product,
                        'confidence': 0.7,
                        'reason': f'complementary_to_{category}'
                    })
        
        return recommendations[:top_k]
    
    def _get_popular_recommendations(self, top_k):
        if not self.popular_products:
            self._calculate_popular_products()
        
        recommendations = []
        for product in self.popular_products[:top_k]:
            recommendations.append({
                'product_id': product,
                'product_name': product,
                'confidence': 0.8,
                'reason': 'popular_item'
            })
        
        return recommendations
    
    def _get_products_by_category(self, category):
        return [pid for pid, data in self.product_data.items() 
                if data.get('category') == category]
    
    def _calculate_popular_products(self):
        product_counts = defaultdict(int)
        
        for customer_transactions in self.transaction_history.values():
            for transaction in customer_transactions:
                product_counts[transaction['product_id']] += 1
        
        self.popular_products = [pid for pid, count in 
                               sorted(product_counts.items(), 
                                    key=lambda x: x[1], reverse=True)]
    
    def update_customer_profile(self, customer_id, transaction):
        self.transaction_history[customer_id].append(transaction)
        
        if len(self.transaction_history[customer_id]) > 100:
            self.transaction_history[customer_id] = self.transaction_history[customer_id][-50:]
        
        product_id = transaction['product_id']
        category = self.product_data.get(product_id, {}).get('category', 'unknown')
        
        if 'category_preferences' not in self.customer_profiles[customer_id]:
            self.customer_profiles[customer_id]['category_preferences'] = defaultdict(int)
        
        self.customer_profiles[customer_id]['category_preferences'][category] += 1
        
        self.customer_profiles[customer_id]['last_visit'] = transaction.get('timestamp')
        self.customer_profiles[customer_id]['total_spent'] = \
            self.customer_profiles[customer_id].get('total_spent', 0) + transaction.get('amount', 0)
    
    def get_customer_insights(self, customer_id):
        profile = self.customer_profiles.get(customer_id, {})
        transaction_count = len(self.transaction_history.get(customer_id, []))
        
        if transaction_count == 0:
            return {
                'customer_id': customer_id,
                'loyalty_level': 'new',
                'preferred_categories': [],
                'shopping_frequency': 'unknown',
                'average_basket_size': 0
            }
        
        category_prefs = profile.get('category_preferences', {})
        preferred_categories = sorted(category_prefs.items(), 
                                    key=lambda x: x[1], reverse=True)[:3]
        
        total_spent = profile.get('total_spent', 0)
        avg_basket_size = total_spent / transaction_count if transaction_count > 0 else 0
        
        if transaction_count > 20:
            loyalty_level = 'premium'
        elif transaction_count > 10:
            loyalty_level = 'regular'
        elif transaction_count > 5:
            loyalty_level = 'occasional'
        else:
            loyalty_level = 'new'
        
        return {
            'customer_id': customer_id,
            'loyalty_level': loyalty_level,
            'preferred_categories': [cat for cat, _ in preferred_categories],
            'shopping_frequency': self._calculate_frequency(customer_id),
            'average_basket_size': avg_basket_size,
            'total_transactions': transaction_count,
            'total_spent': total_spent
        }
    
    def _calculate_frequency(self, customer_id):
        transactions = self.transaction_history.get(customer_id, [])
        if len(transactions) < 2:
            return 'unknown'
        
        timestamps = [t.get('timestamp') for t in transactions if t.get('timestamp')]
        if len(timestamps) < 2:
            return 'unknown'
        
        try:
            time_diffs = []
            for i in range(1, len(timestamps)):
                if timestamps[i] and timestamps[i-1]:
                    diff = (timestamps[i] - timestamps[i-1]).days
                    time_diffs.append(diff)
            
            if not time_diffs:
                return 'unknown'
            
            avg_days = np.mean(time_diffs)
            
            if avg_days < 7:
                return 'weekly'
            elif avg_days < 30:
                return 'monthly'
            else:
                return 'occasional'
        except:
            return 'unknown'
    
    def generate_personalized_offers(self, customer_id, current_context=None):
        insights = self.get_customer_insights(customer_id)
        recommendations = self.get_recommendations(customer_id, [])
        
        offers = []
        
        for rec in recommendations[:3]:
            product_name = rec['product_name']
            category = self.product_data.get(product_name, {}).get('category', 'unknown')
            
            offer = {
                'product': product_name,
                'category': category,
                'discount': self._calculate_discount(insights['loyalty_level']),
                'reason': f"Based on your interest in {category}",
                'validity': '7_days',
                'personalized': True
            }
            offers.append(offer)
        
        if insights['loyalty_level'] == 'premium':
            offers.append({
                'product': 'premium_bundle',
                'category': 'special',
                'discount': '15%',
                'reason': 'Premium customer appreciation',
                'validity': '30_days',
                'personalized': True
            })
        
        return offers
    
    def _calculate_discount(self, loyalty_level):
        discounts = {
            'new': '10%',
            'occasional': '12%',
            'regular': '15%',
            'premium': '20%'
        }
        return discounts.get(loyalty_level, '10%')
    
    def save_model(self, model_path):
        if self.model:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'product_encoder': self.product_encoder,
                'product_data': self.product_data
            }, model_path)
    
    def load_model(self, model_path):
        try:
            checkpoint = torch.load(model_path)
            self.product_encoder = checkpoint['product_encoder']
            self.product_data = checkpoint['product_data']
            
            num_products = len(self.product_encoder.classes_)
            self.model = TransformerRecommendationModel(num_products)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except:
            print("Failed to load model, using fallback recommendations")