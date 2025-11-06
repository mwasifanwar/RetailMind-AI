# utils/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

class RetailVisualizer:
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    def plot_customer_heatmap(self, heatmap_data, store_layout=None):
        fig, ax = plt.subplots(figsize=(12, 8))
        
        im = ax.imshow(heatmap_data, cmap='hot', alpha=0.7, 
                      extent=[0, 640, 480, 0])
        
        if store_layout is not None:
            ax.imshow(store_layout, alpha=0.3, cmap='gray',
                     extent=[0, 640, 480, 0])
        
        ax.set_title('Customer Traffic Heatmap')
        ax.set_xlabel('Store Width (pixels)')
        ax.set_ylabel('Store Height (pixels)')
        
        plt.colorbar(im, ax=ax, label='Customer Density')
        plt.tight_layout()
        
        return fig
    
    def plot_customer_paths(self, customer_paths, store_layout=None):
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if store_layout is not None:
            ax.imshow(store_layout, alpha=0.3, cmap='gray',
                     extent=[0, 640, 480, 0])
        
        for i, path in enumerate(customer_paths):
            if len(path) > 1:
                x_coords = [p[0] for p in path]
                y_coords = [p[1] for p in path]
                
                ax.plot(x_coords, y_coords, linewidth=2, alpha=0.7,
                       label=f'Customer {i+1}')
        
        ax.set_title('Customer Movement Paths')
        ax.set_xlabel('Store Width (pixels)')
        ax.set_ylabel('Store Height (pixels)')
        ax.legend()
        plt.tight_layout()
        
        return fig
    
    def plot_shelf_occupancy(self, shelf_data, shelf_id):
        categories = list(shelf_data.keys())
        occupancy_rates = [data['occupancy_rate'] for data in shelf_data.values()]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(categories, occupancy_rates, color=self.colors[:len(categories)])
        
        ax.set_title(f'Shelf {shelf_id} - Product Occupancy Rates')
        ax.set_xlabel('Product Categories')
        ax.set_ylabel('Occupancy Rate')
        ax.set_ylim(0, 1)
        
        for bar, rate in zip(bars, occupancy_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{rate:.1%}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    def plot_sales_trends(self, sales_data, time_period='daily'):
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Revenue Trend', 'Transaction Volume', 
                          'Average Transaction Value', 'Customer Count'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                  [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        dates = [item['date'] for item in sales_data]
        revenue = [item['daily_revenue'] for item in sales_data]
        transactions = [item['transaction_count'] for item in sales_data]
        avg_value = [item['avg_transaction_value'] for item in sales_data]
        customers = [item['unique_customers'] for item in sales_data]
        
        fig.add_trace(
            go.Scatter(x=dates, y=revenue, name='Revenue', line=dict(color='blue')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=dates, y=transactions, name='Transactions', marker_color='green'),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=dates, y=avg_value, name='Avg Value', line=dict(color='red')),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=dates, y=customers, name='Customers', line=dict(color='purple')),
            row=2, col=2
        )
        
        fig.update_layout(height=600, title_text="Sales Performance Dashboard")
        
        return fig
    
    def plot_customer_segmentation(self, segments_data):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        segment_ids = []
        customer_counts = []
        avg_loyalty = []
        avg_conversion = []
        
        for segment_id, data in segments_data.items():
            segment_ids.append(f"Segment {segment_id}")
            customer_counts.append(data['customer_count'])
            avg_loyalty.append(data['avg_loyalty_score'])
            avg_conversion.append(data['avg_conversion_rate'])
        
        ax1.pie(customer_counts, labels=segment_ids, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Customer Distribution by Segment')
        
        ax2.bar(segment_ids, avg_loyalty, color=self.colors[:len(segment_ids)])
        ax2.set_title('Average Loyalty Score by Segment')
        ax2.set_ylabel('Loyalty Score')
        
        ax3.bar(segment_ids, avg_conversion, color=self.colors[:len(segment_ids)])
        ax3.set_title('Average Conversion Rate by Segment')
        ax3.set_ylabel('Conversion Rate')
        
        scatter_data = []
        for segment_id, data in segments_data.items():
            for customer_id in data['customer_ids'][:10]:
                scatter_data.append({
                    'segment': segment_id,
                    'loyalty': data['avg_loyalty_score'],
                    'conversion': data['avg_conversion_rate']
                })
        
        if scatter_data:
            df = pd.DataFrame(scatter_data)
            sns.scatterplot(data=df, x='loyalty', y='conversion', hue='segment', ax=ax4, s=100)
            ax4.set_title('Customer Segments: Loyalty vs Conversion')
        
        plt.tight_layout()
        return fig
    
    def create_store_layout_visualization(self, store_layout, customer_heatmap, 
                                        product_placements, recommendations):
        fig, ax = plt.subplots(figsize=(15, 10))
        
        ax.imshow(store_layout, alpha=0.5, cmap='gray')
        
        heatmap_resized = cv2.resize(customer_heatmap, 
                                   (store_layout.shape[1], store_layout.shape[0]))
        ax.imshow(heatmap_resized, cmap='hot', alpha=0.3)
        
        for product, location in product_placements.items():
            x, y = location
            ax.scatter(x, y, s=100, label=product, alpha=0.7)
        
        for rec in recommendations:
            if rec['type'] == 'bottleneck_relief':
                x, y = rec['location']
                ax.scatter(x, y, s=200, marker='X', color='red', 
                          label='Bottleneck')
            elif rec['type'] == 'dead_zone_activation':
                x, y = rec['location']
                ax.scatter(x, y, s=200, marker='s', color='blue',
                          label='Dead Zone')
        
        ax.set_title('Store Layout Optimization')
        ax.set_xlabel('Store Width')
        ax.set_ylabel('Store Height')
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_recommendation_performance(self, recommendation_data):
        products = [rec['product_name'] for rec in recommendation_data]
        confidences = [rec['confidence'] for rec in recommendation_data]
        reasons = [rec['reason'] for rec in recommendation_data]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.barh(products, confidences, color=self.colors[:len(products)])
        
        ax.set_xlabel('Confidence Score')
        ax.set_title('Product Recommendation Scores')
        ax.set_xlim(0, 1)
        
        for i, (bar, reason) in enumerate(zip(bars, reasons)):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   reason, va='center', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def save_visualization(self, fig, filename, format='png', dpi=300):
        fig.savefig(filename, format=format, dpi=dpi, bbox_inches='tight')
        plt.close(fig)