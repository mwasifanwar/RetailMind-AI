# utils/report_generator.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from collections import defaultdict

class ReportGenerator:
    def __init__(self):
        self.report_templates = {}
        self._load_templates()
    
    def _load_templates(self):
        self.report_templates = {
            'daily_summary': self._daily_summary_template,
            'weekly_analytics': self._weekly_analytics_template,
            'customer_insights': self._customer_insights_template,
            'inventory_status': self._inventory_status_template,
            'performance_review': self._performance_review_template
        }
    
    def generate_report(self, report_type, data, **kwargs):
        if report_type not in self.report_templates:
            raise ValueError(f"Unknown report type: {report_type}")
        
        template_func = self.report_templates[report_type]
        return template_func(data, **kwargs)
    
    def _daily_summary_template(self, data, date=None):
        if date is None:
            date = datetime.now().date()
        
        report = {
            'report_type': 'daily_summary',
            'date': date.isoformat(),
            'generated_at': datetime.now().isoformat(),
            'executive_summary': self._generate_executive_summary(data),
            'key_metrics': self._extract_key_metrics(data),
            'customer_analytics': self._summarize_customer_analytics(data),
            'sales_performance': self._summarize_sales_performance(data),
            'inventory_status': self._summarize_inventory_status(data),
            'recommendations': self._generate_daily_recommendations(data),
            'alerts': self._generate_alerts(data)
        }
        
        return report
    
    def _weekly_analytics_template(self, data, start_date=None, end_date=None):
        if start_date is None:
            start_date = datetime.now().date() - timedelta(days=7)
        if end_date is None:
            end_date = datetime.now().date()
        
        report = {
            'report_type': 'weekly_analytics',
            'period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'generated_at': datetime.now().isoformat(),
            'weekly_overview': self._generate_weekly_overview(data),
            'trend_analysis': self._analyze_weekly_trends(data),
            'customer_segmentation': self._segment_weekly_customers(data),
            'product_performance': self._analyze_product_performance(data),
            'operational_insights': self._generate_operational_insights(data),
            'strategic_recommendations': self._generate_strategic_recommendations(data)
        }
        
        return report
    
    def _customer_insights_template(self, data, customer_segment=None):
        report = {
            'report_type': 'customer_insights',
            'generated_at': datetime.now().isoformat(),
            'customer_demographics': self._analyze_customer_demographics(data),
            'behavior_patterns': self._analyze_behavior_patterns(data),
            'loyalty_analysis': self._analyze_customer_loyalty(data),
            'segmentation_analysis': self._perform_customer_segmentation(data),
            'retention_metrics': self._calculate_retention_metrics(data),
            'personalization_opportunities': self._identify_personalization_opportunities(data)
        }
        
        if customer_segment:
            report['segment_focus'] = customer_segment
        
        return report
    
    def _inventory_status_template(self, data, include_predictions=True):
        report = {
            'report_type': 'inventory_status',
            'generated_at': datetime.now().isoformat(),
            'current_inventory': self._summarize_current_inventory(data),
            'stock_level_analysis': self._analyze_stock_levels(data),
            'restock_recommendations': self._generate_restock_recommendations(data),
            'shelf_optimization': self._analyze_shelf_optimization(data),
            'inventory_turnover': self._calculate_inventory_turnover(data)
        }
        
        if include_predictions:
            report['demand_forecasting'] = self._generate_demand_forecasts(data)
        
        return report
    
    def _performance_review_template(self, data, period='monthly'):
        report = {
            'report_type': 'performance_review',
            'period': period,
            'generated_at': datetime.now().isoformat(),
            'performance_scorecard': self._generate_performance_scorecard(data),
            'kpi_analysis': self._analyze_kpis(data),
            'comparative_analysis': self._perform_comparative_analysis(data),
            'improvement_opportunities': self._identify_improvement_opportunities(data),
            'action_plan': self._generate_action_plan(data)
        }
        
        return report
    
    def _generate_executive_summary(self, data):
        summary = {
            'total_customers': data.get('customer_metrics', {}).get('customer_count', 0),
            'total_revenue': data.get('sales_metrics', {}).get('total_revenue', 0),
            'conversion_rate': data.get('sales_metrics', {}).get('conversion_rate', 0),
            'inventory_health': data.get('inventory_metrics', {}).get('average_stock_ratio', 0),
            'key_achievements': self._identify_key_achievements(data),
            'main_challenges': self._identify_main_challenges(data)
        }
        
        return summary
    
    def _extract_key_metrics(self, data):
        metrics = {}
        
        metrics.update(data.get('customer_metrics', {}))
        metrics.update(data.get('sales_metrics', {}))
        metrics.update(data.get('inventory_metrics', {}))
        
        return {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    
    def _summarize_customer_analytics(self, data):
        customer_data = data.get('customer_data', {})
        
        return {
            'store_traffic': customer_data.get('customer_count', 0),
            'average_dwell_time': customer_data.get('average_dwell_time', 0),
            'peak_hours': customer_data.get('peak_hours', []),
            'popular_areas': customer_data.get('popular_areas', {}),
            'customer_satisfaction': self._estimate_customer_satisfaction(customer_data)
        }
    
    def _summarize_sales_performance(self, data):
        sales_data = data.get('sales_data', {})
        
        return {
            'total_sales': sales_data.get('total_revenue', 0),
            'transaction_count': sales_data.get('total_transactions', 0),
            'average_transaction_value': sales_data.get('average_transaction_value', 0),
            'top_selling_products': sales_data.get('top_products', []),
            'sales_growth': sales_data.get('sales_growth', 0)
        }
    
    def _summarize_inventory_status(self, data):
        inventory_data = data.get('inventory_data', {})
        
        return {
            'total_products': inventory_data.get('total_products', 0),
            'low_stock_items': inventory_data.get('low_stock_items', 0),
            'out_of_stock_items': inventory_data.get('out_of_stock_items', 0),
            'restock_alerts': inventory_data.get('restock_alerts', []),
            'inventory_turnover': inventory_data.get('inventory_turnover', 0)
        }
    
    def _generate_daily_recommendations(self, data):
        recommendations = []
        
        customer_data = data.get('customer_data', {})
        if customer_data.get('customer_count', 0) < 50:
            recommendations.append({
                'type': 'marketing',
                'priority': 'medium',
                'message': 'Consider running promotions to increase store traffic',
                'impact': 'high'
            })
        
        inventory_data = data.get('inventory_data', {})
        if inventory_data.get('out_of_stock_items', 0) > 5:
            recommendations.append({
                'type': 'operations',
                'priority': 'high',
                'message': 'Urgent restock needed for multiple products',
                'impact': 'critical'
            })
        
        sales_data = data.get('sales_data', {})
        if sales_data.get('conversion_rate', 0) < 0.3:
            recommendations.append({
                'type': 'sales',
                'priority': 'medium',
                'message': 'Focus on improving in-store conversion rates',
                'impact': 'medium'
            })
        
        return recommendations
    
    def _generate_alerts(self, data):
        alerts = []
        
        inventory_data = data.get('inventory_data', {})
        if inventory_data.get('out_of_stock_items', 0) > 10:
            alerts.append({
                'level': 'critical',
                'type': 'inventory',
                'message': 'High number of out-of-stock items affecting sales',
                'action_required': True
            })
        
        customer_data = data.get('customer_data', {})
        if customer_data.get('bottleneck_score', 0) > 0.8:
            alerts.append({
                'level': 'warning',
                'type': 'operations',
                'message': 'Store layout may be causing customer congestion',
                'action_required': True
            })
        
        return alerts
    
    def _generate_weekly_overview(self, data):
        return {
            'weekly_revenue': sum(day.get('daily_revenue', 0) for day in data.get('daily_sales', [])),
            'weekly_customers': sum(day.get('unique_customers', 0) for day in data.get('daily_sales', [])),
            'busiest_day': self._find_busiest_day(data),
            'top_performing_category': self._find_top_category(data),
            'week_over_week_growth': self._calculate_week_over_week_growth(data)
        }
    
    def _analyze_weekly_trends(self, data):
        trends = {
            'revenue_trend': self._calculate_trend(data.get('daily_sales', []), 'daily_revenue'),
            'customer_trend': self._calculate_trend(data.get('daily_sales', []), 'unique_customers'),
            'seasonal_patterns': self._identify_seasonal_patterns(data),
            'emerging_trends': self._identify_emerging_trends(data)
        }
        
        return trends
    
    def _segment_weekly_customers(self, data):
        segments = data.get('customer_segments', {})
        
        segment_analysis = {}
        for segment_id, segment_data in segments.items():
            segment_analysis[segment_id] = {
                'size': segment_data.get('customer_count', 0),
                'average_value': segment_data.get('avg_basket_size', 0),
                'loyalty_score': segment_data.get('avg_loyalty_score', 0),
                'preferred_products': segment_data.get('preferred_categories', [])
            }
        
        return segment_analysis
    
    def _analyze_product_performance(self, data):
        products = data.get('product_performance', {})
        
        return {
            'top_sellers': sorted(products.items(), key=lambda x: x[1].get('revenue', 0), reverse=True)[:10],
            'slow_movers': sorted(products.items(), key=lambda x: x[1].get('revenue', 0))[:5],
            'high_margin_products': sorted(products.items(), key=lambda x: x[1].get('margin', 0), reverse=True)[:5],
            'cross_sell_opportunities': self._identify_cross_sell_opportunities(products)
        }
    
    def _generate_operational_insights(self, data):
        return {
            'staffing_recommendations': self._generate_staffing_recommendations(data),
            'layout_optimizations': self._suggest_layout_optimizations(data),
            'inventory_improvements': self._suggest_inventory_improvements(data),
            'customer_experience_enhancements': self._suggest_customer_experience_improvements(data)
        }
    
    def _generate_strategic_recommendations(self, data):
        recommendations = []
        
        customer_growth = data.get('weekly_overview', {}).get('week_over_week_growth', 0)
        if customer_growth < 0.1:
            recommendations.append({
                'category': 'customer_acquisition',
                'priority': 'high',
                'recommendation': 'Implement targeted marketing campaigns to drive customer growth',
                'expected_impact': '15-25% increase in new customers'
            })
        
        inventory_turnover = data.get('inventory_metrics', {}).get('inventory_turnover', 0)
        if inventory_turnover < 2.0:
            recommendations.append({
                'category': 'inventory_management',
                'priority': 'medium',
                'recommendation': 'Optimize inventory levels based on demand forecasting',
                'expected_impact': '20-30% reduction in carrying costs'
            })
        
        return recommendations
    
    def _identify_key_achievements(self, data):
        achievements = []
        
        if data.get('sales_metrics', {}).get('sales_growth', 0) > 0.15:
            achievements.append('Strong sales growth exceeding targets')
        
        if data.get('customer_metrics', {}).get('conversion_rate', 0) > 0.4:
            achievements.append('Excellent customer conversion rates')
        
        if data.get('inventory_metrics', {}).get('stockout_rate', 0) < 0.05:
            achievements.append('Effective inventory management with minimal stockouts')
        
        return achievements
    
    def _identify_main_challenges(self, data):
        challenges = []
        
        if data.get('customer_metrics', {}).get('customer_count', 0) < 100:
            challenges.append('Low store traffic requiring marketing initiatives')
        
        if data.get('inventory_metrics', {}).get('low_stock_items', 0) > 15:
            challenges.append('Multiple products requiring urgent restocking')
        
        return challenges
    
    def _estimate_customer_satisfaction(self, customer_data):
        dwell_time = customer_data.get('average_dwell_time', 0)
        movement_efficiency = customer_data.get('movement_efficiency', 0)
        
        if dwell_time > 1800 and movement_efficiency > 0.7:
            return 'high'
        elif dwell_time > 900 and movement_efficiency > 0.5:
            return 'medium'
        else:
            return 'low'
    
    def _find_busiest_day(self, data):
        daily_sales = data.get('daily_sales', [])
        if not daily_sales:
            return 'Unknown'
        
        busiest = max(daily_sales, key=lambda x: x.get('unique_customers', 0))
        return busiest.get('date', 'Unknown')
    
    def _find_top_category(self, data):
        categories = data.get('category_performance', {})
        if not categories:
            return 'Unknown'
        
        top_category = max(categories.items(), key=lambda x: x[1].get('revenue', 0))
        return top_category[0]
    
    def _calculate_week_over_week_growth(self, data):
        current_week = data.get('weekly_overview', {}).get('weekly_revenue', 0)
        previous_week = data.get('previous_week_data', {}).get('weekly_revenue', current_week * 0.9)
        
        if previous_week > 0:
            return (current_week - previous_week) / previous_week
        else:
            return 0.0
    
    def _calculate_trend(self, data, metric_key):
        if len(data) < 2:
            return 'stable'
        
        values = [day.get(metric_key, 0) for day in data]
        if all(v == 0 for v in values):
            return 'stable'
        
        x = np.arange(len(values))
        y = np.array(values)
        
        slope = np.polyfit(x, y, 1)[0]
        
        if slope > np.std(y) * 0.5:
            return 'increasing'
        elif slope < -np.std(y) * 0.5:
            return 'decreasing'
        else:
            return 'stable'
    
    def _identify_seasonal_patterns(self, data):
        return {
            'weekend_effect': self._analyze_weekend_effect(data),
            'time_of_day_patterns': self._analyze_time_of_day_patterns(data),
            'category_seasonality': self._analyze_category_seasonality(data)
        }
    
    def _identify_emerging_trends(self, data):
        trends = []
        
        product_trends = data.get('emerging_products', {})
        for product, trend_data in product_trends.items():
            if trend_data.get('growth_rate', 0) > 0.5:
                trends.append({
                    'product': product,
                    'trend': 'rapid_growth',
                    'confidence': trend_data.get('confidence', 0)
                })
        
        return trends
    
    def _analyze_customer_demographics(self, data):
        return {
            'age_distribution': data.get('age_distribution', {}),
            'gender_breakdown': data.get('gender_breakdown', {}),
            'location_analysis': data.get('customer_locations', {}),
            'loyalty_tiers': data.get('loyalty_tiers', {})
        }
    
    def _analyze_behavior_patterns(self, data):
        return {
            'shopping_styles': data.get('shopping_styles', {}),
            'visit_frequency': data.get('visit_frequency', {}),
            'basket_composition': data.get('basket_analysis', {}),
            'preferred_times': data.get('preferred_visiting_times', {})
        }
    
    def _analyze_customer_loyalty(self, data):
        return {
            'loyalty_score_distribution': data.get('loyalty_scores', {}),
            'retention_rates': data.get('retention_metrics', {}),
            'lifetime_value_analysis': data.get('customer_lifetime_value', {}),
            'churn_risk_assessment': data.get('churn_risk', {})
        }
    
    def _perform_customer_segmentation(self, data):
        segments = data.get('customer_segments', {})
        
        segment_profiles = {}
        for segment_id, segment_data in segments.items():
            segment_profiles[segment_id] = {
                'profile': segment_data.get('profile', {}),
                'size': segment_data.get('size', 0),
                'value': segment_data.get('average_value', 0),
                'needs': segment_data.get('key_needs', [])
            }
        
        return segment_profiles
    
    def _calculate_retention_metrics(self, data):
        return {
            'overall_retention_rate': data.get('retention_rate', 0),
            'cohort_analysis': data.get('cohort_analysis', {}),
            'churn_analysis': data.get('churn_analysis', {}),
            'reactivation_opportunities': data.get('reactivation_candidates', [])
        }
    
    def _identify_personalization_opportunities(self, data):
        opportunities = []
        
        for segment_id, segment_data in data.get('customer_segments', {}).items():
            opportunities.append({
                'segment': segment_id,
                'personalization_type': 'targeted_offers',
                'opportunity_size': segment_data.get('size', 0),
                'expected_impact': '10-15% increase in conversion'
            })
        
        return opportunities
    
    def _summarize_current_inventory(self, data):
        inventory = data.get('inventory_data', {})
        
        return {
            'total_sku_count': inventory.get('total_products', 0),
            'total_inventory_value': inventory.get('total_value', 0),
            'category_breakdown': inventory.get('category_distribution', {}),
            'age_analysis': inventory.get('inventory_age', {})
        }
    
    def _analyze_stock_levels(self, data):
        inventory = data.get('inventory_data', {})
        
        return {
            'optimal_stock_items': inventory.get('optimal_stock_count', 0),
            'overstock_items': inventory.get('overstock_count', 0),
            'understock_items': inventory.get('low_stock_items', 0),
            'stockout_items': inventory.get('out_of_stock_items', 0),
            'stock_health_index': self._calculate_stock_health_index(inventory)
        }
    
    def _generate_restock_recommendations(self, data):
        recommendations = []
        
        inventory = data.get('inventory_data', {})
        for product in inventory.get('restock_alerts', []):
            recommendations.append({
                'product': product.get('name', 'Unknown'),
                'urgency': product.get('urgency', 'medium'),
                'recommended_quantity': product.get('recommended_quantity', 0),
                'reason': product.get('reason', 'Low stock level')
            })
        
        return recommendations
    
    def _analyze_shelf_optimization(self, data):
        shelf_data = data.get('shelf_analysis', {})
        
        return {
            'occupancy_rates': shelf_data.get('occupancy_rates', {}),
            'performance_by_location': shelf_data.get('location_performance', {}),
            'optimization_opportunities': shelf_data.get('optimization_suggestions', [])
        }
    
    def _calculate_inventory_turnover(self, data):
        inventory = data.get('inventory_data', {})
        
        return {
            'overall_turnover': inventory.get('inventory_turnover', 0),
            'category_turnover': inventory.get('category_turnover', {}),
            'slow_moving_items': inventory.get('slow_movers', []),
            'fast_moving_items': inventory.get('fast_movers', [])
        }
    
    def _generate_demand_forecasts(self, data):
        forecasts = data.get('demand_forecasts', {})
        
        return {
            'short_term_forecast': forecasts.get('next_7_days', {}),
            'category_forecasts': forecasts.get('category_predictions', {}),
            'seasonal_adjustments': forecasts.get('seasonal_factors', {}),
            'forecast_confidence': forecasts.get('confidence_scores', {})
        }
    
    def _generate_performance_scorecard(self, data):
        metrics = data.get('performance_metrics', {})
        
        return {
            'financial_metrics': {
                'revenue_growth': metrics.get('revenue_growth', 0),
                'profit_margin': metrics.get('profit_margin', 0),
                'inventory_turnover': metrics.get('inventory_turnover', 0)
            },
            'customer_metrics': {
                'satisfaction_score': metrics.get('customer_satisfaction', 0),
                'retention_rate': metrics.get('retention_rate', 0),
                'acquisition_cost': metrics.get('acquisition_cost', 0)
            },
            'operational_metrics': {
                'conversion_rate': metrics.get('conversion_rate', 0),
                'average_transaction_value': metrics.get('avg_transaction_value', 0),
                'staff_efficiency': metrics.get('staff_efficiency', 0)
            }
        }
    
    def _analyze_kpis(self, data):
        kpis = data.get('kpi_data', {})
        
        analysis = {}
        for kpi, value in kpis.items():
            target = kpis.get(f'{kpi}_target', value * 1.1)
            status = 'above_target' if value >= target else 'below_target'
            variance = (value - target) / target if target != 0 else 0
            
            analysis[kpi] = {
                'current_value': value,
                'target_value': target,
                'variance': variance,
                'status': status
            }
        
        return analysis
    
    def _perform_comparative_analysis(self, data):
        comparison_data = data.get('comparison_data', {})
        
        return {
            'period_comparison': comparison_data.get('period_comparison', {}),
            'benchmark_comparison': comparison_data.get('benchmark_comparison', {}),
            'competitive_analysis': comparison_data.get('competitive_analysis', {})
        }
    
    def _identify_improvement_opportunities(self, data):
        opportunities = []
        
        kpi_analysis = data.get('kpi_analysis', {})
        for kpi, analysis in kpi_analysis.items():
            if analysis.get('status') == 'below_target':
                opportunities.append({
                    'area': kpi,
                    'current_performance': analysis.get('current_value', 0),
                    'target_performance': analysis.get('target_value', 0),
                    'improvement_gap': analysis.get('variance', 0),
                    'priority': 'high' if abs(analysis.get('variance', 0)) > 0.2 else 'medium'
                })
        
        return opportunities
    
    def _generate_action_plan(self, data):
        action_plan = []
        
        opportunities = data.get('improvement_opportunities', [])
        for opportunity in opportunities:
            action_plan.append({
                'objective': f"Improve {opportunity['area']}",
                'actions': self._generate_specific_actions(opportunity),
                'timeline': '30 days',
                'responsible_party': 'Store Manager',
                'success_metrics': [f"{opportunity['area']} reaches target level"]
            })
        
        return action_plan
    
    def _generate_specific_actions(self, opportunity):
        area = opportunity['area']
        
        action_templates = {
            'conversion_rate': [
                'Train staff on upselling techniques',
                'Optimize store layout for better product discovery',
                'Implement targeted promotions'
            ],
            'customer_satisfaction': [
                'Enhance customer service training',
                'Improve store cleanliness and organization',
                'Gather and act on customer feedback'
            ],
            'inventory_turnover': [
                'Implement demand forecasting system',
                'Optimize reorder points and quantities',
                'Liquidate slow-moving inventory'
            ]
        }
        
        return action_templates.get(area, ['Develop specific improvement strategies'])
    
    def _analyze_weekend_effect(self, data):
        daily_sales = data.get('daily_sales', [])
        if not daily_sales:
            return 'insufficient_data'
        
        weekday_sales = [day for day in daily_sales 
                        if datetime.fromisoformat(day['date']).weekday() < 5]
        weekend_sales = [day for day in daily_sales 
                        if datetime.fromisoformat(day['date']).weekday() >= 5]
        
        if not weekday_sales or not weekend_sales:
            return 'insufficient_data'
        
        avg_weekday = np.mean([day.get('daily_revenue', 0) for day in weekday_sales])
        avg_weekend = np.mean([day.get('daily_revenue', 0) for day in weekend_sales])
        
        if avg_weekday > 0:
            weekend_effect = (avg_weekend - avg_weekday) / avg_weekday
        else:
            weekend_effect = 0
        
        if weekend_effect > 0.3:
            return 'strong_weekend_effect'
        elif weekend_effect > 0.1:
            return 'moderate_weekend_effect'
        else:
            return 'minimal_weekend_effect'
    
    def _analyze_time_of_day_patterns(self, data):
        hourly_data = data.get('hourly_sales', {})
        if not hourly_data:
            return {}
        
        peak_hours = sorted(hourly_data.items(), key=lambda x: x[1].get('transaction_count', 0), reverse=True)[:3]
        
        return {
            'peak_hours': [hour for hour, _ in peak_hours],
            'revenue_distribution': hourly_data,
            'shopping_patterns': self._identify_shopping_patterns(hourly_data)
        }
    
    def _analyze_category_seasonality(self, data):
        category_trends = data.get('category_trends', {})
        
        seasonal_categories = {}
        for category, trends in category_trends.items():
            if trends.get('seasonality_score', 0) > 0.7:
                seasonal_categories[category] = {
                    'peak_season': trends.get('peak_season', 'unknown'),
                    'seasonality_strength': trends.get('seasonality_score', 0)
                }
        
        return seasonal_categories
    
    def _identify_cross_sell_opportunities(self, products):
        opportunities = []
        
        product_affinities = data.get('product_affinities', {})
        for product1, affinities in product_affinities.items():
            for product2, affinity in affinities.items():
                if affinity > 0.3 and product1 != product2:
                    opportunities.append({
                        'product_a': product1,
                        'product_b': product2,
                        'affinity_score': affinity,
                        'opportunity_type': 'cross_sell'
                    })
        
        return sorted(opportunities, key=lambda x: x['affinity_score'], reverse=True)[:10]
    
    def _generate_staffing_recommendations(self, data):
        traffic_patterns = data.get('customer_traffic', {})
        sales_data = data.get('sales_data', {})
        
        recommendations = []
        
        peak_hours = traffic_patterns.get('peak_hours', [])
        for hour in peak_hours:
            recommendations.append({
                'time': f"{hour}:00",
                'recommendation': 'Increase staff coverage',
                'reason': 'High customer traffic during this hour'
            })
        
        return recommendations
    
    def _suggest_layout_optimizations(self, data):
        layout_data = data.get('store_layout_analysis', {})
        
        suggestions = []
        
        bottlenecks = layout_data.get('bottlenecks', [])
        for bottleneck in bottlenecks:
            suggestions.append({
                'area': bottleneck.get('location', 'unknown'),
                'suggestion': 'Widen aisle or reposition displays',
                'expected_impact': 'Reduce congestion by 20-30%'
            })
        
        dead_zones = layout_data.get('dead_zones', [])
        for zone in dead_zones:
            suggestions.append({
                'area': zone.get('location', 'unknown'),
                'suggestion': 'Place promotional items or high-demand products',
                'expected_impact': 'Increase area utilization by 15-25%'
            })
        
        return suggestions
    
    def _suggest_inventory_improvements(self, data):
        inventory_data = data.get('inventory_analysis', {})
        
        suggestions = []
        
        slow_movers = inventory_data.get('slow_moving_items', [])
        for item in slow_movers[:5]:
            suggestions.append({
                'product': item.get('name', 'Unknown'),
                'suggestion': 'Consider discounting or promotional placement',
                'reason': 'Low inventory turnover'
            })
        
        return suggestions
    
    def _suggest_customer_experience_improvements(self, data):
        customer_feedback = data.get('customer_feedback', {})
        behavior_data = data.get('customer_behavior', {})
        
        suggestions = []
        
        if behavior_data.get('average_dwell_time', 0) > 1800:
            suggestions.append({
                'area': 'Checkout Process',
                'suggestion': 'Implement express checkout lanes',
                'reason': 'Long customer dwell times may indicate checkout delays'
            })
        
        return suggestions
    
    def _calculate_stock_health_index(self, inventory):
        total_items = inventory.get('total_products', 1)
        optimal_items = inventory.get('optimal_stock_count', 0)
        overstock_items = inventory.get('overstock_count', 0)
        understock_items = inventory.get('low_stock_items', 0)
        
        health_score = (optimal_items - 0.5 * overstock_items - 0.7 * understock_items) / total_items
        return max(0.0, min(1.0, health_score))
    
    def _identify_shopping_patterns(self, hourly_data):
        patterns = []
        
        morning_hours = [8, 9, 10, 11]
        afternoon_hours = [12, 13, 14, 15, 16]
        evening_hours = [17, 18, 19, 20]
        
        morning_sales = sum(hourly_data.get(hour, {}).get('transaction_count', 0) for hour in morning_hours)
        afternoon_sales = sum(hourly_data.get(hour, {}).get('transaction_count', 0) for hour in afternoon_hours)
        evening_sales = sum(hourly_data.get(hour, {}).get('transaction_count', 0) for hour in evening_hours)
        
        total_sales = morning_sales + afternoon_sales + evening_sales
        if total_sales > 0:
            if evening_sales / total_sales > 0.4:
                patterns.append('evening_shopping_dominant')
            elif afternoon_sales / total_sales > 0.4:
                patterns.append('afternoon_shopping_dominant')
            elif morning_sales / total_sales > 0.4:
                patterns.append('morning_shopping_dominant')
            else:
                patterns.append('balanced_shopping_pattern')
        
        return patterns

    def export_report(self, report, format='json', filename=None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_type = report.get('report_type', 'report')
            filename = f"{report_type}_{timestamp}.{format}"
        
        if format.lower() == 'json':
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        elif format.lower() == 'csv':
            self._export_csv(report, filename)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return filename
    
    def _export_csv(self, report, filename):
        flattened_data = self._flatten_report(report)
        df = pd.DataFrame([flattened_data])
        df.to_csv(filename, index=False)
    
    def _flatten_report(self, report, parent_key='', sep='_'):
        items = []
        for k, v in report.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_report(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        items.extend(self._flatten_report(item, f"{new_key}{sep}{i}", sep=sep).items())
                    else:
                        items.append((f"{new_key}{sep}{i}", item))
            else:
                items.append((new_key, v))
        return dict(items)