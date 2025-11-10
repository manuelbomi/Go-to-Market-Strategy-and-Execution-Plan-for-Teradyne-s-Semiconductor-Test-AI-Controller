import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# ---------------------------------------------
# CUSTOMER ROI VALIDATION FRAMEWORK WITH VISUALS
# ---------------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class CustomerROITracker:
    def __init__(self):
        self.success_metrics = {
            'test_time_reduction': {'threshold': 0.25, 'weight': 0.3},
            'dppm_improvement': {'threshold': 0.5, 'weight': 0.4},
            'false_failure_reduction': {'threshold': 0.3, 'weight': 0.2},
            'engineer_productivity': {'threshold': 0.2, 'weight': 0.1}
        }
    
    def track_customer_success(self, customer_data):
        """Track and validate customer success metrics"""
        results = {}
        total_score = 0
        
        for metric, criteria in self.success_metrics.items():
            actual = customer_data.get(metric, 0)
            threshold = criteria['threshold']
            weight = criteria['weight']
            
            # Calculate metric achievement
            achievement = min(actual / threshold, 1.0) if threshold > 0 else 0
            metric_score = achievement * weight * 100
            total_score += metric_score
            
            results[metric] = {
                'actual': actual,
                'threshold': threshold, 
                'achievement': achievement,
                'score': metric_score
            }
        
        results['total_score'] = total_score
        results['success_tier'] = self._determine_success_tier(total_score)
        
        return results
    
    def _determine_success_tier(self, score):
        if score >= 90:
            return 'Reference Customer'
        elif score >= 75:
            return 'Successful Deployment'
        elif score >= 60:
            return 'Moderate Success'
        else:
            return 'Needs Improvement'
    
    def visualize_results(self, results):
        """Generate ROI visualizations for customer success metrics"""
        # Filter out summary keys
        metrics = [m for m in results.keys() if m not in ['total_score', 'success_tier']]
        
        actuals = [results[m]['actual'] for m in metrics]
        thresholds = [results[m]['threshold'] for m in metrics]
        scores = [results[m]['score'] for m in metrics]
        achievements = [results[m]['achievement'] * 100 for m in metrics]
        
        sns.set(style="whitegrid", palette="deep", font_scale=1.1)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # --- 1. Actual vs Target Comparison ---
        x = np.arange(len(metrics))
        width = 0.35
        axes[0].bar(x - width/2, actuals, width, label='Actual', color='#4C72B0')
        axes[0].bar(x + width/2, thresholds, width, label='Target', color='#55A868')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=30)
        axes[0].set_ylabel('Improvement Ratio')
        axes[0].set_title('Actual vs Target Performance')
        axes[0].legend()
        
        # --- 2. Weighted Metric Scores ---
        sns.barplot(x=[m.replace('_', ' ').title() for m in metrics],
                    y=scores, ax=axes[1], color='#8172B2')
        axes[1].set_ylabel('Weighted Score Contribution')
        axes[1].set_title('Weighted ROI Metric Scores')
        axes[1].set_ylim(0, 40)
        for i, score in enumerate(scores):
            axes[1].text(i, score + 1, f"{score:.1f}", ha='center', fontsize=10)
        
        # --- 3. Overall Achievement Summary ---
        colors = {'Reference Customer':'#2ECC71', 
                  'Successful Deployment':'#F1C40F', 
                  'Moderate Success':'#E67E22',
                  'Needs Improvement':'#E74C3C'}
        tier = results['success_tier']
        axes[2].barh([''], [results['total_score']], color=colors[tier])
        axes[2].set_xlim(0, 100)
        axes[2].set_title('Overall Customer ROI Score')
        axes[2].text(results['total_score'] + 1, 0, 
                     f"{results['total_score']:.1f}/100\n({tier})",
                     va='center', fontsize=12, weight='bold')
        axes[2].set_yticks([])
        axes[2].axvline(60, color='gray', linestyle='--', lw=1)
        axes[2].axvline(75, color='gray', linestyle='--', lw=1)
        axes[2].axvline(90, color='gray', linestyle='--', lw=1)
        axes[2].text(60, -0.25, 'Moderate', fontsize=8)
        axes[2].text(75, -0.25, 'Success', fontsize=8)
        axes[2].text(90, -0.25, 'Reference', fontsize=8)
        
        plt.tight_layout()
        plt.show()


# -------------------------------
# EXECUTION EXAMPLE
# -------------------------------

tracker = CustomerROITracker()

example_customer = {
    'test_time_reduction': 0.35,  # 35% improvement
    'dppm_improvement': 0.65,     # 65% better
    'false_failure_reduction': 0.25,  # 25% reduction
    'engineer_productivity': 0.30  # 30% more productive
}

results = tracker.track_customer_success(example_customer)

print("CUSTOMER SUCCESS VALIDATION")
print("=" * 40)
for metric, data in results.items():
    if metric not in ['total_score', 'success_tier']:
        print(f"{metric.replace('_', ' ').title()}:")
        print(f"  Actual: {data['actual']:.1%} (Target: {data['threshold']:.1%})")
        print(f"  Achievement: {data['achievement']:.1%}")
        print(f"  Score: {data['score']:.1f}")

print(f"\nOverall Success Score: {results['total_score']:.1f}/100")
print(f"Tier: {results['success_tier']}")

# Visualize performance
tracker.visualize_results(results)
