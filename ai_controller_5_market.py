import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# ---------------------------
# Section 4: KPI Dashboard & Customer Success Visualization
# ---------------------------
class AutomotiveKPIDashboard:
    def __init__(self):
        self.kpis = {
            'business_metrics': {
                'ai_attach_rate_automotive': {'current': 0.05, 'target': 0.40},
                'automotive_revenue_growth': {'current': 0.08, 'target': 0.35},
                'deal_size_premium': {'current': 1.1, 'target': 1.4},
                'customer_satisfaction_auto': {'current': 65, 'target': 85}
            },
            'technical_metrics': {
                'test_time_reduction_auto': {'current': 0.15, 'target': 0.40},
                'dppm_improvement_customers': {'current': 0.3, 'target': 0.70},
                'adoption_rate_features': {'current': 0.20, 'target': 0.80},
                'inference_speed_auto': {'current': 25, 'target': 8}
            },
            'customer_metrics': {
                'lighthouse_references': {'current': 0, 'target': 5},
                'case_studies_published': {'current': 0, 'target': 12},
                'industry_awards': {'current': 0, 'target': 3},
                'standard_adoption': {'current': 0, 'target': 2}
            }
        }
    def calculate_adoption_score(self):
        total_score = 0
        max_score = 0
        for category, metrics in self.kpis.items():
            for metric, values in metrics.items():
                current = values['current']
                target = values['target']
                if target > 0:
                    # normalize across numeric scales: if metric looks like NPS (0-100), scale appropriately
                    if target > 1 and current > 1:  # likely percentage or absolute measure
                        score = min(current / target, 1.0) * 100
                    else:
                        score = min(current / target, 1.0) * 100
                    total_score += score
                    max_score += 100
        return (total_score / max_score) * 100 if max_score > 0 else 0

dashboard = AutomotiveKPIDashboard()
adoption_score = dashboard.calculate_adoption_score()
print(f"\nOverall Automotive Adoption Score: {adoption_score:.1f}/100")

# Prepare KPI progress bars (stacked horizontal bars)
def kpi_progress_df(kpis):
    rows = []
    for cat, metrics in kpis.items():
        for metric, vals in metrics.items():
            current = vals['current']
            target = vals['target']
            progress = min(current / target, 1.0) if target>0 else 0
            rows.append({'category':cat, 'metric':metric, 'current':current, 'target':target, 'progress':progress})
    return pd.DataFrame(rows)

kpi_df = kpi_progress_df(dashboard.kpis)
# plot grouped by category
fig5, ax5 = plt.subplots(figsize=(9,5))
categories = kpi_df['category'].unique()
y_pos = np.arange(len(kpi_df))
bars = ax5.barh(y_pos, kpi_df['progress'], align='center')
ax5.set_yticks(y_pos)
ax5.set_yticklabels(kpi_df['metric'])
ax5.set_xlim(0,1.05)
ax5.set_xlabel('Progress (0-100% of target)')
ax5.set_title('KPI Progress toward Targets (current/target)')
for i, b in enumerate(bars):
    ax5.text(b.get_width()+0.02, b.get_y()+b.get_height()/2, f"{kpi_df['progress'].iloc[i]*100:.0f}%", va='center')
plt.tight_layout()
plt.show()

# Customer Success Tracker visual (radar-like using polar bar)
class CustomerROITracker:
    def __init__(self):
        self.success_metrics = {
            'test_time_reduction': {'threshold': 0.25, 'weight': 0.3},
            'dppm_improvement': {'threshold': 0.5, 'weight': 0.4},
            'false_failure_reduction': {'threshold': 0.3, 'weight': 0.2},
            'engineer_productivity': {'threshold': 0.2, 'weight': 0.1}
        }
    def track_customer_success(self, customer_data):
        results = {}
        total_score = 0
        for metric, criteria in self.success_metrics.items():
            actual = customer_data.get(metric, 0)
            threshold = criteria['threshold']
            weight = criteria['weight']
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
        return results

example_customer = {
    'test_time_reduction': 0.35,
    'dppm_improvement': 0.65,
    'false_failure_reduction': 0.25,
    'engineer_productivity': 0.30
}

tracker = CustomerROITracker()
results = tracker.track_customer_success(example_customer)

# Radar-style polar bar (simple)
metrics = list(tracker.success_metrics.keys())
achievements = [results[m]['achievement'] for m in metrics]
angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
fig6 = plt.figure(figsize=(6,5))
ax6 = fig6.add_subplot(111, polar=True)
bars = ax6.bar(angles, achievements, width=0.9*(2*np.pi/len(metrics)), bottom=0.0)
ax6.set_xticks(angles)
ax6.set_xticklabels([m.replace('_',' ').title() for m in metrics])
ax6.set_title('Customer Success Achievement (0-1 scale)')
for r, bar in zip(achievements, bars):
    ax6.text(bar.get_x()+bar.get_width()/2, r+0.03, f"{r*100:.0f}%", ha='center')
plt.tight_layout()
plt.show()