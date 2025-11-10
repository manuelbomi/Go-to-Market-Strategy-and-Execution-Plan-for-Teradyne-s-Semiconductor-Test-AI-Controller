# ------------------------------------------
# SECTION 5: SALES ROI CALCULATOR WITH VISUALS
# ------------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.ticker as mtick

class AutomotiveROICalculator:
    def __init__(self):
        self.industry_benchmarks = {
            'test_cost_per_hour': 75,
            'warranty_cost_per_failure': 2500,
            'recall_cost_multiplier': 100,
            'engineering_cost_per_hour': 120
        }
    
    def calculate_customer_roi(self, customer_profile):
        """Calculate ROI for automotive AI test automation investment"""
        annual_volume = customer_profile['annual_volume']
        current_test_time = customer_profile['current_test_time']
        current_dppm = customer_profile['current_dppm']
        device_cost = customer_profile['device_cost']

        # AI Test Controller Expected Benefits
        ai_benefits = {
            'test_time_reduction': 0.35,
            'dppm_improvement': 0.60,
            'false_failure_reduction': 0.25,
            'engineering_efficiency': 0.30
        }

        # Calculate benefit categories
        test_time_savings = (current_test_time * self.industry_benchmarks['test_cost_per_hour'] *
                             ai_benefits['test_time_reduction'] * annual_volume)

        quality_savings = (current_dppm * ai_benefits['dppm_improvement'] / 1e6 *
                           annual_volume * self.industry_benchmarks['warranty_cost_per_failure'])

        false_failure_savings = (ai_benefits['false_failure_reduction'] * annual_volume *
                                 device_cost * 0.01)

        engineering_savings = (customer_profile.get('engineering_hours', 2000) *
                               self.industry_benchmarks['engineering_cost_per_hour'] *
                               ai_benefits['engineering_efficiency'])

        total_annual_savings = (test_time_savings + quality_savings +
                                false_failure_savings + engineering_savings)

        # Investment and ROI computations
        ai_system_cost = 1_800_000
        annual_maintenance = 180_000
        first_year_roi = total_annual_savings - ai_system_cost - annual_maintenance
        ongoing_annual_roi = total_annual_savings - annual_maintenance
        payback_months = (ai_system_cost / total_annual_savings) * 12 if total_annual_savings > 0 else np.inf

        return {
            'total_annual_savings': total_annual_savings,
            'first_year_roi': first_year_roi,
            'ongoing_annual_roi': ongoing_annual_roi,
            'payback_months': payback_months,
            'savings_breakdown': {
                'Test Time Reduction': test_time_savings,
                'Quality Improvement': quality_savings,
                'False Failure Reduction': false_failure_savings,
                'Engineering Efficiency': engineering_savings
            },
            'ai_system_cost': ai_system_cost,
            'annual_maintenance': annual_maintenance
        }


# ------------------------------------------
# EXECUTION: EXAMPLE AUTOMOTIVE CUSTOMER
# ------------------------------------------

calculator = AutomotiveROICalculator()
example_automotive_customer = {
    'company': 'Major Tier 1 Supplier',
    'annual_volume': 5_000_000,
    'current_test_time': 10,
    'current_dppm': 8,
    'device_cost': 85,
    'engineering_hours': 5000
}
roi_results = calculator.calculate_customer_roi(example_automotive_customer)

# Console summary
print("\nAUTOMOTIVE CUSTOMER ROI ANALYSIS")
print("=" * 50)
print(f"Total Annual Savings: ${roi_results['total_annual_savings']:,.0f}")
print(f"First-Year ROI: ${roi_results['first_year_roi']:,.0f}")
print(f"Ongoing Annual ROI: ${roi_results['ongoing_annual_roi']:,.0f}")
print(f"Payback Period: {roi_results['payback_months']:.1f} months")
print("=" * 50)

# ------------------------------------------
# VISUALIZATIONS
# ------------------------------------------

sns.set(style="whitegrid", palette="deep", font_scale=1.1)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Teradyne Automotive AI Test Controller – ROI Visualization", fontsize=15, weight='bold')

# --- 1. Savings Breakdown (Pie) ---
labels = list(roi_results['savings_breakdown'].keys())
vals = list(roi_results['savings_breakdown'].values())
axes[0, 0].pie(vals, labels=labels, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("Paired"))
axes[0, 0].set_title("Savings Breakdown (Proportion)")

# --- 2. Savings Breakdown (Bar) ---
sns.barplot(x=labels, y=vals, ax=axes[0, 1], palette="Paired")
axes[0, 1].set_title("Savings Breakdown (Absolute $)")
axes[0, 1].set_ylabel("USD")
axes[0, 1].tick_params(axis='x', rotation=25)
for i, v in enumerate(vals):
    axes[0, 1].text(i, v + max(vals)*0.01, f"${v/1e6:.2f}M", ha='center', fontsize=10)

# --- 3. Waterfall Chart: ROI Components ---
categories = ['AI System Cost', 'Annual Maintenance', 'Total Savings', 'First-Year ROI']
values = [-roi_results['ai_system_cost'], -roi_results['annual_maintenance'], 
          roi_results['total_annual_savings'], roi_results['first_year_roi']]
bars = sns.barplot(x=categories, y=values, ax=axes[1, 0], palette=['#E74C3C', '#E67E22', '#2ECC71', '#3498DB'])
axes[1, 0].set_title("First-Year ROI Composition")
axes[1, 0].set_ylabel("USD")
axes[1, 0].tick_params(axis='x', rotation=15)
for i, v in enumerate(values):
    axes[1, 0].text(i, v + (0.05 * max(values)), f"${v/1e6:.2f}M", ha='center', fontsize=9)

# --- 4. KPI Gauge: Payback Period ---
payback = roi_results['payback_months']
axes[1, 1].axis('off')
axes[1, 1].set_xlim(0, 1)
axes[1, 1].set_ylim(0, 1)
axes[1, 1].text(0.5, 0.6, f"{payback:.1f} mo", ha='center', fontsize=30, weight='bold', color='#2ECC71' if payback <= 12 else '#F39C12')
axes[1, 1].text(0.5, 0.35, "Payback Period", ha='center', fontsize=13)
axes[1, 1].text(0.5, 0.2, "Goal: ≤ 12 months", ha='center', fontsize=10, color='gray')
circle = plt.Circle((0.5, 0.55), 0.25, color='#2ECC71' if payback <= 12 else '#F39C12', alpha=0.15)
axes[1, 1].add_artist(circle)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

print("\nAll ROI visualizations generated successfully.")
