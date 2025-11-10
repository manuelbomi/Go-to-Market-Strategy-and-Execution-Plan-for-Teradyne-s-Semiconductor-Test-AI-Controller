import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# ---------------------------
# Section 2: Lighthouse Program (candidates + ROI visuals)
# ---------------------------
class LighthouseProgram:
    def __init__(self):
        self.participant_criteria = {
            'technical_readiness': ['AEC-Q100 qualified', 'ISO-26262 experience', 'Data infrastructure'],
            'business_commitment': ['Dedicated team', 'Executive sponsor', 'ROI measurement'],
            'strategic_alignment': ['Market leader', 'Innovation focus', 'Quality crisis']
        }
        self.program_benefits = {
            'pricing': '50% discount on first year',
            'support': 'Dedicated AI engineer on-site',
            'influence': 'Direct input to product roadmap',
            'marketing': 'Co-branded case studies and webinars'
        }
    def identify_lighthouse_candidates(self):
        candidates = [
            {
                'company': 'Bosch',
                'segment': 'Tier 1',
                'use_case': 'Radar SoC quality improvement',
                'current_dppm': 5,
                'target_dppm': 0.5,
                'test_time': 10,
                'strategic_value': 'High - market leader'
            },
            {
                'company': 'Tesla',
                'segment': 'Tier 2',
                'use_case': 'Battery management IC test time reduction',
                'current_dppm': 8,
                'target_dppm': 2,
                'test_time': 8,
                'strategic_value': 'High - innovation brand'
            },
            {
                'company': 'Continental',
                'segment': 'Tier 1',
                'use_case': 'Brake controller reliability prediction',
                'current_dppm': 3,
                'target_dppm': 0.1,
                'test_time': 12,
                'strategic_value': 'Medium - conservative adopter'
            }
        ]
        return candidates
    
    def calculate_program_roi(self, candidate):
        annual_volume = 2_000_000
        device_cost = 45
        test_cost_per_hour = 60
        test_time_reduction = 0.35
        dppm_improvement = 0.7
        false_failure_reduction = 0.4

        test_savings = (candidate['test_time'] * test_cost_per_hour * test_time_reduction * annual_volume)
        # quality savings estimate (warranty/recall avoidance) — conservative model:
        # current defects per year: current_dppm / 1e6 * annual_volume
        defects_avoided = candidate['current_dppm'] * dppm_improvement / 1e6 * annual_volume
        quality_savings = defects_avoided * device_cost * 50  # factor for warranty/recall etc (illustrative)
        total_annual_savings = test_savings + quality_savings
        ai_controller_cost = 1_500_000
        payback_months = ai_controller_cost / total_annual_savings * 12 if total_annual_savings>0 else np.inf

        return {
            'annual_savings': total_annual_savings,
            'payback_months': payback_months,
            'test_savings': test_savings,
            'quality_savings': quality_savings
        }

lighthouse = LighthouseProgram()
candidates = lighthouse.identify_lighthouse_candidates()

# Compute ROIs and make plots
roi_list = []
for cand in candidates:
    roi = lighthouse.calculate_program_roi(cand)
    roi['company'] = cand['company']
    roi['test_time'] = cand['test_time']
    roi['current_dppm'] = cand['current_dppm']
    roi_list.append(roi)

roi_df = pd.DataFrame(roi_list).set_index('company')

# Print summary
print("\nLIGHTHOUSE PROGRAM ROI SUMMARY")
print(roi_df[['annual_savings','payback_months']].to_string(float_format='${:,.0f}'.format))

# Visualization: stacked bar of savings breakdown
fig2, ax2 = plt.subplots(figsize=(8,4))
inds = np.arange(len(roi_df))
bar1 = ax2.bar(inds, roi_df['test_savings'], label='Test Savings')
bar2 = ax2.bar(inds, roi_df['quality_savings'], bottom=roi_df['test_savings'], label='Quality Savings')
ax2.set_xticks(inds)
ax2.set_xticklabels(roi_df.index)
ax2.set_ylabel('Annual Savings ($)')
ax2.set_title('Lighthouse Candidate Savings Breakdown')
ax2.legend()
plt.tight_layout()
plt.show()

# Waterfall-like: absolute values stacked, and payback months as secondary axis
fig2b, ax2b = plt.subplots(figsize=(8,4))
ax2b.bar(roi_df.index, roi_df['annual_savings'])
ax2b.set_ylabel('Total Annual Savings ($)')
ax2b.set_title('Total Annual Savings & Payback (Months)')
ax3 = ax2b.twinx()
ax3.plot(roi_df.index, roi_df['payback_months'], marker='o', color='orange', label='Payback (months)')
ax3.set_ylabel('Payback (months)')
ax3.legend(loc='upper right')
plt.tight_layout()
plt.show()
