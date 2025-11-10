import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# ---------------------------
# Section 3: Vertical Solutions & Ecosystem Forecast
# ---------------------------
automotive_solutions = {
    'ADAS_Quality_Suite': {
        'target_chips': ['Radar SoC', 'Vision Processor', 'Sensor Fusion'],
        'key_features': [
            'Predictive reliability scoring',
            'Adaptive environmental stress testing', 
            'Cross-lot anomaly detection',
            'AEC-Q100 compliance automation'
        ],
        'integration_partners': ['Ansys', 'Synopsys', 'MathWorks'],
        'price_tier': 'Premium'
    },
    'EV_Powertrain_Optimizer': {
        'target_chips': ['Battery Management IC', 'Power Converter', 'Motor Controller'],
        'key_features': [
            'Intelligent power cycling test',
            'Lifetime aging prediction',
            'Thermal runaway early detection',
            'Production test correlation'
        ],
        'integration_partners': ['Siemens', 'NVIDIA', 'Keysight'],
        'price_tier': 'Enterprise' 
    },
    'Automotive_Microcontroller_Accelerator': {
        'target_chips': ['Safety MCU', 'Gateway Controller', 'Body Domain'],
        'key_features': [
            'Automated FMEDA test generation',
            'ISO-26262 compliance checking',
            'Multi-core interference testing',
            'Security vulnerability screening'
        ],
        'integration_partners': ['ARM', 'Green Hills', 'ETAS'],
        'price_tier': 'Standard'
    }
}

# Visual 1: Feature counts by solution
sol_names = []
feature_counts = []
partner_counts = []
for sol, d in automotive_solutions.items():
    sol_names.append(sol.replace('_',' '))
    feature_counts.append(len(d['key_features']))
    partner_counts.append(len(d['integration_partners']))

fig3, ax31 = plt.subplots(1,2, figsize=(10,4))
ax31[0].bar(sol_names, feature_counts)
ax31[0].set_title('Number of Key Features per Vertical Solution')
ax31[0].set_ylabel('Feature Count')
ax31[0].tick_params(axis='x', rotation=20)

ax31[1].bar(sol_names, partner_counts)
ax31[1].set_title('Number of Integration Partners per Solution')
ax31[1].set_ylabel('Partner Count')
ax31[1].tick_params(axis='x', rotation=20)
plt.tight_layout()
plt.show()

# Ecosystem expansion forecast (line trends)
class EcosystemExpansion:
    def __init__(self):
        pass
    def forecast_market_penetration(self):
        years = [2024, 2025, 2026]
        penetration = {
            'tier_1_safety': [0.05, 0.25, 0.60],
            'tier_2_ev': [0.08, 0.35, 0.70],
            'tier_3_infotainment': [0.12, 0.45, 0.80],
            'overall_automotive': [0.08, 0.35, 0.70]
        }
        automotive_test_market = 1.2  # $B
        teradyne_share = 0.4
        ai_premium = 0.3
        revenue_impact = []
        for year, penetration_rate in zip(years, penetration['overall_automotive']):
            revenue = automotive_test_market * teradyne_share * penetration_rate * (1 + ai_premium)
            revenue_impact.append(revenue)
        return years, penetration, revenue_impact

ecosystem = EcosystemExpansion()
years, penetration, revenue_impact = ecosystem.forecast_market_penetration()

# Plot penetration by tier
fig4, ax4 = plt.subplots(figsize=(8,4))
for tier, vals in penetration.items():
    ax4.plot(years, vals, marker='o', label=tier.replace('_',' ').title())
ax4.set_title('3-Year Penetration Forecast by Tier')
ax4.set_xlabel('Year')
ax4.set_ylabel('Penetration Rate')
ax4.legend()
plt.tight_layout()
plt.show()

# Plot revenue impact
fig4b, ax4b = plt.subplots(figsize=(6,3))
ax4b.bar(years, revenue_impact)
ax4b.set_title('3-Year Revenue Impact (illustrative $B)')
ax4b.set_ylabel('Revenue ($B)')
plt.tight_layout()
plt.show()