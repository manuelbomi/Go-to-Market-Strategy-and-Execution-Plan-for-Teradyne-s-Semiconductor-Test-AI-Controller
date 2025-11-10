import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams['figure.dpi'] = 120

# ---------------------------
# Section 1: Customer Segmentation
# ---------------------------
customer_segments = {
    'Tier 1: Safety-Critical Leaders': {
        'Examples': ['Bosch', 'Continental', 'Denso', 'ZF'],
        'Focus': 'ADAS, braking, steering chips',
        'DPPM Target': '0.1-1 DPPM',
        'Test Time Pain': '8-12 hours/device',
        'Value Driver': 'Quality leadership, liability reduction',
        'Deal Size': '$5-20M',
        'Sales Cycle': '9-12 months'
    },
    'Tier 2: EV/Powertrain Focused': {
        'Examples': ['Tesla', 'LG Magna', 'Marelli', 'Vitesco'],
        'Focus': 'Battery management, power conversion',
        'DPPM Target': '1-5 DPPM', 
        'Test Time Pain': '6-10 hours/device',
        'Value Driver': 'Time-to-market, cost reduction',
        'Deal Size': '$3-10M',
        'Sales Cycle': '6-9 months'
    },
    'Tier 3: Infotainment & Comfort': {
        'Examples': ['Harman', 'Alpine', 'Panasonic Automotive', 'Visteon'],
        'Focus': 'Displays, audio, connectivity',
        'DPPM Target': '10-50 DPPM',
        'Test Time Pain': '4-8 hours/device',
        'Value Driver': 'Cost reduction, competitive pricing',
        'Deal Size': '$1-5M',
        'Sales Cycle': '3-6 months'
    }
}

segment_df = pd.DataFrame(customer_segments).T

# Preprocess numeric fields
def first_number_from_range(s):
    # extract first numeric token
    return float(s.split('-')[0].replace('$', '').replace('M', '').replace(' DPPM','').strip())

sizes = [first_number_from_range(x) for x in segment_df['Deal Size']]
cycles = [int(x.split('-')[0]) for x in segment_df['Sales Cycle']]
dppm_vals = [float(x.split('-')[0]) for x in segment_df['DPPM Target']]
test_times = [float(x.split('-')[0]) for x in segment_df['Test Time Pain']]

# Additional data for heatmap: value drivers numeric encoding
value_drivers = pd.DataFrame({
    'Quality Leadership': [1.0, 0.7, 0.3],
    'Cost Reduction': [0.4, 0.9, 1.0],
    'Time-to-Market': [0.6, 0.8, 0.7]
}, index=segment_df.index)

fig1, axes = plt.subplots(2, 2, figsize=(12, 9))
# Plot 1: Deal Size vs Sales Cycle scatter
axes[0,0].scatter(sizes, cycles, s=[300, 300, 300], c=[0,1,2], cmap='viridis')
axes[0,0].set_xlabel('Deal Size ($M, first of range)')
axes[0,0].set_ylabel('Sales Cycle (Months, lower bound)')
axes[0,0].set_title('Deal Size vs Sales Cycle by Segment')
for i, seg in enumerate(segment_df.index):
    axes[0,0].annotate(seg.split(':')[0], (sizes[i], cycles[i]), xytext=(6,6), textcoords='offset points')

# Plot 2: DPPM Target vs Test Time scatter
axes[0,1].scatter(dppm_vals, test_times, s=[300,300,300], c=[0,1,2], cmap='plasma')
axes[0,1].set_xlabel('DPPM Target (lower is better)')
axes[0,1].set_ylabel('Test Time (hrs, lower bound)')
axes[0,1].set_title('Quality vs Test Time Burden')
for i, seg in enumerate(segment_df.index):
    axes[0,1].annotate(seg.split(':')[0], (dppm_vals[i], test_times[i]), xytext=(6,6), textcoords='offset points')

# Plot 3: Value Driver bar chart (grouped)
value_drivers.plot(kind='bar', ax=axes[1,0])
axes[1,0].set_title('Primary Value Drivers by Segment')
axes[1,0].set_ylabel('Importance (0-1)')
axes[1,0].tick_params(axis='x', rotation=30)

# Plot 4: Market opportunity pie (example opportunity sizes in $B)
opportunity = [20, 15, 8]
axes[1,1].pie(opportunity, labels=[s.split(':')[0] for s in segment_df.index], autopct='%1.0f%%', startangle=90)
axes[1,1].set_title('Total Addressable Market by Segment ($B - illustrative)')

plt.tight_layout()
plt.show()

# Heatmap of value drivers
fig_h, ax_h = plt.subplots(figsize=(6,3))
im = ax_h.imshow(value_drivers.values, aspect='auto', cmap='YlGnBu')
ax_h.set_xticks(np.arange(value_drivers.shape[1]))
ax_h.set_xticklabels(value_drivers.columns, rotation=30, ha='right')
ax_h.set_yticks(np.arange(value_drivers.shape[0]))
ax_h.set_yticklabels([s.split(':')[0] for s in value_drivers.index])
ax_h.set_title('Value Driver Importance Heatmap (0-1)')
cbar = plt.colorbar(im, ax=ax_h)
plt.tight_layout()
plt.show()