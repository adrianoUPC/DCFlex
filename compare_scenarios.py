# %% Scenario Comparison Script
"""
This script creates comprehensive visualization plots comparing energy metrics
across three LAD-FLEX scenarios (BASELINE, DAM, mFRR) for ZETA=0 (no task duration uncertainty).

Outputs:
- Boxplot comparison (5 metrics)
- Bar chart comparison (mean ± std)
- Combined publication-ready figure
- Time series overlay
- Percentage breakdown stacked bar chart
- Notable points comparison (9 points: P1-P3, Q1-Q3, R1-R3)
- Hourly efficiency comparison (energy per hour of flexibility window)
- Summary statistics CSV table
"""

# %% Section 1: Imports and Configuration
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

# Configuration
SIMULATION = "DEFAULT"
ZETA = 0
zeta_suffix = "ZETA_0"

# Scenario list
scenarios = ['BASELINE', 'DAM', 'mFRR']

# Colorblind-friendly color scheme
SCENARIO_COLORS = {
    'BASELINE': '#0571B0',  # Blue
    'DAM': '#FC9272',       # Soft red
    'mFRR': '#A6D854'       # Soft green
}

# Scenario labels with parameters
SCENARIO_LABELS = {
    'BASELINE': 'BASELINE\n(600s, 3h, β=0.7)',
    'DAM': 'DAM\n(1800s, 4h, β=0.8)',
    'mFRR': 'mFRR\n(180s, 1h, β=0.9)'
}

# Short display labels for plots/legends (DAM is shown as INTRADAY)
DISPLAY_LABELS = {
    'BASELINE': 'BASELINE',
    'DAM': 'INTRADAY',
    'mFRR': 'mFRR'
}

# Sample week date range
SAMPLE_WEEK_START = pd.Timestamp('1970-02-01')
SAMPLE_WEEK_END = pd.Timestamp('1970-02-08')

# Output directory
OUTPUT_DIR = 'PLOTS/SCENARIO_COMPARISON'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Metric information
METRIC_INFO = {
    'deferrable_energy_kWh': {'label': 'Deferrable Energy (kWh)', 'short': 'Deferrable Energy'},
    'total_energy_kWh': {'label': 'Total Energy (kWh)', 'short': 'Total Energy'},
    'percentage_flexible_energy': {'label': 'Flexible Energy (%)', 'short': 'Flexible %'},
    'deferrable_task_count': {'label': 'Deferrable Task Count', 'short': 'Task Count'},
    'rebound_energy_kWh': {'label': 'Rebound Energy (kWh)', 'short': 'Rebound Energy'}
}

METRICS = list(METRIC_INFO.keys())

# %% Section 2: Data Loading
print("="*70)
print("Loading scenario data...")
print("="*70)

scenario_data = {}
scenario_sample_week = {}

for scenario in scenarios:
    filename = f'EXPORTS/df_summary_{SIMULATION}_{scenario}_{zeta_suffix}.joblib'
    print(f"Loading {scenario}: {filename}")
    df = joblib.load(filename)
    scenario_data[scenario] = df

    # Filter to sample week
    df_week = df.loc[SAMPLE_WEEK_START:SAMPLE_WEEK_END]
    scenario_sample_week[scenario] = df_week
    print(f"  - Full data shape: {df.shape}")
    print(f"  - Sample week shape: {df_week.shape}")

print("\nData loading complete!")
print("="*70)

# %% Section 3: Helper Functions

def create_combined_figure(scenario_data, metrics, save_path, save_pdf=True):
    """
    Create combined publication-ready figure with boxplots and bar charts.
    Focus on 3 key metrics: Deferrable Energy, % Flexible, Task Count

    Parameters:
    -----------
    scenario_data : dict
        Dictionary with {scenario_name: df_summary_filtered}
    metrics : list
        List of 3 key metric column names
    save_path : str
        Path to save output PNG file (without extension)
    save_pdf : bool
        Whether to also save PDF version
    """
    sns.set(style="whitegrid")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Compute statistics (kept for possible downstream use)
    stats = {}
    for scenario in ['BASELINE', 'DAM', 'mFRR']:
        stats[scenario] = {}
        for metric in metrics:
            stats[scenario][metric] = {
                'mean': scenario_data[scenario][metric].mean(),
                'std': scenario_data[scenario][metric].std()
            }

    # Single row: Boxplots only
    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # Prepare data for boxplot
        data_to_plot = []
        labels = []
        colors = []

        for scenario in ['BASELINE', 'DAM', 'mFRR']:
            data_to_plot.append(scenario_data[scenario][metric].values)
            labels.append(DISPLAY_LABELS.get(scenario, scenario))
            colors.append(SCENARIO_COLORS[scenario])

        # Create boxplot
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                        widths=0.5, showfliers=True, flierprops={'markersize': 4})

        # Color boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Color whiskers and caps
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(2)

        # Styling
        ax.set_title(METRIC_INFO[metric]['short'], fontsize=20)
        # Per-metric y-axis label
        if metric == 'deferrable_energy_kWh':
            ylabel = 'Energy (kWh)'
        elif metric == 'deferrable_task_count':
            ylabel = 'n_tasks'
        elif metric == 'percentage_flexible_energy':
            ylabel = 'Flexible (%)'
        else:
            ylabel = 'Value'

        ax.set_ylabel(ylabel, fontsize=18)
        ax.tick_params(axis='both', labelsize=16)
        ax.grid(axis='y', alpha=0.3)

        # Thicker spines
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.2)

    # Improve layout and x-tick styling
    for ax in axes:
        # Bold & larger x tick labels
        xticklabels = [t.get_text() for t in ax.get_xticklabels()]
        ax.set_xticklabels(xticklabels, fontsize=18, fontweight='bold')

    # Suptitle and layout: reserve top space so title doesn't overlap
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.subplots_adjust(left=0.08, top=0.94)
    plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
    if save_pdf:
        plt.savefig(f'{save_path}.pdf', bbox_inches='tight')
    plt.show()

    sns.reset_defaults()
    print(f"Combined figure saved to {save_path}.png/.pdf")


def create_time_series_overlay(scenario_data, metrics, save_path, save_pdf=True):
    """
    Create time series overlay showing hourly evolution across scenarios.

    Parameters:
    -----------
    scenario_data : dict
        Dictionary with {scenario_name: df_summary_filtered}
    metrics : list
        List of 3 metric column names to plot
    save_path : str
        Path to save output PNG file (without extension)
    save_pdf : bool
        Whether to also save PDF version
    """
    sns.set(style="whitegrid")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        for scenario in ['BASELINE', 'DAM', 'mFRR']:
            # Group by timestamp and compute mean/std
            df = scenario_data[scenario].reset_index()
            grouped = df.groupby('timestamp')[metric].agg(['mean', 'std'])

            timestamps = grouped.index
            means = grouped['mean'].values
            stds = grouped['std'].values

            # Plot line
            ax.plot(timestamps, means, label=scenario, color=SCENARIO_COLORS[scenario],
                   linewidth=2.5, alpha=0.9)

            # Plot shaded std band
            ax.fill_between(timestamps, means - stds, means + stds,
                           color=SCENARIO_COLORS[scenario], alpha=0.2)

        # Formatting
        ax.set_title(METRIC_INFO[metric]['short'], fontsize=16, fontweight='bold')

        # Per-metric y-axis label
        if metric == 'deferrable_energy_kWh':
            ylabel = 'Energy (kWh)'
        elif metric == 'deferrable_task_count':
            ylabel = 'n_tasks'
        elif metric == 'percentage_flexible_energy':
            ylabel = 'Flexible (%)'
        else:
            ylabel = 'Value'

        ax.set_ylabel(ylabel, fontsize=18)
        ax.tick_params(axis='both', labelsize=16)
        # ax.legend(fontsize=12, loc='best')
        ax.grid(alpha=0.3)

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, fontsize=16)

        # Thicker spines
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.2)

    # Reserve top space for suptitle
    # Reserve top/bottom space for suptitle and shared legend
    plt.tight_layout(rect=[0, 0.06, 1, 0.92])

    # Shared legend at the bottom (3 columns)
    handles = [Line2D([0], [0], color=SCENARIO_COLORS[s], lw=3) for s in ['BASELINE', 'DAM', 'mFRR']]
    labels = [DISPLAY_LABELS.get(s, s) for s in ['BASELINE', 'DAM', 'mFRR']]
    fig.subplots_adjust(bottom=0.17)
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=20, frameon=False)
    plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
    if save_pdf:
        plt.savefig(f'{save_path}.pdf', bbox_inches='tight')
    plt.show()

    sns.reset_defaults()
    print(f"Time series overlay saved to {save_path}.png/.pdf")


def create_summary_table(scenario_data, metrics, save_path):
    """
    Create summary statistics table and export to CSV.

    Parameters:
    -----------
    scenario_data : dict
        Dictionary with {scenario_name: df_summary_filtered}
    metrics : list
        List of metric column names
    save_path : str
        Path to save CSV file
    """
    rows = []

    for scenario in ['BASELINE', 'DAM', 'mFRR']:
        for metric in metrics:
            data = scenario_data[scenario][metric]
            rows.append({
                'Scenario': scenario,
                'Metric': METRIC_INFO[metric]['short'],
                'Mean': data.mean(),
                'Std': data.std(),
                'Min': data.min(),
                'Q25': data.quantile(0.25),
                'Median': data.median(),
                'Q75': data.quantile(0.75),
                'Max': data.max(),
                'Count': data.count()
            })

    df_table = pd.DataFrame(rows)
    df_table.to_csv(save_path, index=False, float_format='%.4f')

    print(f"\nSummary statistics table saved to {save_path}")
    print("\nPreview:")
    print(df_table.head(10).to_string(index=False))

    return df_table


def create_notable_points_comparison(scenario_data_full, save_path, save_pdf=True):
    """
    Compare the 9 notable points (P1-P3, Q1-Q3, R1-R3) across all scenarios.

    Parameters:
    -----------
    scenario_data_full : dict
        Dictionary with {scenario_name: df_summary_full_data} (NOT filtered to sample week)
    save_path : str
        Path to save output PNG file (without extension)
    save_pdf : bool
        Whether to also save PDF version
    """
    sns.set(style="whitegrid")

    # Load the notable points for each scenario
    points_data = {}
    point_names = ['P1', 'P2', 'P3', 'Q1', 'Q2', 'Q3', 'R1', 'R2', 'R3']

    for scenario in ['BASELINE', 'DAM', 'mFRR']:
        try:
            points_dates_file = f"EXPORTS/points_dates_{SIMULATION}_{scenario}_{zeta_suffix}.joblib"
            points_dates = joblib.load(points_dates_file)

            # Extract deferrable energy for each point
            points_data[scenario] = {}
            for point_name in point_names:
                point_key = point_name.lower()
                if point_key in points_dates:
                    timestamp = points_dates[point_key]
                    # Get the deferrable energy for this timestamp
                    df = scenario_data_full[scenario]

                    # Filter by timestamp and extract the value
                    df_filtered = df[df.index.get_level_values('timestamp') == timestamp]
                    if len(df_filtered) > 0:
                        point_value = float(df_filtered['deferrable_energy_kWh'].iloc[0])
                    else:
                        point_value = 0.0

                    points_data[scenario][point_name] = point_value
                else:
                    points_data[scenario][point_name] = 0.0
        except FileNotFoundError:
            print(f"Warning: Notable points file not found for {scenario}")
            points_data[scenario] = {p: 0.0 for p in point_names}
        except Exception as e:
            print(f"Warning: Error loading points for {scenario}: {e}")
            points_data[scenario] = {p: 0.0 for p in point_names}

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(16, 7))

    x = np.arange(len(point_names))
    width = 0.25

    # Plot bars for each scenario
    for i, scenario in enumerate(['BASELINE', 'DAM', 'mFRR']):
        offset = (i - 1) * width
        values = [points_data[scenario][p] for p in point_names]

        bars = ax.bar(x + offset, values, width,
                      label=scenario, color=SCENARIO_COLORS[scenario],
                      alpha=0.8, edgecolor='black', linewidth=1.5)

        # Add value labels on bars (only if value > 1)
        for j, (bar, val) in enumerate(zip(bars, values)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=12)

    # Formatting
    ax.set_xlabel('Notable Points', fontsize=18)
    ax.set_ylabel('Deferrable Energy (kWh)', fontsize=18)
    ax.set_title('Comparison of 9 Notable Points Across Scenarios\n(P1-P3: Top flexibility | Q1-Q3: 75th percentile | R1-R3: 50th percentile)',
                 fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(point_names, fontsize=18)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(fontsize=20, loc='upper right', framealpha=0.95)
    ax.grid(axis='y', alpha=0.3)

    # Add separator lines between P, Q, R groups
    ax.axvline(2.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.axvline(5.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)

    # Thicker spines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.2)

    plt.tight_layout()
    plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
    if save_pdf:
        plt.savefig(f'{save_path}.pdf', bbox_inches='tight')
    plt.show()

    sns.reset_defaults()
    print(f"Notable points comparison saved to {save_path}.png/.pdf")


def create_hourly_efficiency_plot(scenario_data, save_path, save_pdf=True):
    """
    Create bar chart showing energy per hour of flexibility window.
    E_norm = (Average Deferrable Energy) / (Window Duration in hours)

    Parameters:
    -----------
    scenario_data : dict
        Dictionary with {scenario_name: df_summary_filtered}
    save_path : str
        Path to save output PNG file (without extension)
    save_pdf : bool
        Whether to also save PDF version
    """
    sns.set(style="whitegrid")

    # Flexibility window durations (in hours)
    flex_window_hours = {
        'BASELINE': 3,  # 3 hours
        'DAM': 4,       # 4 hours
        'mFRR': 1       # 1 hour
    }

    # Calculate metrics
    efficiency_data = {}
    for scenario in ['BASELINE', 'DAM', 'mFRR']:
        avg_deferrable = scenario_data[scenario]['deferrable_energy_kWh'].mean()
        window_hours = flex_window_hours[scenario]
        efficiency = avg_deferrable / window_hours

        efficiency_data[scenario] = {
            'avg_deferrable': avg_deferrable,
            'window_hours': window_hours,
            'efficiency_kWh_per_h': efficiency
        }

    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 7))

    scenarios_list = ['BASELINE', 'DAM', 'mFRR']
    efficiencies = [efficiency_data[s]['efficiency_kWh_per_h'] for s in scenarios_list]
    colors = [SCENARIO_COLORS[s] for s in scenarios_list]

    bars = ax.bar(scenarios_list, efficiencies, color=colors, alpha=0.8,
                  edgecolor='black', linewidth=2)

    # Add value annotations
    for i, (scenario, bar) in enumerate(zip(scenarios_list, bars)):
        eff = efficiency_data[scenario]['efficiency_kWh_per_h']
        avg_def = efficiency_data[scenario]['avg_deferrable']
        window = efficiency_data[scenario]['window_hours']

        # Efficiency value on top of bar
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
               f'{eff:.2f} kWh/h', ha='center', va='bottom',
               fontsize=14)


    # Highlight highest efficiency
    max_idx = np.argmax(efficiencies)
    bars[max_idx].set_linewidth(3)
    bars[max_idx].set_edgecolor('darkgreen')

    # Formatting
    ax.set_ylabel('Energy per Hour of Flexibility Window (kWh/h)', fontsize=16)
    ax.set_title('Hourly Efficiency Comparison',
                 fontsize=18)
    ax.tick_params(axis='both', labelsize=14)
    # make xticks larger
    ax.set_xticklabels(scenarios_list, fontsize=16)
    
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(efficiencies) * 1.3)

    # Thicker spines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.2)

    plt.tight_layout()
    plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
    if save_pdf:
        plt.savefig(f'{save_path}.pdf', bbox_inches='tight')
    plt.show()

    sns.reset_defaults()
    print(f"Hourly efficiency plot saved to {save_path}.png/.pdf")

    # Print key insight
    print("\n" + "="*70)
    print("HOURLY EFFICIENCY ANALYSIS")
    print("="*70)
    for scenario in scenarios_list:
        data = efficiency_data[scenario]
        print(f"{scenario:10s}: {data['avg_deferrable']:6.1f} kWh / {data['window_hours']} h = {data['efficiency_kWh_per_h']:.2f} kWh/h")
    print("="*70)
    print(f"Key insight: {scenarios_list[max_idx]} delivers HIGHEST energy per hour despite {'lowest' if max_idx == 2 else 'different'} total volume!")
    print("="*70 + "\n")


# %% Section 4: Generate Plots

print("\n" + "="*70)
print("GENERATING PLOTS")
print("="*70 + "\n")


# Plot 3: Combined publication figure (3 key metrics)
print("\n3. Creating combined publication figure...")
key_metrics = ['deferrable_energy_kWh', 'percentage_flexible_energy', 'deferrable_task_count']
create_combined_figure(
    scenario_sample_week,
    key_metrics,
    f'{OUTPUT_DIR}/combined_comparison_publication_ready',
    save_pdf=True
)

# Plot 4: Time series overlay (3 key metrics)
print("\n4. Creating time series overlay...")
create_time_series_overlay(
    scenario_sample_week,
    key_metrics,
    f'{OUTPUT_DIR}/time_series_overlay_sample_week',
    save_pdf=True
)


# Plot 6: Notable points comparison (9 points across scenarios)
print("\n6. Creating notable points comparison (P1-P3, Q1-Q3, R1-R3)...")
create_notable_points_comparison(
    scenario_data,  # Use full data, not sample week
    f'{OUTPUT_DIR}/notable_points_comparison',
    save_pdf=True
)

# Plot 7: Hourly efficiency comparison
print("\n7. Creating hourly efficiency comparison...")
create_hourly_efficiency_plot(
    scenario_sample_week,
    f'{OUTPUT_DIR}/hourly_efficiency_comparison',
    save_pdf=True
)

# %% Section 5: Export Summary Table

print("\n8. Exporting summary statistics table...")
df_summary_table = create_summary_table(
    scenario_sample_week,
    METRICS,
    'EXPORTS/scenario_comparison_summary_sample_week.csv'
)