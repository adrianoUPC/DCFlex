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

def create_boxplot_comparison(scenario_data, metrics, save_path, save_pdf=True):
    """
    Create side-by-side boxplots comparing scenarios across all metrics.

    Parameters:
    -----------
    scenario_data : dict
        Dictionary with {scenario_name: df_summary_filtered}
    metrics : list
        List of metric column names to plot
    save_path : str
        Path to save output PNG file (without extension)
    save_pdf : bool
        Whether to also save PDF version
    """
    sns.set(style="whitegrid")

    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 6))

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # Prepare data for boxplot
        data_to_plot = []
        labels = []
        colors = []

        for scenario in ['BASELINE', 'DAM', 'mFRR']:
            data_to_plot.append(scenario_data[scenario][metric].values)
            labels.append(SCENARIO_LABELS[scenario])
            colors.append(SCENARIO_COLORS[scenario])

        # Create boxplot
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                        widths=0.6, showfliers=True, flierprops={'markersize': 4})

        # Color boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Color whiskers and caps
        for whisker in bp['whiskers']:
            whisker.set_linewidth(1.5)
        for cap in bp['caps']:
            cap.set_linewidth(1.5)
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(2)

        # Styling
        ax.set_title(METRIC_INFO[metric]['short'], fontsize=14, fontweight='bold')
        ax.set_ylabel('Value', fontsize=13)
        ax.tick_params(axis='both', labelsize=11)
        ax.tick_params(axis='x', rotation=0)
        ax.grid(axis='y', alpha=0.3)

        # Thicker spines
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.2)

    fig.suptitle('Scenario Comparison: Sample Week Flexibility Metrics (Feb 1-7, 1970)',
                 fontsize=18, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
    if save_pdf:
        plt.savefig(f'{save_path}.pdf', bbox_inches='tight')
    plt.show()

    sns.reset_defaults()
    print(f"Boxplot comparison saved to {save_path}.png/.pdf")


def create_bar_chart_comparison(scenario_data, metrics, save_path, save_pdf=True):
    """
    Create bar charts with error bars (mean ± std) comparing scenarios.

    Parameters:
    -----------
    scenario_data : dict
        Dictionary with {scenario_name: df_summary_filtered}
    metrics : list
        List of metric column names to plot
    save_path : str
        Path to save output PNG file (without extension)
    save_pdf : bool
        Whether to also save PDF version
    """
    sns.set(style="whitegrid")

    # Create 2×3 grid (5 metrics + 1 normalized comparison)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # Compute statistics
    stats = {}
    for scenario in ['BASELINE', 'DAM', 'mFRR']:
        stats[scenario] = {}
        for metric in metrics:
            stats[scenario][metric] = {
                'mean': scenario_data[scenario][metric].mean(),
                'std': scenario_data[scenario][metric].std()
            }

    # Plot first 5 metrics
    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        means = [stats[s][metric]['mean'] for s in ['BASELINE', 'DAM', 'mFRR']]
        stds = [stats[s][metric]['std'] for s in ['BASELINE', 'DAM', 'mFRR']]
        colors = [SCENARIO_COLORS[s] for s in ['BASELINE', 'DAM', 'mFRR']]
        labels = [SCENARIO_LABELS[s] for s in ['BASELINE', 'DAM', 'mFRR']]

        x_pos = np.arange(len(labels))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color=colors, alpha=0.7,
                      edgecolor='black', linewidth=1.5)

        ax.set_xlabel('Scenario', fontsize=13, fontweight='bold')
        ax.set_ylabel('Mean ± Std', fontsize=13, fontweight='bold')
        ax.set_title(METRIC_INFO[metric]['short'], fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=11)
        ax.tick_params(axis='y', labelsize=11)
        ax.grid(axis='y', alpha=0.3)

        # Thicker spines
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.2)

    # Panel 6: Normalized comparison
    ax = axes[5]

    # Normalize each metric to [0, 1] based on max across scenarios
    normalized_data = {}
    for scenario in ['BASELINE', 'DAM', 'mFRR']:
        normalized_data[scenario] = []

    for metric in metrics:
        max_val = max([stats[s][metric]['mean'] for s in ['BASELINE', 'DAM', 'mFRR']])
        if max_val > 0:
            for scenario in ['BASELINE', 'DAM', 'mFRR']:
                normalized_data[scenario].append(stats[scenario][metric]['mean'] / max_val)
        else:
            for scenario in ['BASELINE', 'DAM', 'mFRR']:
                normalized_data[scenario].append(0)

    # Grouped bar chart
    x = np.arange(len(metrics))
    width = 0.25

    for i, scenario in enumerate(['BASELINE', 'DAM', 'mFRR']):
        offset = (i - 1) * width
        ax.bar(x + offset, normalized_data[scenario], width,
               label=SCENARIO_LABELS[scenario], color=SCENARIO_COLORS[scenario],
               alpha=0.7, edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Metric', fontsize=13, fontweight='bold')
    ax.set_ylabel('Normalized Value [0-1]', fontsize=13, fontweight='bold')
    ax.set_title('Normalized Comparison (All Metrics)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_INFO[m]['short'] for m in metrics], fontsize=9, rotation=15, ha='right')
    ax.tick_params(axis='y', labelsize=11)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.1)

    # Thicker spines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.2)

    fig.suptitle('Mean ± Std Comparison: Sample Week (Feb 1-7, 1970)',
                 fontsize=18, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
    if save_pdf:
        plt.savefig(f'{save_path}.pdf', bbox_inches='tight')
    plt.show()

    sns.reset_defaults()
    print(f"Bar chart comparison saved to {save_path}.png/.pdf")


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

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Compute statistics
    stats = {}
    for scenario in ['BASELINE', 'DAM', 'mFRR']:
        stats[scenario] = {}
        for metric in metrics:
            stats[scenario][metric] = {
                'mean': scenario_data[scenario][metric].mean(),
                'std': scenario_data[scenario][metric].std()
            }

    # Row 1: Boxplots
    for idx, metric in enumerate(metrics):
        ax = axes[0, idx]

        # Prepare data for boxplot
        data_to_plot = []
        labels = []
        colors = []

        for scenario in ['BASELINE', 'DAM', 'mFRR']:
            data_to_plot.append(scenario_data[scenario][metric].values)
            labels.append(scenario.replace('_', '\n'))
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
        ax.set_title(METRIC_INFO[metric]['short'], fontsize=14, fontweight='bold')
        ax.set_ylabel('Value', fontsize=12)
        ax.tick_params(axis='both', labelsize=11)
        ax.grid(axis='y', alpha=0.3)

        # Thicker spines
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.2)

    # Row 2: Bar charts
    for idx, metric in enumerate(metrics):
        ax = axes[1, idx]

        means = [stats[s][metric]['mean'] for s in ['BASELINE', 'DAM', 'mFRR']]
        stds = [stats[s][metric]['std'] for s in ['BASELINE', 'DAM', 'mFRR']]
        colors = [SCENARIO_COLORS[s] for s in ['BASELINE', 'DAM', 'mFRR']]
        labels = [s for s in ['BASELINE', 'DAM', 'mFRR']]

        x_pos = np.arange(len(labels))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color=colors, alpha=0.7,
                      edgecolor='black', linewidth=1.5)

        ax.set_xlabel('Scenario', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean ± Std', fontsize=12)
        ax.set_title(METRIC_INFO[metric]['short'], fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=11)
        ax.tick_params(axis='y', labelsize=11)
        ax.grid(axis='y', alpha=0.3)

        # Thicker spines
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.2)

    # Add row labels
    axes[0, 0].text(-0.3, 0.5, 'Distribution (Boxplots)', transform=axes[0, 0].transAxes,
                    fontsize=16, fontweight='bold', va='center', ha='right', rotation=90)
    axes[1, 0].text(-0.3, 0.5, 'Summary Statistics', transform=axes[1, 0].transAxes,
                    fontsize=16, fontweight='bold', va='center', ha='right', rotation=90)

    fig.suptitle('Scenario Comparison: Key Flexibility Metrics (Sample Week)',
                 fontsize=18, fontweight='bold', y=0.98)

    plt.tight_layout()
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
        ax.set_title(METRIC_INFO[metric]['short'], fontsize=14, fontweight='bold')
        ax.set_xlabel('Time', fontsize=13)
        ax.set_ylabel('Value', fontsize=13)
        ax.tick_params(axis='both', labelsize=11)
        ax.legend(fontsize=11, loc='best')
        ax.grid(alpha=0.3)

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)

        # Thicker spines
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.2)

    fig.suptitle('Time Series Comparison: Sample Week (Feb 1-7, 1970)',
                 fontsize=18, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
    if save_pdf:
        plt.savefig(f'{save_path}.pdf', bbox_inches='tight')
    plt.show()

    sns.reset_defaults()
    print(f"Time series overlay saved to {save_path}.png/.pdf")


def create_percentage_breakdown(scenario_data, save_path, save_pdf=True):
    """
    Create stacked horizontal bar chart showing energy composition breakdown.

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

    fig, ax = plt.subplots(figsize=(12, 5))

    # Compute means for each scenario
    breakdown = {}
    for scenario in ['BASELINE', 'DAM', 'mFRR']:
        deferrable = scenario_data[scenario]['deferrable_energy_kWh'].mean()
        total = scenario_data[scenario]['total_energy_kWh'].mean()
        rebound = scenario_data[scenario]['rebound_energy_kWh'].mean()
        non_deferrable = total - deferrable

        breakdown[scenario] = {
            'deferrable': deferrable,
            'non_deferrable': max(0, non_deferrable),  # Ensure non-negative
            'rebound': rebound,
            'total': total
        }

    # Create stacked horizontal bars
    scenarios_list = ['BASELINE', 'DAM', 'mFRR']
    y_pos = np.arange(len(scenarios_list))

    # Deferrable portion (green)
    deferrable_vals = [breakdown[s]['deferrable'] for s in scenarios_list]
    non_deferrable_vals = [breakdown[s]['non_deferrable'] for s in scenarios_list]
    rebound_vals = [breakdown[s]['rebound'] for s in scenarios_list]

    # Plot stacked bars
    p1 = ax.barh(y_pos, deferrable_vals, color='#66C2A5', alpha=0.8,
                 edgecolor='black', linewidth=1.5, label='Deferrable Energy')
    p2 = ax.barh(y_pos, non_deferrable_vals, left=deferrable_vals,
                 color='#CCCCCC', alpha=0.8, edgecolor='black', linewidth=1.5,
                 label='Non-Deferrable Energy')

    # Add rebound as separate component (not stacked, offset below)
    y_offset = 0.3
    p3 = ax.barh(y_pos - y_offset, rebound_vals, height=0.2, color='#FC8D62',
                 alpha=0.8, edgecolor='black', linewidth=1.5, label='Rebound Energy')

    # Add percentage annotations
    for i, scenario in enumerate(scenarios_list):
        total = breakdown[scenario]['total']
        deferrable_pct = (breakdown[scenario]['deferrable'] / total) * 100 if total > 0 else 0
        non_deferrable_pct = (breakdown[scenario]['non_deferrable'] / total) * 100 if total > 0 else 0

        # Deferrable annotation
        ax.text(breakdown[scenario]['deferrable'] / 2, y_pos[i],
               f'{deferrable_pct:.1f}%', ha='center', va='center',
               fontsize=12, fontweight='bold', color='black')

        # Non-deferrable annotation
        ax.text(breakdown[scenario]['deferrable'] + breakdown[scenario]['non_deferrable'] / 2,
               y_pos[i], f'{non_deferrable_pct:.1f}%', ha='center', va='center',
               fontsize=12, fontweight='bold', color='black')

    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels([SCENARIO_LABELS[s] for s in scenarios_list], fontsize=13)
    ax.set_xlabel('Energy (kWh)', fontsize=14, fontweight='bold')
    ax.set_title('Energy Composition Breakdown by Scenario (Mean, Sample Week)',
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(axis='x', alpha=0.3)
    ax.tick_params(axis='x', labelsize=12)

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
    print(f"Percentage breakdown saved to {save_path}.png/.pdf")


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
            if val > 1:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    # Formatting
    ax.set_xlabel('Notable Points', fontsize=14, fontweight='bold')
    ax.set_ylabel('Deferrable Energy (kWh)', fontsize=14, fontweight='bold')
    ax.set_title('Comparison of 9 Notable Points Across Scenarios\n(P1-P3: Top flexibility | Q1-Q3: 75th percentile | R1-R3: 50th percentile)',
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(point_names, fontsize=13)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(fontsize=13, loc='upper right', framealpha=0.95)
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
               fontsize=14, fontweight='bold')

        # Formula below bar
        ax.text(bar.get_x() + bar.get_width()/2, -0.15,
               f'{avg_def:.1f} kWh / {window} h',
               ha='center', va='top', fontsize=11, style='italic')

    # Highlight highest efficiency
    max_idx = np.argmax(efficiencies)
    bars[max_idx].set_linewidth(3)
    bars[max_idx].set_edgecolor('darkgreen')

    # Add annotation for highest
    ax.text(max_idx, efficiencies[max_idx] + 0.2,
           '← HIGHEST efficiency!', ha='left', va='center',
           fontsize=12, fontweight='bold', color='darkgreen',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))

    # Formatting
    ax.set_ylabel('Energy per Hour of Flexibility Window (kWh/h)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Scenario', fontsize=14, fontweight='bold')
    ax.set_title('Hourly Efficiency Comparison\nE_norm = (Avg Deferrable Energy) / (Window Duration)',
                 fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', labelsize=13)
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

# Plot 1: Boxplot comparison (all 5 metrics)
print("1. Creating boxplot comparison (5 metrics)...")
create_boxplot_comparison(
    scenario_sample_week,
    METRICS,
    f'{OUTPUT_DIR}/boxplot_comparison_5metrics_sample_week',
    save_pdf=True
)

# Plot 2: Bar chart comparison (mean ± std)
print("\n2. Creating bar chart comparison (mean ± std)...")
create_bar_chart_comparison(
    scenario_sample_week,
    METRICS,
    f'{OUTPUT_DIR}/bar_chart_comparison_sample_week',
    save_pdf=True
)

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

# Plot 5: Percentage breakdown stacked bar
print("\n5. Creating percentage breakdown stacked bar chart...")
create_percentage_breakdown(
    scenario_sample_week,
    f'{OUTPUT_DIR}/percentage_breakdown_stacked',
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

print("\n" + "="*70)
print("ALL PLOTS AND TABLES GENERATED SUCCESSFULLY!")
print("="*70)
print(f"\nOutputs saved to:")
print(f"  - Plots: {OUTPUT_DIR}/")
print(f"  - Table: EXPORTS/scenario_comparison_summary_sample_week.csv")
print("\nGenerated files:")
print("  1. boxplot_comparison_5metrics_sample_week.png/pdf")
print("  2. bar_chart_comparison_sample_week.png/pdf")
print("  3. combined_comparison_publication_ready.png/pdf")
print("  4. time_series_overlay_sample_week.png/pdf")
print("  5. percentage_breakdown_stacked.png/pdf")
print("  6. notable_points_comparison.png/pdf")
print("  7. hourly_efficiency_comparison.png/pdf")
print("  8. scenario_comparison_summary_sample_week.csv")
print("="*70)
