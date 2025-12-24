import matplotlib.pyplot as plt
import pandas as pd


def plot_ladflex_before_after(point_name, points_dates, res_dfs, latencies, 
                               fixed_flex_window, fixed_delta_notification, 
                               fixed_beta, fixed_gamma_buffer,
                               save_path=None, show=True, title_string=None):
    """
    Generate a LAD-Flex before/after plot showing DC total power vs power after LAD-Flex.
    
    Parameters:
    -----------
    point_name : str
        Name of the point to plot (e.g., 'p1', 'p2', 'q1', etc.)
    points_dates : dict
        Dictionary mapping point names to timestamps
    res_dfs : dict
        Dictionary containing simulation results
    latencies : list
        List of latency values
    fixed_flex_window : pd.Timedelta
        Fixed flexibility window duration
    fixed_delta_notification : pd.Timedelta
        Fixed notification time
    fixed_beta : float
        Fixed beta parameter
    fixed_gamma_buffer : pd.Timedelta
        Fixed gamma buffer duration
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    show : bool, default=True
        Whether to display the plot
    title_string : str, optional
        Scenario description string to prepend to the title
        
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    # Create tuple key for accessing res_dfs
    tuple_point = (points_dates[point_name], latencies[0], fixed_flex_window,
                   fixed_delta_notification, fixed_beta, fixed_gamma_buffer)
    
    # Extract time points
    t_0 = res_dfs[tuple_point]['t0']
    t_1 = res_dfs[tuple_point]['t1']
    t_2 = res_dfs[tuple_point]['t2']
    t_end = res_dfs[tuple_point]['tend']
    
    # Extract power data
    power_def = res_dfs[tuple_point]['df_t0_to_tend_def'].power_kW
    power_total = res_dfs[tuple_point]['df_t0_to_tend_total'].power_kW
    
    # Align indices
    common_index = power_def.index.union(power_total.index)
    power_def_aligned = power_def.reindex(common_index, method='nearest')
    power_total_aligned = power_total.reindex(common_index, method='nearest')
    
    # Calculate power difference
    power_diff = power_total_aligned - power_def_aligned
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot power curves
    ax.step(power_total.index, power_total.values, where='post', 
            label='DC Total Power', color='blue', linewidth=2)
    ax.step(power_diff.index, power_diff.values, where='post', linestyle='--', 
            color='purple', label='DC Power after LAD-Flex', linewidth=2)
    
    # Fill regions
    # Before t1 (notification period)
    ax.fill_between(power_total.index, power_total.values, power_diff.values,
                    where=(power_total.index <= t_1), 
                    color='lightblue', alpha=0.5)
    
    # Between t1 and t2 (flexible energy)
    ax.fill_between(power_total.index, power_total.values, power_diff.values,
                    where=((power_total.index > t_1) & (power_total.index <= t_2)),
                    color='lightgreen', alpha=0.5, label='Flexible energy')
    
    # After t2 (rebound energy)
    ax.fill_between(power_total.index, power_total.values, power_diff.values,
                    where=(power_total.index > t_2), 
                    color='lightblue', alpha=0.5, label='Rebound energy')
    
    # Add vertical lines and labels
    for t, label in zip([t_0, t_1, t_2, t_end], ['$t_0$', '$t_1$', '$t_2$', '$t_{end}$']):
        ax.axvline(t, linestyle='--', color='red', alpha=0.3, linewidth=2)
        ax.text(t, -0.07, label, color="red",
                ha='center', va='top', fontsize=16,
                transform=ax.get_xaxis_transform(), clip_on=False)
    
    # Styling
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True, alpha=0.3)
    
    # Set title with optional scenario string
    if title_string:
        ax.set_title(f'{title_string}: LAD-Flex Before and After', 
                    fontsize=18)
    else:
        ax.set_title(f'{point_name.upper()}: LAD-Flex Before and After', 
                    fontsize=18)
    
    ax.set_xlabel('Time', fontsize=16)
    ax.set_ylabel('Power (kW)', fontsize=16)
    ax.legend(fontsize=14, loc="best")
    
    # Show dark spines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.2)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    # Show plot
    if show:
        plt.show()
    
    return fig, ax



