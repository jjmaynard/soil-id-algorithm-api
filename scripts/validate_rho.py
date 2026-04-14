"""
Validate and visualize vertical correlation in simulated soil profiles.

This script helps assess whether the rho values produce realistic results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict


def plot_correlation_decay(sim_data: pd.DataFrame,
                           property_col: str,
                           rho_expected: float,
                           max_lag: int = 5,
                           depth_col: str = 'hzdept_r',
                           profile_id: str = 'compname_grp'):
    """
    Plot how correlation decays with depth lag.
    
    For AR(1), correlation should follow: rho^lag
    e.g., if rho=0.8, then lag-2 = 0.64, lag-3 = 0.51, etc.
    """
    depths = sorted(sim_data[depth_col].unique())
    
    correlations = []
    for lag in range(1, min(max_lag + 1, len(depths))):
        corr_values = []
        
        for profile in sim_data[profile_id].unique():
            profile_data = sim_data[sim_data[profile_id] == profile].sort_values(depth_col)
            
            if len(profile_data) < lag + 1:
                continue
                
            for i in range(len(depths) - lag):
                mask1 = (sim_data[profile_id] == profile) & (sim_data[depth_col] == depths[i])
                mask2 = (sim_data[profile_id] == profile) & (sim_data[depth_col] == depths[i + lag])
                
                vals1 = sim_data.loc[mask1, property_col].values
                vals2 = sim_data.loc[mask2, property_col].values
                
                if len(vals1) > 1 and len(vals2) > 1 and len(vals1) == len(vals2):
                    corr = np.corrcoef(vals1, vals2)[0, 1]
                    if not np.isnan(corr):
                        corr_values.append(corr)
        
        if corr_values:
            correlations.append({
                'lag': lag,
                'correlation': np.median(corr_values),
                'n': len(corr_values)
            })
    
    if not correlations:
        print("Insufficient data for correlation analysis")
        return
    
    # Plot
    df_corr = pd.DataFrame(correlations)
    lags = df_corr['lag'].values
    observed = df_corr['correlation'].values
    expected = rho_expected ** lags
    
    plt.figure(figsize=(10, 6))
    plt.plot(lags, observed, 'o-', label='Observed', markersize=8, linewidth=2)
    plt.plot(lags, expected, 's--', label=f'Expected (ρ={rho_expected})', markersize=6, linewidth=2)
    plt.xlabel('Depth Lag (horizons)', fontsize=12)
    plt.ylabel('Correlation', fontsize=12)
    plt.title(f'Vertical Correlation Decay: {property_col}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.ylim(0, 1)
    
    # Add text with statistics
    rmse = np.sqrt(np.mean((observed - expected)**2))
    plt.text(0.02, 0.98, f'RMSE: {rmse:.3f}\nTarget ρ: {rho_expected:.2f}',
             transform=plt.gca().transAxes, 
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=10)
    
    plt.tight_layout()
    return plt.gcf(), df_corr


def plot_example_profiles(sim_data: pd.DataFrame,
                          property_col: str,
                          n_profiles: int = 5,
                          depth_col: str = 'hzdept_r',
                          profile_id: str = 'compname_grp'):
    """
    Plot example simulated soil profiles to visually assess realism.
    """
    profiles = sim_data[profile_id].unique()[:n_profiles]
    
    fig, axes = plt.subplots(1, n_profiles, figsize=(15, 6), sharey=True)
    if n_profiles == 1:
        axes = [axes]
    
    for idx, (ax, profile) in enumerate(zip(axes, profiles)):
        profile_data = sim_data[sim_data[profile_id] == profile].sort_values(depth_col)
        
        # Get depth midpoints
        depths = (profile_data[depth_col].values + profile_data['hzdepb_r'].values) / 2
        values = profile_data[property_col].values
        
        ax.plot(values, depths, 'o-', linewidth=2, markersize=6)
        ax.set_xlabel(property_col, fontsize=10)
        if idx == 0:
            ax.set_ylabel('Depth (cm)', fontsize=11)
        ax.set_title(f'Profile {idx+1}', fontsize=11)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Example Simulated Soil Profiles: {property_col}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def validate_rho_values(sim_data: pd.DataFrame,
                       rho_dict: Dict[str, float],
                       property_mapping: Dict[str, str] = None):
    """
    Comprehensive validation report for rho values.
    """
    if property_mapping is None:
        property_mapping = {
            'ilr1': 'ilr1',
            'ilr2': 'ilr2',
            'bulk_density': 'bulk_density_third_bar',
            'water_retention_33': 'water_retention_third_bar',
            'water_retention_1500': 'water_retention_15_bar',
            'rfv': 'rfv'
        }
    
    print("=" * 80)
    print("VERTICAL CORRELATION VALIDATION REPORT")
    print("=" * 80)
    print()
    
    depths = sorted(sim_data['hzdept_r'].unique())
    print(f"Number of depth levels: {len(depths)}")
    print(f"Depths: {depths}")
    print(f"Number of simulations per depth: {len(sim_data[sim_data['hzdept_r'] == depths[0]])}")
    print()
    
    results = []
    
    for prop_key, col_name in property_mapping.items():
        if col_name not in sim_data.columns:
            continue
            
        rho_target = rho_dict.get(prop_key, None)
        if rho_target is None:
            continue
        
        # Calculate lag-1 correlation
        corr_values = []
        for i in range(len(depths) - 1):
            mask1 = sim_data['hzdept_r'] == depths[i]
            mask2 = sim_data['hzdept_r'] == depths[i+1]
            
            vals1 = sim_data.loc[mask1, col_name].values
            vals2 = sim_data.loc[mask2, col_name].values
            
            if len(vals1) > 1 and len(vals2) > 1 and len(vals1) == len(vals2):
                corr = np.corrcoef(vals1, vals2)[0, 1]
                if not np.isnan(corr):
                    corr_values.append(corr)
        
        if corr_values:
            rho_observed = np.mean(corr_values)
            error = abs(rho_observed - rho_target)
            
            results.append({
                'property': prop_key,
                'target_rho': rho_target,
                'observed_rho': rho_observed,
                'error': error,
                'status': '✓' if error < 0.05 else '⚠'
            })
    
    # Print results table
    print(f"{'Property':<25} {'Target ρ':<12} {'Observed ρ':<12} {'Error':<10} {'Status':<8}")
    print("-" * 80)
    for r in results:
        print(f"{r['property']:<25} {r['target_rho']:<12.3f} {r['observed_rho']:<12.3f} "
              f"{r['error']:<10.3f} {r['status']:<8}")
    
    print()
    print("Status: ✓ = Good (error < 0.05), ⚠ = Review (error >= 0.05)")
    print()
    
    # Overall assessment
    avg_error = np.mean([r['error'] for r in results])
    print(f"Average error: {avg_error:.4f}")
    
    if avg_error < 0.03:
        print("✓ Overall: EXCELLENT - Rho values are well-calibrated")
    elif avg_error < 0.05:
        print("✓ Overall: GOOD - Rho values are acceptable")
    else:
        print("⚠ Overall: NEEDS REVIEW - Consider adjusting rho values")
    
    print()
    print("=" * 80)
    
    return pd.DataFrame(results)


# Example usage
if __name__ == "__main__":
    print("This script provides validation functions for vertical correlation.")
    print("\nExample usage:")
    print("""
    import pandas as pd
    from validate_rho import validate_rho_values, plot_correlation_decay
    
    # Load simulated data
    sim_data = pd.read_csv('simulated_profiles.csv')
    
    # Define your rho values
    rho_dict = {
        'ilr1': 0.85,
        'ilr2': 0.85,
        'bulk_density': 0.75,
        'water_retention_33': 0.70,
        'water_retention_1500': 0.70,
        'rfv': 0.60,
    }
    
    # Validate
    results = validate_rho_values(sim_data, rho_dict)
    
    # Plot correlation decay for bulk density
    fig, df = plot_correlation_decay(sim_data, 'bulk_density_third_bar', rho_expected=0.75)
    plt.savefig('correlation_decay.png', dpi=300)
    
    # Plot example profiles
    fig = plot_example_profiles(sim_data, 'bulk_density_third_bar', n_profiles=5)
    plt.savefig('example_profiles.png', dpi=300)
    """)
