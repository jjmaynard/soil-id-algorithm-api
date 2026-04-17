"""
Validate that within-horizon correlations are preserved after vertical AR(1).

This function should be run on actual simulation output to verify the
theoretical analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


def check_within_horizon_correlations(
    sim_data_df: pd.DataFrame,
    target_correlation_matrix: np.ndarray,
    property_names: list = None,
    depth_col: str = 'hzdept_r',
    tolerance: float = 0.10
) -> pd.DataFrame:
    """
    Verify that within-horizon correlations match the target after vertical AR(1).
    
    Parameters:
    -----------
    sim_data_df : pd.DataFrame
        Simulated data AFTER applying vertical correlation
    target_correlation_matrix : np.ndarray
        The GLOBAL_CORRELATION_MATRIX that was used for Cholesky decomposition
    property_names : list
        Names of properties in order matching correlation matrix
        Default: ['ilr1', 'ilr2', 'bulk_density', 'water_retention_33', 
                  'water_retention_1500', 'rfv']
    depth_col : str
        Column name for depth
    tolerance : float
        Acceptable deviation from target correlation (default 0.10 = 10%)
        
    Returns:
    --------
    pd.DataFrame with validation results
    """
    if property_names is None:
        property_names = [
            'ilr1', 'ilr2', 'bulk_density_third_bar',
            'water_retention_third_bar', 'water_retention_15_bar', 'rfv'
        ]
    
    # Map to actual column names in dataframe
    col_mapping = {
        'ilr1': 'ilr1',
        'ilr2': 'ilr2',
        'bulk_density': 'bulk_density_third_bar',
        'water_retention_33': 'water_retention_third_bar',
        'water_retention_1500': 'water_retention_15_bar',
        'rfv': 'rfv'
    }
    
    # Get available columns
    available_cols = [col for col in property_names if col in sim_data_df.columns]
    
    if len(available_cols) < 2:
        print("Error: Need at least 2 properties to check correlations")
        return None
    
    depths = sorted(sim_data_df[depth_col].unique())
    
    print("=" * 80)
    print("WITHIN-HORIZON CORRELATION VALIDATION")
    print("=" * 80)
    print(f"Number of depths: {len(depths)}")
    print(f"Properties analyzed: {len(available_cols)}")
    print(f"Tolerance: ±{tolerance:.2%}")
    print()
    
    results = []
    
    # For each depth, compute actual correlation matrix
    for depth in depths:
        depth_data = sim_data_df[sim_data_df[depth_col] == depth]
        
        if len(depth_data) < 10:
            print(f"Warning: Depth {depth} has only {len(depth_data)} samples, skipping")
            continue
        
        # Compute correlation matrix for this depth
        depth_corr = depth_data[available_cols].corr().values
        
        # Compare with target (only for dimensions that match)
        n_props = len(available_cols)
        target_subset = target_correlation_matrix[:n_props, :n_props]
        
        # Calculate element-wise differences
        diff = depth_corr - target_subset
        
        # Analyze each pair
        for i in range(n_props):
            for j in range(i+1, n_props):  # Upper triangle only
                target_corr = target_subset[i, j]
                actual_corr = depth_corr[i, j]
                difference = actual_corr - target_corr
                pct_error = (difference / target_corr * 100) if abs(target_corr) > 0.01 else 0
                
                status = "✓" if abs(difference) <= tolerance else "✗"
                
                results.append({
                    'depth': depth,
                    'property_1': available_cols[i],
                    'property_2': available_cols[j],
                    'target_corr': target_corr,
                    'actual_corr': actual_corr,
                    'difference': difference,
                    'pct_error': pct_error,
                    'within_tolerance': abs(difference) <= tolerance,
                    'status': status
                })
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        print("No results to analyze")
        return None
    
    # Summary statistics
    print("SUMMARY STATISTICS:")
    print("-" * 80)
    print(f"Total pairs analyzed: {len(results_df)}")
    print(f"Within tolerance: {results_df['within_tolerance'].sum()} ({results_df['within_tolerance'].mean()*100:.1f}%)")
    print(f"Outside tolerance: {(~results_df['within_tolerance']).sum()} ({(~results_df['within_tolerance']).mean()*100:.1f}%)")
    print()
    print(f"Mean absolute difference: {results_df['difference'].abs().mean():.4f}")
    print(f"Max absolute difference: {results_df['difference'].abs().max():.4f}")
    print(f"RMS error: {np.sqrt((results_df['difference']**2).mean()):.4f}")
    print()
    
    # Worst offenders
    print("TOP 5 LARGEST DISTORTIONS:")
    print("-" * 80)
    worst = results_df.nlargest(5, 'difference', keep='all')[
        ['property_1', 'property_2', 'depth', 'target_corr', 'actual_corr', 'difference', 'status']
    ]
    print(worst.to_string(index=False))
    print()
    
    # By depth analysis
    print("DISTORTION BY DEPTH:")
    print("-" * 80)
    by_depth = results_df.groupby('depth').agg({
        'difference': lambda x: x.abs().mean(),
        'within_tolerance': 'mean'
    }).round(4)
    by_depth.columns = ['Avg_Abs_Diff', 'Pct_Within_Tol']
    print(by_depth.to_string())
    print()
    
    # Overall verdict
    print("=" * 80)
    print("OVERALL ASSESSMENT:")
    print("=" * 80)
    
    mean_abs_diff = results_df['difference'].abs().mean()
    max_abs_diff = results_df['difference'].abs().max()
    pct_within = results_df['within_tolerance'].mean() * 100
    
    if mean_abs_diff < 0.03 and pct_within > 95:
        print("✓ EXCELLENT: Correlations very well preserved")
        print("  Mean distortion < 0.03 and >95% within tolerance")
    elif mean_abs_diff < 0.05 and pct_within > 85:
        print("✓ GOOD: Correlations adequately preserved")
        print("  Mean distortion < 0.05 and >85% within tolerance")
    elif mean_abs_diff < 0.10 and pct_within > 70:
        print("⚠ ACCEPTABLE: Some distortion but likely usable")
        print("  Mean distortion < 0.10 and >70% within tolerance")
    else:
        print("✗ CONCERNING: Significant distortion detected")
        print("  Consider adjusting rho values or using alternative method")
    
    print()
    print(f"Recommendation: {'Continue with current approach' if mean_abs_diff < 0.05 else 'Review rho values'}")
    print()
    
    return results_df


def compare_rho_value_scenarios(
    sim_data_df: pd.DataFrame,
    target_correlation_matrix: np.ndarray,
    rho_scenarios: Dict[str, Dict[str, float]],
    property_names: list = None
) -> pd.DataFrame:
    """
    Compare distortion levels under different rho value configurations.
    
    This helps determine if adjusting rho values would reduce distortion.
    
    Parameters:
    -----------
    sim_data_df : pd.DataFrame
        Base simulated data (before vertical correlation)
    target_correlation_matrix : np.ndarray
        Target correlation matrix
    rho_scenarios : dict
        Dictionary mapping scenario names to rho value dictionaries
        Example: {'Current': {...}, 'Constrained': {...}, 'Uniform': {...}}
    property_names : list
        Property names
        
    Returns:
    --------
    pd.DataFrame comparing scenarios
    """
    # Implementation would re-apply vertical correlation with different rho values
    # and compare distortion levels
    pass


# Example usage
if __name__ == "__main__":
    print("This module provides validation functions for correlation preservation.")
    print()
    print("Example usage:")
    print("""
from soil_id.soil_sim import soil_sim, GLOBAL_CORRELATION_MATRIX
from scripts.check_correlation_preservation import check_within_horizon_correlations

# Run simulation
aws_PIW90, var_imp = soil_sim(muhorzdata_pd)

# To validate, you would need to modify soil_sim to return sim_data_df
# Then run:
results = check_within_horizon_correlations(
    sim_data_df, 
    GLOBAL_CORRELATION_MATRIX,
    tolerance=0.10  # 10% tolerance
)

# Review results
print(results[results['within_tolerance'] == False])
""")
