"""
Estimate vertical autocorrelation (rho) coefficients from soil profile data.

This script calculates lag-1 autocorrelation for each soil property across
depth increments to determine appropriate rho values for vertical correlation.
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr


def estimate_vertical_rho(profile_data: pd.DataFrame, 
                          properties: list,
                          depth_col: str = 'hzdept_r',
                          profile_id_col: str = 'mukey') -> dict:
    """
    Calculate lag-1 vertical autocorrelation (rho) for each property.
    
    Parameters:
    -----------
    profile_data : pd.DataFrame
        Soil profile data with multiple horizons per profile
    properties : list
        List of property column names to analyze
    depth_col : str
        Column name for horizon depth
    profile_id_col : str
        Column name identifying unique soil profiles
        
    Returns:
    --------
    dict : Property names mapped to rho values
    """
    rho_values = {}
    
    for prop in properties:
        if prop not in profile_data.columns:
            print(f"Warning: {prop} not found in data")
            continue
            
        correlations = []
        
        # Group by profile
        for profile_id, profile_df in profile_data.groupby(profile_id_col):
            # Sort by depth
            profile_df = profile_df.sort_values(depth_col).reset_index(drop=True)
            
            # Need at least 2 horizons
            if len(profile_df) < 2:
                continue
                
            # Get property values, excluding NaN
            values = profile_df[prop].values
            if np.isnan(values).any():
                continue
                
            # Calculate lag-1 correlation (adjacent horizons)
            if len(values) >= 2:
                # Spearman correlation between z and z+1
                rho, _ = spearmanr(values[:-1], values[1:])
                if not np.isnan(rho):
                    correlations.append(rho)
        
        if correlations:
            # Median across all profiles
            rho_values[prop] = np.median(correlations)
            print(f"{prop:25s}: rho = {rho_values[prop]:.3f}  (n={len(correlations)} profiles)")
        else:
            print(f"{prop:25s}: insufficient data")
    
    return rho_values


def estimate_rho_by_depth_lag(profile_data: pd.DataFrame,
                              property_name: str,
                              max_lag: int = 5,
                              depth_col: str = 'hzdept_r',
                              profile_id_col: str = 'mukey') -> pd.DataFrame:
    """
    Calculate correlation at different depth lags to see decay pattern.
    
    This helps determine if exponential decay model is appropriate.
    
    Returns:
    --------
    pd.DataFrame with columns: lag, rho, n_pairs
    """
    results = []
    
    for lag in range(1, max_lag + 1):
        correlations = []
        
        for profile_id, profile_df in profile_data.groupby(profile_id_col):
            profile_df = profile_df.sort_values(depth_col).reset_index(drop=True)
            
            if len(profile_df) < lag + 1:
                continue
                
            values = profile_df[property_name].values
            if np.isnan(values).any():
                continue
                
            # Correlation at this lag
            if len(values) >= lag + 1:
                rho, _ = spearmanr(values[:-lag], values[lag:])
                if not np.isnan(rho):
                    correlations.append(rho)
        
        if correlations:
            results.append({
                'lag': lag,
                'rho': np.median(correlations),
                'n_pairs': len(correlations)
            })
    
    return pd.DataFrame(results)


# Example usage:
if __name__ == "__main__":
    # Load your soil profile data
    # This should have multiple rows per soil profile, one per horizon
    
    # Example with SSURGO/KSSL data structure:
    # df = pd.read_csv('soil_horizons.csv')
    
    # For demonstration, create synthetic data
    print("Example: Estimating rho values from soil profile data\n")
    print("=" * 70)
    
    # Simulated example
    np.random.seed(42)
    n_profiles = 100
    avg_horizons = 4
    
    data = []
    for i in range(n_profiles):
        n_hz = np.random.randint(2, 7)  # 2-6 horizons per profile
        
        # Simulate correlated properties with depth
        # True rho = 0.8 for clay, 0.6 for rfv
        clay_0 = np.random.uniform(10, 40)
        rfv_0 = np.random.uniform(0, 30)
        
        for hz in range(n_hz):
            # AR(1) process
            if hz == 0:
                clay = clay_0
                rfv = rfv_0
            else:
                clay = 0.8 * clay + np.random.normal(0, 5)
                rfv = 0.6 * rfv + np.random.normal(0, 10)
            
            data.append({
                'mukey': f'profile_{i}',
                'hzdept_r': hz * 20,  # 20cm increments
                'clay': np.clip(clay, 0, 100),
                'rfv': np.clip(rfv, 0, 100),
            })
    
    df = pd.DataFrame(data)
    
    # Estimate rho values
    print("\n1. Lag-1 Autocorrelation (recommended rho values):")
    print("-" * 70)
    rho_values = estimate_vertical_rho(df, ['clay', 'rfv'])
    
    # Analyze decay with depth
    print("\n2. Correlation vs Depth Lag (clay):")
    print("-" * 70)
    decay_df = estimate_rho_by_depth_lag(df, 'clay', max_lag=5)
    print(decay_df.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("\nRecommendation: Use lag-1 rho values in AR(1) model")
    print("For exponential decay model: rho(h) = rho^(h/Δh)")
