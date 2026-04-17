"""
Test whether vertical correlation preserves within-horizon property correlations.

This tests the key question: Does applying AR(1) vertical correlation AFTER
Cholesky decomposition distort the within-horizon correlation structure?
"""

import numpy as np
import pandas as pd


def test_correlation_preservation():
    """
    Demonstrate the correlation distortion problem with naive AR(1) implementation.
    """
    print("=" * 80)
    print("TESTING CORRELATION PRESERVATION")
    print("=" * 80)
    print()
    
    # Simulate the original process using Cholesky decomposition
    np.random.seed(42)
    n_sim = 1000
    n_depths = 3
    
    # Target correlation matrix (simplified 3x3 for clarity)
    # Property 1: Clay
    # Property 2: Bulk Density  
    # Property 3: Water Retention
    target_corr = np.array([
        [1.0,  -0.70,  0.80],  # Clay: negatively corr with BD, positively with WR
        [-0.70,  1.0, -0.60],  # Bulk Density: negatively corr with both
        [0.80, -0.60,  1.0],   # Water Retention: follows clay
    ])
    
    print("TARGET WITHIN-HORIZON CORRELATION MATRIX:")
    print(target_corr)
    print()
    
    # Step 1: Generate correlated properties at each depth using Cholesky
    # (This mimics the original simulate_correlated_triangular approach)
    
    data = []
    for depth in [0, 30, 60]:
        # Generate uncorrelated normal
        uncorrelated = np.random.normal(size=(n_sim, 3))
        
        # Apply Cholesky to induce correlation
        L = np.linalg.cholesky(target_corr)
        correlated = uncorrelated @ L.T
        
        # Transform to desired scale (e.g., clay: 20±10, BD: 1.4±0.2, WR: 0.3±0.1)
        clay = correlated[:, 0] * 10 + 25
        bulk_density = correlated[:, 1] * 0.2 + 1.4
        water_retention = correlated[:, 2] * 0.1 + 0.3
        
        for i in range(n_sim):
            data.append({
                'depth': depth,
                'clay': clay[i],
                'bulk_density': bulk_density[i],
                'water_retention': water_retention[i],
            })
    
    df_before = pd.DataFrame(data)
    
    # Calculate within-horizon correlation BEFORE vertical correlation
    print("WITHIN-HORIZON CORRELATIONS BEFORE VERTICAL AR(1):")
    print("(Should match target matrix above)")
    print()
    for depth in [0, 30, 60]:
        depth_data = df_before[df_before['depth'] == depth]
        corr_matrix = depth_data[['clay', 'bulk_density', 'water_retention']].corr().values
        print(f"Depth {depth}cm:")
        print(corr_matrix.round(3))
        print()
    
    # Step 2: Apply naive AR(1) vertical correlation (property-by-property)
    # This is what the current implementation does
    
    df_after = df_before.copy()
    rho_values = {'clay': 0.80, 'bulk_density': 0.75, 'water_retention': 0.80}
    
    depths = sorted(df_after['depth'].unique())
    
    for prop, rho in rho_values.items():
        for i, depth in enumerate(depths[1:], start=1):
            prev_depth = depths[i-1]
            
            # Get current and previous values
            curr_mask = df_after['depth'] == depth
            prev_mask = df_after['depth'] == prev_depth
            
            y_curr = df_after.loc[curr_mask, prop].values
            y_prev = df_after.loc[prev_mask, prop].values
            
            # Standardize
            y_curr_std = (y_curr - y_curr.mean()) / y_curr.std()
            y_prev_std = (y_prev - y_prev.mean()) / y_prev.std()
            
            # Apply AR(1)
            y_new_std = rho * y_prev_std + np.sqrt(1 - rho**2) * y_curr_std
            
            # Back-transform
            y_new = y_new_std * y_curr.std() + y_curr.mean()
            
            df_after.loc[curr_mask, prop] = y_new
    
    # Calculate within-horizon correlation AFTER vertical correlation
    print("=" * 80)
    print("WITHIN-HORIZON CORRELATIONS AFTER VERTICAL AR(1):")
    print("(Check for distortion from target)")
    print()
    
    distortions = []
    for depth in [0, 30, 60]:
        depth_data = df_after[df_after['depth'] == depth]
        corr_matrix = depth_data[['clay', 'bulk_density', 'water_retention']].corr().values
        print(f"Depth {depth}cm:")
        print(corr_matrix.round(3))
        
        # Calculate distortion
        if depth > 0:  # Skip first depth (unchanged)
            error = np.abs(corr_matrix - target_corr).mean()
            distortions.append(error)
        print()
    
    # Summary
    print("=" * 80)
    print("ANALYSIS:")
    print("=" * 80)
    
    if distortions:
        avg_distortion = np.mean(distortions)
        print(f"Average correlation distortion: {avg_distortion:.4f}")
        print()
        
        if avg_distortion < 0.05:
            print("✓ MINIMAL DISTORTION - Correlations well preserved")
        elif avg_distortion < 0.15:
            print("⚠ MODERATE DISTORTION - Some correlation structure altered")
        else:
            print("✗ SEVERE DISTORTION - Correlation structure significantly altered")
    
    print()
    print("KEY INSIGHT:")
    print("The naive AR(1) approach applies vertical correlation to each property")
    print("INDEPENDENTLY. This can distort the cross-property correlations that were")
    print("carefully established by the Cholesky decomposition.")
    print()
    print("The degree of distortion depends on:")
    print("  1. How different the rho values are across properties")
    print("  2. The strength of the original correlations")
    print("  3. The number of depth levels")
    print()
    
    return df_before, df_after, target_corr


if __name__ == "__main__":
    df_before, df_after, target_corr = test_correlation_preservation()
