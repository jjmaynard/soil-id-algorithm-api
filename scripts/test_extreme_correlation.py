"""
Test correlation preservation under extreme conditions.

This tests worst-case scenarios where vertical correlation could significantly
distort within-horizon correlations.
"""

import numpy as np
import pandas as pd


def test_extreme_case():
    """
    Test with:
    1. Very different rho values (0.9 vs 0.3)
    2. Strong original correlations (0.95)
    3. Many depth levels
    """
    print("=" * 80)
    print("EXTREME CASE TEST: Varying Rho Values")
    print("=" * 80)
    print()
    
    np.random.seed(42)
    n_sim = 1000
    
    # Strong correlation between properties
    target_corr = np.array([
        [1.0,  0.95],   # Very strong positive correlation
        [0.95, 1.0],
    ])
    
    print("TARGET CORRELATION: 0.95")
    print()
    
    # Test different rho combinations
    test_cases = [
        ("Matched rho (0.80, 0.80)", 0.80, 0.80),
        ("Similar rho (0.80, 0.75)", 0.80, 0.75),
        ("Different rho (0.90, 0.60)", 0.90, 0.60),
        ("Very different rho (0.95, 0.30)", 0.95, 0.30),
    ]
    
    results = []
    
    for case_name, rho1, rho2 in test_cases:
        # Generate data with Cholesky
        data = []
        depths = [0, 20, 40, 60, 80, 100]  # More depths
        
        for depth in depths:
            uncorrelated = np.random.normal(size=(n_sim, 2))
            L = np.linalg.cholesky(target_corr)
            correlated = uncorrelated @ L.T
            
            prop1 = correlated[:, 0] * 10 + 50
            prop2 = correlated[:, 1] * 10 + 50
            
            for i in range(n_sim):
                data.append({
                    'depth': depth,
                    'prop1': prop1[i],
                    'prop2': prop2[i],
                })
        
        df = pd.DataFrame(data)
        
        # Apply AR(1) with different rho values
        for i, depth in enumerate(depths[1:], start=1):
            prev_depth = depths[i-1]
            
            # Property 1 with rho1
            curr_mask = df['depth'] == depth
            prev_mask = df['depth'] == prev_depth
            
            for prop, rho in [('prop1', rho1), ('prop2', rho2)]:
                y_curr = df.loc[curr_mask, prop].values
                y_prev = df.loc[prev_mask, prop].values
                
                y_curr_std = (y_curr - y_curr.mean()) / y_curr.std()
                y_prev_std = (y_prev - y_prev.mean()) / y_prev.std()
                
                y_new_std = rho * y_prev_std + np.sqrt(1 - rho**2) * y_curr_std
                y_new = y_new_std * y_curr.std() + y_curr.mean()
                
                df.loc[curr_mask, prop] = y_new
        
        # Check correlation at each depth
        distortions = []
        for depth in depths[1:]:  # Skip first (unchanged)
            depth_data = df[df['depth'] == depth]
            actual_corr = depth_data[['prop1', 'prop2']].corr().iloc[0, 1]
            distortion = abs(actual_corr - 0.95)
            distortions.append(distortion)
        
        avg_distortion = np.mean(distortions)
        max_distortion = np.max(distortions)
        
        results.append({
            'case': case_name,
            'rho1': rho1,
            'rho2': rho2,
            'rho_diff': abs(rho1 - rho2),
            'avg_distortion': avg_distortion,
            'max_distortion': max_distortion,
        })
        
        print(f"{case_name}:")
        print(f"  Rho difference: {abs(rho1-rho2):.2f}")
        print(f"  Avg distortion: {avg_distortion:.4f}")
        print(f"  Max distortion: {max_distortion:.4f}")
        
        if avg_distortion < 0.05:
            print(f"  Status: ✓ Acceptable")
        elif avg_distortion < 0.15:
            print(f"  Status: ⚠ Moderate concern")
        else:
            print(f"  Status: ✗ Significant problem")
        print()
    
    # Analysis
    print("=" * 80)
    print("FINDINGS:")
    print("=" * 80)
    
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    print()
    
    print("CONCLUSION:")
    print("-" * 80)
    
    # Check relationship between rho_diff and distortion
    if df_results['rho_diff'].corr(df_results['avg_distortion']) > 0.7:
        print("✗ CONFIRMED: Larger differences in rho values cause more distortion")
    else:
        print("✓ GOOD: Distortion is minimal even with different rho values")
    
    print()
    print("RECOMMENDATION:")
    if df_results['avg_distortion'].max() < 0.10:
        print("✓ Current AR(1) implementation is ACCEPTABLE")
        print("  Distortion remains < 0.10 even in extreme cases")
    else:
        print("⚠ Consider alternative approach for high-correlation properties")
    
    return results


if __name__ == "__main__":
    results = test_extreme_case()
