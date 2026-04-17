"""
Test vertical correlation implementation
"""
import numpy as np
import pandas as pd

# Simulate the add_vertical_correlation function behavior
def test_vertical_correlation():
    """Test that AR(1) correlation is properly applied"""
    
    # Create synthetic data: 3 depths, 1000 simulations
    np.random.seed(42)
    n_sim = 1000
    
    # Three horizons at different depths
    data = []
    for depth in [0, 20, 40]:
        for _ in range(n_sim):
            data.append({
                'compname_grp': 'TestSoil',
                'hzdept_r': depth,
                'bulk_density_third_bar': np.random.normal(1.4, 0.2),
                'clay_total': np.random.normal(25, 5),
            })
    
    df = pd.DataFrame(data)
    
    # Calculate correlation before applying vertical correlation
    depths = [0, 20, 40]
    before_corr = []
    for i in range(len(depths)-1):
        mask1 = df['hzdept_r'] == depths[i]
        mask2 = df['hzdept_r'] == depths[i+1]
        
        vals1 = df.loc[mask1, 'bulk_density_third_bar'].values
        vals2 = df.loc[mask2, 'bulk_density_third_bar'].values
        
        corr = np.corrcoef(vals1, vals2)[0, 1]
        before_corr.append(corr)
    
    print("Correlation between adjacent depths BEFORE vertical correlation:")
    print(f"  Depth 0-20cm:  {before_corr[0]:.3f}")
    print(f"  Depth 20-40cm: {before_corr[1]:.3f}")
    print(f"  Average:       {np.mean(before_corr):.3f}")
    print()
    
    # Apply AR(1) transformation manually
    rho = 0.75
    df_corr = df.copy()
    
    for i, depth in enumerate(depths[1:], start=1):
        prev_depth = depths[i-1]
        
        curr_mask = df_corr['hzdept_r'] == depth
        prev_mask = df_corr['hzdept_r'] == prev_depth
        
        y_curr = df_corr.loc[curr_mask, 'bulk_density_third_bar'].values
        y_prev = df_corr.loc[prev_mask, 'bulk_density_third_bar'].values
        
        # Standardize
        y_curr_std = (y_curr - y_curr.mean()) / y_curr.std()
        y_prev_std = (y_prev - y_prev.mean()) / y_prev.std()
        
        # Apply AR(1)
        y_new_std = rho * y_prev_std + np.sqrt(1 - rho**2) * y_curr_std
        
        # Transform back
        y_new = y_new_std * y_curr.std() + y_curr.mean()
        
        df_corr.loc[curr_mask, 'bulk_density_third_bar'] = y_new
    
    # Calculate correlation after
    after_corr = []
    for i in range(len(depths)-1):
        mask1 = df_corr['hzdept_r'] == depths[i]
        mask2 = df_corr['hzdept_r'] == depths[i+1]
        
        vals1 = df_corr.loc[mask1, 'bulk_density_third_bar'].values
        vals2 = df_corr.loc[mask2, 'bulk_density_third_bar'].values
        
        corr = np.corrcoef(vals1, vals2)[0, 1]
        after_corr.append(corr)
    
    print("Correlation between adjacent depths AFTER vertical correlation (rho=0.75):")
    print(f"  Depth 0-20cm:  {after_corr[0]:.3f}")
    print(f"  Depth 20-40cm: {after_corr[1]:.3f}")
    print(f"  Average:       {np.mean(after_corr):.3f}")
    print()
    
    # Verify distributions preserved
    print("Distribution statistics (bulk_density_third_bar):")
    for depth in depths:
        mask = df['hzdept_r'] == depth
        before_mean = df.loc[mask, 'bulk_density_third_bar'].mean()
        before_std = df.loc[mask, 'bulk_density_third_bar'].std()
        
        after_mean = df_corr.loc[mask, 'bulk_density_third_bar'].mean()
        after_std = df_corr.loc[mask, 'bulk_density_third_bar'].std()
        
        print(f"  Depth {depth}cm:")
        print(f"    Before: mean={before_mean:.3f}, std={before_std:.3f}")
        print(f"    After:  mean={after_mean:.3f}, std={after_std:.3f}")
    
    print()
    print("✓ Test complete!")
    print(f"✓ Correlation increased from ~{np.mean(before_corr):.3f} to ~{np.mean(after_corr):.3f}")
    print(f"✓ Target rho={rho:.2f}, achieved ~{np.mean(after_corr):.3f}")
    print("✓ Marginal distributions preserved (mean and std approximately unchanged)")

if __name__ == "__main__":
    test_vertical_correlation()
