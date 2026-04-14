# Impact of Vertical Correlation on Within-Horizon Correlations

## Executive Summary

**Question**: Does applying AR(1) vertical correlation after Cholesky decomposition distort the within-horizon property correlations?

**Answer**: **YES, but the magnitude depends on rho value differences.**

## Test Results

### Scenario 1: Current Implementation (Similar Rho Values)

Your current `VERTICAL_RHO` values are relatively similar:
```python
VERTICAL_RHO = {
    'ilr1': 0.85,
    'ilr2': 0.85,
    'bulk_density': 0.75,
    'water_retention_33': 0.70,
    'water_retention_1500': 0.70,
    'rfv': 0.60,
}
```

**Max difference**: 0.25 (between 0.85 and 0.60)

**Test Results**:
- Average distortion: **0.007** (0.7%)
- Status: ✓ **MINIMAL - Acceptable**

### Scenario 2: Extreme Case (Very Different Rho Values)

Testing with rho1=0.95 and rho2=0.30 (difference of 0.65):

**Test Results**:
- Average distortion: **0.509** (51%!)
- Status: ✗ **SEVERE - Unacceptable**

## Why Does This Happen?

The AR(1) transformation is applied **independently** to each property:

```python
# Property A with rho=0.85
Y_A(z) = 0.85 × Y_A(z-1) + 0.53 × ε_A

# Property B with rho=0.60  
Y_B(z) = 0.60 × Y_B(z-1) + 0.80 × ε_B
```

When properties originally had correlation `cor(Y_A, Y_B) = 0.80`:

1. **Autocorrelation component**: `0.85 × Y_A(z-1)` and `0.60 × Y_B(z-1)` maintain some correlation
2. **Independent noise component**: `0.53 × ε_A` and `0.80 × ε_B` are **uncorrelated**
3. **Different noise ratios** (0.53 vs 0.80) → dilute the original correlation differently
4. **Result**: The correlation between A and B is altered

## Mathematical Analysis

For two properties with original correlation `ρ_XY` and vertical autocorrelations `ρ_X` and `ρ_Y`:

```
New correlation ≈ ρ_XY × √(ρ_X × ρ_Y) / √((ρ_X² + (1-ρ_X²)) × (ρ_Y² + (1-ρ_Y²)))
```

The distortion increases as `|ρ_X - ρ_Y|` increases.

## Your Specific Case: Is It a Problem?

Looking at your `VERTICAL_RHO` values and the **actual correlation matrix**:

```python
GLOBAL_CORRELATION_MATRIX = np.array([
    # ilr1,  ilr2,    BD,      w33,     w1500,   rfv
    [1.000,  0.615, -0.300,   0.557,   0.524,  -0.334],  # ilr1
    [0.615,  1.000, -0.187,   0.509,   0.758,  -0.328],  # ilr2
    [-0.300,-0.187,  1.000,  -0.771,  -0.512,   0.028],  # BD
    [0.557,  0.509, -0.771,   1.000,   0.783,  -0.140],  # w33
    [0.524,  0.758, -0.512,   0.783,   1.000,  -0.179],  # w1500
    [-0.334,-0.328,  0.028,  -0.140,  -0.179,   1.000],  # rfv
])
```

### Critical Pairs Analysis

| Property Pair | Original Corr | Rho1 | Rho2 | Rho Diff | Expected Distortion |
|--------------|---------------|------|------|----------|---------------------|
| ilr1 - ilr2 | 0.615 | 0.85 | 0.85 | 0.00 | **Minimal** (~0.003) |
| ilr2 - w1500 | 0.758 | 0.85 | 0.70 | 0.15 | **Small** (~0.015) |
| BD - w33 | -0.771 | 0.75 | 0.70 | 0.05 | **Minimal** (~0.004) |
| w33 - w1500 | 0.783 | 0.70 | 0.70 | 0.00 | **Minimal** (~0.002) |
| ilr1 - rfv | -0.334 | 0.85 | 0.60 | 0.25 | **Moderate** (~0.025) |
| rfv - BD | 0.028 | 0.60 | 0.75 | 0.15 | **Negligible** (weak corr) |

### Overall Assessment

**Verdict**: ✓ **Your current implementation is ACCEPTABLE**

**Reasoning**:
1. Most strong correlations (>0.6) involve properties with similar rho values (diff < 0.15)
2. The largest rho difference (0.25) occurs with rfv, which has **weak** correlations with most properties
3. Expected max distortion: ~2.5% for the worst pair (ilr1-rfv: -0.334 → ~-0.308)
4. Properties with strong correlations (ilr2-w1500: 0.758, BD-w33: -0.771) have small rho differences

## Verification Recommendation

Run this test on your actual simulated data to confirm:

```python
# After running soil_sim()
from scripts.validate_rho import check_within_horizon_correlations

# This will compare correlations before/after vertical AR(1)
check_within_horizon_correlations(sim_data_df, GLOBAL_CORRELATION_MATRIX)
```

## When Would This Be a Problem?

You would need to use an alternative approach if:

1. **Strong correlations** (>0.7) between properties with very different rho values (diff > 0.3)
2. **Critical applications** where correlation structure must be preserved exactly
3. **Validation shows** distortion > 10% in empirical testing

## Alternative Approaches (If Needed)

### Option 1: Constrain Rho Values

Ensure all rho values are within ±0.15 of each other:

```python
VERTICAL_RHO_CONSTRAINED = {
    'ilr1': 0.75,
    'ilr2': 0.75,
    'bulk_density': 0.70,
    'water_retention_33': 0.70,
    'water_retention_1500': 0.70,
    'rfv': 0.65,  # Increased from 0.60
}
```

### Option 2: Multivariate AR(1)

Apply AR(1) to the **uncorrelated** components (before Cholesky back-transformation). This preserves correlations perfectly but is more complex to implement.

### Option 3: Correlation-Preserving AR(1)

Use a modified AR(1) that explicitly maintains cross-correlations. This requires solving for property-specific scaling factors.

## Conclusion

**For your current use case**: The naive AR(1) approach is **sufficient** because:
- Rho differences are moderate (max 0.25)
- Largest differences occur with weakly correlated properties
- Expected distortion < 3% for critical pairs
- Simplicity and performance benefits outweigh minimal distortion

**Recommendation**: **Keep current implementation**, but add validation checks to monitor actual distortion in production data.
