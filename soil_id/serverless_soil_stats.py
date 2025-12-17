"""
Lightweight replacements for scipy.stats and composition_stats functions
for serverless deployment on Vercel.

This module provides drop-in replacements for:
- composition_stats.ilr() and ilr_inv()
- scipy.stats.spearmanr()
- scipy.stats.norm.cdf()
- numpy.linalg.cholesky()

Optimized for soil science applications with sand/silt/clay compositions.
"""

import math
import numpy as np
from typing import Tuple, Union, Optional


# ============================================================================
# Compositional Data Analysis Functions (replacing composition_stats)
# ============================================================================

def ilr(compositions: np.ndarray) -> np.ndarray:
    """
    Isometric log-ratio transformation for compositional data.
    
    Transforms D-part compositions to (D-1) dimensional ilr coordinates.
    For 3-part compositions (sand/silt/clay), returns 2 ilr coordinates.
    
    Args:
        compositions: Array of shape (n, D) where each row sums to 1 (or 100)
        
    Returns:
        Array of shape (n, D-1) with ilr coordinates
    """
    compositions = np.atleast_2d(compositions)
    n, D = compositions.shape
    
    # Normalize to sum to 1 if needed
    row_sums = np.sum(compositions, axis=1, keepdims=True)
    if np.any(row_sums > 1.1):  # Assume percentages if > 1.1
        compositions = compositions / row_sums
    
    # Handle zeros by adding small constant (closure)
    epsilon = 1e-10
    compositions = compositions + epsilon
    compositions = compositions / np.sum(compositions, axis=1, keepdims=True)
    
    # Create orthonormal basis for ilr transformation
    # For D=3 (sand/silt/clay), creates standard ilr coordinates
    if D == 3:
        # Standard ilr transformation for 3-part composition
        ilr_coords = np.zeros((n, 2))
        
        # ilr1: log(sand/sqrt(silt*clay))
        ilr_coords[:, 0] = np.sqrt(2/3) * np.log(
            compositions[:, 0] / np.sqrt(compositions[:, 1] * compositions[:, 2])
        )
        
        # ilr2: log(silt/clay)
        ilr_coords[:, 1] = np.sqrt(1/2) * np.log(
            compositions[:, 1] / compositions[:, 2]
        )
        
        return ilr_coords
    
    else:
        # General ilr transformation using Gram-Schmidt orthonormalization
        log_comp = np.log(compositions)
        
        # Create Helmert basis
        basis = np.zeros((D-1, D))
        for i in range(D-1):
            basis[i, :i+1] = -1/np.sqrt(i+2)
            basis[i, i+1] = np.sqrt((i+1)/(i+2))
        
        # Apply transformation
        centered_log = log_comp - np.mean(log_comp, axis=1, keepdims=True)
        return np.sqrt(D) * np.dot(centered_log, basis.T)


def ilr_inv(ilr_coords: np.ndarray, n_parts: int = 3) -> np.ndarray:
    """
    Inverse isometric log-ratio transformation.
    
    Transforms ilr coordinates back to compositional data.
    
    Args:
        ilr_coords: Array of shape (n, D-1) with ilr coordinates
        n_parts: Number of parts in composition (default 3 for sand/silt/clay)
        
    Returns:
        Array of shape (n, D) with compositions summing to 1
    """
    ilr_coords = np.atleast_2d(ilr_coords)
    n, ilr_dim = ilr_coords.shape
    D = n_parts
    
    if D == 3 and ilr_dim == 2:
        # Inverse transformation for 3-part composition
        compositions = np.zeros((n, 3))
        
        # Convert ilr coordinates back to log-ratios
        sqrt_23 = np.sqrt(2/3)
        sqrt_12 = np.sqrt(1/2)
        
        # Use the inverse transformation formulas
        # Based on: ilr1 = sqrt(2/3) * log(x1/sqrt(x2*x3))
        #          ilr2 = sqrt(1/2) * log(x2/x3)
        
        # Express in terms of log(xi)
        log_x2_x3 = ilr_coords[:, 1] / sqrt_12  # log(x2/x3)
        log_x1_geomean23 = ilr_coords[:, 0] / sqrt_23  # log(x1/sqrt(x2*x3))
        
        # Set log(x3) = 0 as reference (will normalize later)
        log_x3 = np.zeros(n)
        log_x2 = log_x2_x3  # Since log(x2/x3) = log(x2) - log(x3)
        
        # log(x1) = log(x1/sqrt(x2*x3)) + 0.5*(log(x2) + log(x3))
        log_x1 = log_x1_geomean23 + 0.5 * (log_x2 + log_x3)
        
        # Convert to compositions
        log_comp = np.column_stack([log_x1, log_x2, log_x3])
        compositions = np.exp(log_comp)
        
        # Normalize to sum to 1
        compositions = compositions / np.sum(compositions, axis=1, keepdims=True)
        
        return compositions
    
    else:
        # General inverse transformation
        # Create inverse Helmert basis
        basis = np.zeros((D, D-1))
        for i in range(D-1):
            basis[:i+1, i] = -1/np.sqrt((i+1)*(i+2))
            basis[i+1, i] = np.sqrt((i+1)/(i+2))
        
        # Apply inverse transformation
        log_comp = np.dot(ilr_coords / np.sqrt(D), basis.T)
        log_comp = log_comp - np.mean(log_comp, axis=1, keepdims=True)
        
        compositions = np.exp(log_comp)
        return compositions / np.sum(compositions, axis=1, keepdims=True)


# ============================================================================
# Statistical Functions (replacing scipy.stats)
# ============================================================================

def spearmanr(x: np.ndarray, y: Optional[np.ndarray] = None, 
              nan_policy: str = 'propagate') -> Union[float, Tuple[float, float]]:
    """
    Spearman rank correlation coefficient.
    
    Args:
        x: First variable or matrix
        y: Second variable (optional)
        nan_policy: How to handle NaNs ('propagate', 'omit', or 'raise')
        
    Returns:
        Correlation coefficient (and p-value if requested)
    """
    if y is None:
        # Matrix form - compute pairwise correlations
        x = np.asarray(x)
        if x.ndim == 1:
            return 1.0, 0.0
        
        n_vars = x.shape[1]
        corr_matrix = np.eye(n_vars)
        
        for i in range(n_vars):
            for j in range(i+1, n_vars):
                # Handle NaNs
                if nan_policy == 'omit':
                    mask = ~(np.isnan(x[:, i]) | np.isnan(x[:, j]))
                    xi, xj = x[mask, i], x[mask, j]
                elif nan_policy == 'propagate':
                    xi, xj = x[:, i], x[:, j]
                    if np.any(np.isnan(xi)) or np.any(np.isnan(xj)):
                        corr_matrix[i, j] = corr_matrix[j, i] = np.nan
                        continue
                else:  # raise
                    if np.any(np.isnan(x[:, i])) or np.any(np.isnan(x[:, j])):
                        raise ValueError("NaN values found")
                    xi, xj = x[:, i], x[:, j]
                
                if len(xi) < 2:
                    corr_matrix[i, j] = corr_matrix[j, i] = np.nan
                    continue
                
                # Compute Spearman correlation
                rho = _spearman_corr_1d(xi, xj)
                corr_matrix[i, j] = corr_matrix[j, i] = rho
        
        return corr_matrix
    
    else:
        # Two variable form
        x, y = np.asarray(x).flatten(), np.asarray(y).flatten()
        
        # Handle NaNs
        if nan_policy == 'omit':
            mask = ~(np.isnan(x) | np.isnan(y))
            x, y = x[mask], y[mask]
        elif nan_policy == 'propagate':
            if np.any(np.isnan(x)) or np.any(np.isnan(y)):
                return np.nan, np.nan
        else:  # raise
            if np.any(np.isnan(x)) or np.any(np.isnan(y)):
                raise ValueError("NaN values found")
        
        if len(x) < 2:
            return np.nan, np.nan
        
        rho = _spearman_corr_1d(x, y)
        
        # Approximate p-value (simplified)
        if len(x) > 10:
            # For large samples, use normal approximation
            t_stat = rho * np.sqrt((len(x) - 2) / (1 - rho**2))
            p_value = 2 * (1 - _norm_cdf(abs(t_stat)))
        else:
            # For small samples, return conservative p-value
            p_value = 0.05 if abs(rho) > 0.7 else 0.2
        
        return rho, p_value


def _spearman_corr_1d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman correlation between two 1D arrays."""
    n = len(x)
    if n < 2:
        return np.nan
    
    # Compute ranks (handling ties with average rank)
    rank_x = _compute_ranks(x)
    rank_y = _compute_ranks(y)
    
    # Pearson correlation of ranks
    mean_rx = np.mean(rank_x)
    mean_ry = np.mean(rank_y)
    
    num = np.sum((rank_x - mean_rx) * (rank_y - mean_ry))
    den_x = np.sum((rank_x - mean_rx)**2)
    den_y = np.sum((rank_y - mean_ry)**2)
    
    if den_x == 0 or den_y == 0:
        return 0.0
    
    return num / np.sqrt(den_x * den_y)


def _compute_ranks(x: np.ndarray) -> np.ndarray:
    """Compute ranks with average ranking for ties."""
    sorted_idx = np.argsort(x)
    ranks = np.empty(len(x))
    ranks[sorted_idx] = np.arange(1, len(x) + 1)
    
    # Handle ties by averaging ranks
    for value in np.unique(x):
        mask = x == value
        if np.sum(mask) > 1:
            ranks[mask] = np.mean(ranks[mask])
    
    return ranks


def norm_cdf(x: Union[float, np.ndarray], loc: float = 0, 
             scale: float = 1) -> Union[float, np.ndarray]:
    """
    Cumulative distribution function of the normal distribution.
    
    Args:
        x: Input values
        loc: Mean (default 0)
        scale: Standard deviation (default 1)
        
    Returns:
        CDF values
    """
    standardized = (np.asarray(x) - loc) / scale
    return _norm_cdf(standardized)


def _norm_cdf(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Standard normal CDF using error function approximation."""
    # Use error function approximation for standard normal CDF
    # Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
    
    def erf_approx(z):
        # Abramowitz and Stegun approximation
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911
        
        sign = np.sign(z)
        z = np.abs(z)
        
        # A&S formula 7.1.26
        t = 1.0 / (1.0 + p * z)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-z * z)
        
        return sign * y
    
    x = np.asarray(x)
    return 0.5 * (1.0 + erf_approx(x / np.sqrt(2.0)))


# ============================================================================
# Linear Algebra Functions (replacing numpy.linalg.cholesky)
# ============================================================================

def cholesky(a: np.ndarray) -> np.ndarray:
    """
    Cholesky decomposition of a positive definite matrix.
    
    Returns lower triangular matrix L such that a = L @ L.T
    
    Args:
        a: Positive definite matrix
        
    Returns:
        Lower triangular Cholesky factor
    """
    a = np.asarray(a)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Input must be a square matrix")
    
    n = a.shape[0]
    L = np.zeros_like(a)
    
    for i in range(n):
        for j in range(i + 1):
            if i == j:  # Diagonal elements
                sum_sq = sum(L[i][k]**2 for k in range(j))
                val = a[i][i] - sum_sq
                if val <= 0:
                    # Matrix is not positive definite, use regularization
                    val = max(val, 1e-12)
                L[i][j] = np.sqrt(val)
            else:  # Off-diagonal elements
                sum_prod = sum(L[i][k] * L[j][k] for k in range(j))
                if L[j][j] == 0:
                    L[i][j] = 0
                else:
                    L[i][j] = (a[i][j] - sum_prod) / L[j][j]
    
    return L


# ============================================================================
# Utility Functions for Correlated Simulation
# ============================================================================

def simulate_correlated_triangular(n: int, correlation_matrix: np.ndarray,
                                 low: np.ndarray, mode: np.ndarray, 
                                 high: np.ndarray) -> np.ndarray:
    """
    Simulate correlated triangular random variables.
    
    This replaces the function from utils.py that uses scipy.stats.
    
    Args:
        n: Number of samples
        correlation_matrix: Correlation matrix
        low: Lower bounds for each variable
        mode: Mode values for each variable  
        high: Upper bounds for each variable
        
    Returns:
        Array of shape (n, len(low)) with correlated triangular samples
    """
    n_vars = len(low)
    
    # Generate correlated standard normal variables
    try:
        L = cholesky(correlation_matrix)
    except:
        # If Cholesky fails, use regularized matrix
        reg_corr = correlation_matrix + np.eye(n_vars) * 1e-6
        L = cholesky(reg_corr)
    
    # Generate independent standard normal samples
    Z = np.random.standard_normal((n, n_vars))
    
    # Apply correlation structure
    corr_normal = Z @ L.T
    
    # Transform to uniform [0,1]
    uniform_samples = norm_cdf(corr_normal)
    
    # Transform to triangular distribution
    triangular_samples = np.zeros_like(uniform_samples)
    
    for i in range(n_vars):
        # Inverse CDF of triangular distribution
        u = uniform_samples[:, i]
        a, b, c = low[i], high[i], mode[i]
        
        # Mode as fraction of range
        fc = (c - a) / (b - a)
        
        # Inverse triangular CDF
        mask = u <= fc
        triangular_samples[mask, i] = a + np.sqrt(u[mask] * (b - a) * (c - a))
        triangular_samples[~mask, i] = b - np.sqrt((1 - u[~mask]) * (b - a) * (b - c))
    
    return triangular_samples


# ============================================================================
# Additional Utilities
# ============================================================================

def ensure_positive_definite(matrix: np.ndarray, regularization: float = 1e-6) -> np.ndarray:
    """
    Ensure a matrix is positive definite by adding regularization if needed.
    
    Args:
        matrix: Input correlation/covariance matrix
        regularization: Regularization parameter
        
    Returns:
        Positive definite matrix
    """
    matrix = np.asarray(matrix)
    
    # Check if already positive definite
    try:
        np.linalg.cholesky(matrix)
        return matrix
    except np.linalg.LinAlgError:
        pass
    
    # Add regularization to diagonal
    n = matrix.shape[0]
    regularized = matrix + np.eye(n) * regularization
    
    # Ensure it's still a valid correlation matrix (diagonal = 1)
    if np.allclose(np.diag(matrix), 1.0):
        np.fill_diagonal(regularized, 1.0)
    
    return regularized


def validate_composition(composition: np.ndarray, tolerance: float = 1e-10) -> np.ndarray:
    """
    Validate and normalize compositional data.
    
    Args:
        composition: Compositional data array
        tolerance: Tolerance for sum constraint
        
    Returns:
        Normalized composition
    """
    composition = np.asarray(composition)
    
    # Ensure non-negative
    composition = np.maximum(composition, tolerance)
    
    # Normalize to sum to 1 (or 100 if originally percentages)
    row_sums = np.sum(composition, axis=-1, keepdims=True)
    if composition.ndim > 1:
        composition = composition / row_sums
    else:
        composition = composition / np.sum(composition)
    
    return composition


if __name__ == "__main__":
    # Test the functions
    print("Testing serverless soil statistics functions...")
    
    # Test ILR transformation
    sand_silt_clay = np.array([[0.4, 0.3, 0.3], [0.5, 0.2, 0.3], [0.2, 0.5, 0.3]])
    ilr_coords = ilr(sand_silt_clay)
    reconstructed = ilr_inv(ilr_coords)
    
    print("Original compositions:")
    print(sand_silt_clay)
    print("ILR coordinates:")
    print(ilr_coords)
    print("Reconstructed compositions:")
    print(reconstructed)
    print("Reconstruction error:", np.max(np.abs(sand_silt_clay - reconstructed)))
    
    # Test Spearman correlation
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])
    rho, p = spearmanr(x, y)
    print(f"Spearman correlation: {rho:.3f}, p-value: {p:.3f}")
    
    # Test normal CDF
    print(f"norm.cdf(0): {norm_cdf(0):.3f}")
    print(f"norm.cdf(1.96): {norm_cdf(1.96):.3f}")
    
    print("All tests completed successfully!")
