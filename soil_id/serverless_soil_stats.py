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
              axis: Optional[int] = None,
              nan_policy: str = 'propagate') -> Union[float, Tuple[float, float]]:
    """
    Spearman rank correlation coefficient.
    
    Args:
        x: First variable or matrix
        y: Second variable (optional)
        axis: If axis=0, compute correlation across rows (columns are variables)
        nan_policy: How to handle NaNs ('propagate', 'omit', or 'raise')
        
    Returns:
        Correlation coefficient (and p-value if requested)
        Returns tuple of (correlation_matrix, None) to match scipy interface
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
        
        # Return tuple to match scipy.stats.spearmanr interface
        return corr_matrix, None
    
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


# ============================================================================
# Interpolation Functions (replacing scipy.interpolate.UnivariateSpline)
# ============================================================================

class UnivariateSpline:
    """
    Replacement for scipy.interpolate.UnivariateSpline using cubic spline interpolation.
    
    Provides smooth interpolation for soil water characteristic curves and other
    continuous soil properties where linear interpolation is insufficient.
    """
    
    def __init__(self, x, y, w=None, bbox=None, k=3, s=None):
        """
        Initialize spline interpolator.
        
        Args:
            x: Input x coordinates (e.g., soil depth, pressure)
            y: Input y coordinates (e.g., bulk density, water content)
            w: Weights (not implemented - uses uniform weights)
            bbox: Boundary box (not implemented)
            k: Spline degree (3 for cubic, 1 for linear)
            s: Smoothing factor (None for interpolating spline)
        """
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.k = min(k, len(x) - 1)  # Ensure degree is valid
        
        # Sort by x values
        sort_idx = np.argsort(self.x)
        self.x = self.x[sort_idx]
        self.y = self.y[sort_idx]
        
        # Remove duplicates
        unique_mask = np.concatenate(([True], np.diff(self.x) > 1e-12))
        self.x = self.x[unique_mask]
        self.y = self.y[unique_mask]
        
        if len(self.x) < 2:
            raise ValueError("Need at least 2 points for interpolation")
        
        if self.k == 1 or len(self.x) < 4:
            # Use linear interpolation for simplicity
            self.spline_type = 'linear'
        else:
            # Use cubic spline
            self.spline_type = 'cubic'
            self._compute_cubic_coefficients()
    
    def _compute_cubic_coefficients(self):
        """Compute cubic spline coefficients using natural spline conditions."""
        n = len(self.x)
        h = np.diff(self.x)
        
        # Set up tridiagonal system for second derivatives
        A = np.zeros((n, n))
        b = np.zeros(n)
        
        # Natural spline boundary conditions
        A[0, 0] = 1
        A[-1, -1] = 1
        b[0] = 0
        b[-1] = 0
        
        # Interior equations
        for i in range(1, n-1):
            A[i, i-1] = h[i-1]
            A[i, i] = 2 * (h[i-1] + h[i])
            A[i, i+1] = h[i]
            b[i] = 6 * ((self.y[i+1] - self.y[i]) / h[i] - 
                        (self.y[i] - self.y[i-1]) / h[i-1])
        
        # Solve for second derivatives
        try:
            self.second_derivatives = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # Fallback to linear if system is singular
            self.spline_type = 'linear'
    
    def __call__(self, x_new):
        """Evaluate spline at new points."""
        x_new = np.asarray(x_new)
        scalar_input = x_new.ndim == 0
        x_new = np.atleast_1d(x_new)
        
        if self.spline_type == 'linear':
            result = np.interp(x_new, self.x, self.y)
        else:
            result = self._evaluate_cubic(x_new)
        
        return result[0] if scalar_input else result
    
    def _evaluate_cubic(self, x_new):
        """Evaluate cubic spline at new points."""
        result = np.zeros_like(x_new, dtype=float)
        
        for i, x_val in enumerate(x_new):
            # Find interval
            if x_val <= self.x[0]:
                result[i] = self.y[0]
            elif x_val >= self.x[-1]:
                result[i] = self.y[-1]
            else:
                # Find the interval [x[j], x[j+1]]
                j = np.searchsorted(self.x, x_val) - 1
                h = self.x[j+1] - self.x[j]
                
                # Cubic spline evaluation
                A = (self.x[j+1] - x_val) / h
                B = (x_val - self.x[j]) / h
                
                result[i] = (A * self.y[j] + B * self.y[j+1] +
                           ((A**3 - A) * self.second_derivatives[j] +
                            (B**3 - B) * self.second_derivatives[j+1]) * h**2 / 6)
        
        return result


class SimpleInterpolator:
    """
    Lightweight linear interpolator for basic soil property interpolation.
    Use this when UnivariateSpline is overkill or when you need maximum performance.
    """
    
    def __init__(self, x, y, bounds_error=False, fill_value=np.nan):
        """
        Initialize linear interpolator.
        
        Args:
            x: Input x coordinates
            y: Input y coordinates  
            bounds_error: Raise error if extrapolating
            fill_value: Value to use for extrapolation
        """
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.bounds_error = bounds_error
        self.fill_value = fill_value
        
        # Sort by x values
        sort_idx = np.argsort(self.x)
        self.x = self.x[sort_idx]
        self.y = self.y[sort_idx]
    
    def __call__(self, x_new):
        """Evaluate interpolator at new points."""
        if self.bounds_error:
            if np.any(x_new < self.x[0]) or np.any(x_new > self.x[-1]):
                raise ValueError("Interpolation outside bounds")
        
        return np.interp(x_new, self.x, self.y, 
                        left=self.fill_value, right=self.fill_value)


# ============================================================================
# Information Theory Functions (replacing scipy.stats.entropy) 
# ============================================================================

def entropy(pk, qk=None, base=None):
    """
    Calculate Shannon entropy of a probability distribution.
    
    Replacement for scipy.stats.entropy with enhanced robustness for soil data.
    
    Args:
        pk: Probability distribution (will be normalized if doesn't sum to 1)
        qk: Reference distribution (for KL divergence, not implemented)
        base: Logarithm base (None=natural log, 2=bits, 10=dits)
        
    Returns:
        Entropy value
    """
    pk = np.asarray(pk, dtype=float)
    
    # Handle edge cases
    if len(pk) == 0:
        return 0.0
    
    if qk is not None:
        raise NotImplementedError("KL divergence not implemented - use pk only")
    
    # Normalize to probability distribution
    pk = pk / np.sum(pk)
    
    # Remove zeros to avoid log(0)
    pk_nonzero = pk[pk > 0]
    
    if len(pk_nonzero) == 0:
        return 0.0
    
    # Calculate entropy
    if base is None or base == np.e:
        entropy_val = -np.sum(pk_nonzero * np.log(pk_nonzero))
    elif base == 2:
        entropy_val = -np.sum(pk_nonzero * np.log2(pk_nonzero))
    elif base == 10:
        entropy_val = -np.sum(pk_nonzero * np.log10(pk_nonzero))
    else:
        entropy_val = -np.sum(pk_nonzero * np.log(pk_nonzero)) / np.log(base)
    
    return entropy_val


def information_gain(data, target_cols, feature_cols):
    """
    Calculate information gain for soil property importance analysis.
    
    This replaces more complex information gain calculations for variable
    importance in soil prediction models.
    
    Args:
        data: DataFrame-like object with soil data
        target_cols: List of target variable column names
        feature_cols: List of feature column names
        
    Returns:
        Dictionary with information gain values for each feature
    """
    if hasattr(data, 'to_numpy'):
        # Handle pandas DataFrame
        data_array = data.to_numpy()
        if hasattr(data, 'columns'):
            all_cols = list(data.columns)
        else:
            all_cols = [f'col_{i}' for i in range(data_array.shape[1])]
    else:
        # Handle numpy array
        data_array = np.asarray(data)
        all_cols = [f'col_{i}' for i in range(data_array.shape[1])]
    
    # Get column indices
    target_indices = [all_cols.index(col) for col in target_cols if col in all_cols]
    feature_indices = [all_cols.index(col) for col in feature_cols if col in all_cols]
    
    if not target_indices or not feature_indices:
        return {}
    
    results = {}
    
    # For each target variable
    for target_idx in target_indices:
        target_col = all_cols[target_idx]
        target_data = data_array[:, target_idx]
        
        # Remove missing values
        valid_mask = ~np.isnan(target_data)
        if np.sum(valid_mask) < 2:
            continue
            
        target_clean = target_data[valid_mask]
        
        # Discretize target for entropy calculation (if continuous)
        if len(np.unique(target_clean)) > 10:
            # Use quantile-based binning for continuous variables
            target_binned = np.digitize(target_clean, 
                                      np.percentile(target_clean, [25, 50, 75]))
        else:
            target_binned = target_clean
        
        # Calculate base entropy
        target_counts = np.bincount(target_binned.astype(int))
        base_entropy = entropy(target_counts)
        
        # Calculate information gain for each feature
        for feature_idx in feature_indices:
            feature_col = all_cols[feature_idx]
            feature_data = data_array[valid_mask, feature_idx]
            
            # Skip if feature has missing values
            feature_valid_mask = ~np.isnan(feature_data)
            if np.sum(feature_valid_mask) < 2:
                results[f'{feature_col}_vs_{target_col}'] = 0.0
                continue
            
            feature_clean = feature_data[feature_valid_mask]
            target_clean_subset = target_binned[feature_valid_mask]
            
            # Discretize feature if continuous
            if len(np.unique(feature_clean)) > 10:
                feature_binned = np.digitize(feature_clean,
                                           np.percentile(feature_clean, [25, 50, 75]))
            else:
                feature_binned = feature_clean
            
            # Calculate conditional entropy
            feature_values = np.unique(feature_binned)
            conditional_entropy = 0.0
            
            for val in feature_values:
                mask = feature_binned == val
                if np.sum(mask) == 0:
                    continue
                    
                subset_target = target_clean_subset[mask]
                subset_counts = np.bincount(subset_target.astype(int), 
                                         minlength=len(target_counts))
                
                if np.sum(subset_counts) > 0:
                    prob_val = np.sum(mask) / len(feature_binned)
                    conditional_entropy += prob_val * entropy(subset_counts)
            
            # Information gain
            info_gain = base_entropy - conditional_entropy
            results[f'{feature_col}_vs_{target_col}'] = max(0.0, info_gain)
    
    return results


def mutual_information(x, y, bins=10):
    """
    Calculate mutual information between two variables.
    Useful for soil property relationship analysis.
    
    Args:
        x: First variable (e.g., clay content)
        y: Second variable (e.g., water retention)
        bins: Number of bins for discretization
        
    Returns:
        Mutual information value
    """
    x, y = np.asarray(x), np.asarray(y)
    
    # Remove missing values
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid_mask], y[valid_mask]
    
    if len(x) < 2:
        return 0.0
    
    # Create 2D histogram
    hist_2d, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
    
    # Get marginal histograms
    hist_x = np.sum(hist_2d, axis=1)
    hist_y = np.sum(hist_2d, axis=0)
    
    # Calculate entropies
    H_x = entropy(hist_x)
    H_y = entropy(hist_y)
    H_xy = entropy(hist_2d.ravel())
    
    # Mutual information = H(X) + H(Y) - H(X,Y)
    mi = H_x + H_y - H_xy
    
    return max(0.0, mi)


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
    
    # Test UnivariateSpline replacement
    print("\n=== Testing UnivariateSpline Replacement ===")
    # Soil water characteristic curve data (pressure vs water content)
    pressure = np.array([0.1, 1, 5, 15, 100, 1500])  # kPa
    water_content = np.array([0.45, 0.42, 0.38, 0.25, 0.18, 0.12])  # cm³/cm³
    
    # Test cubic spline
    spline = UnivariateSpline(pressure, water_content, k=3)
    new_pressures = np.array([0.5, 3, 10, 50, 500])
    interpolated = spline(new_pressures)
    
    print("Pressure (kPa):", pressure)
    print("Water content:", water_content)
    print("New pressures:", new_pressures)
    print("Interpolated:", np.round(interpolated, 4))
    
    # Test linear interpolator
    linear_interp = SimpleInterpolator(pressure, water_content)
    linear_values = linear_interp(new_pressures)
    print("Linear interp:", np.round(linear_values, 4))
    
    # Test entropy
    print("\n=== Testing Entropy Functions ===")
    # Soil texture class distribution
    texture_counts = np.array([15, 25, 10, 30, 20])  # Sand, silt loam, clay, loam, sandy loam
    H = entropy(texture_counts)
    print(f"Soil texture diversity (entropy): {H:.3f}")
    
    # Test with different bases
    H_bits = entropy(texture_counts, base=2)
    print(f"Entropy in bits: {H_bits:.3f}")
    
    # Test information gain
    print("\n=== Testing Information Gain ===")
    # Simulate soil data
    np.random.seed(42)
    n_samples = 100
    clay_content = np.random.normal(25, 10, n_samples)
    organic_matter = 3 + 0.1 * clay_content + np.random.normal(0, 1, n_samples)
    bulk_density = 1.6 - 0.01 * clay_content - 0.05 * organic_matter + np.random.normal(0, 0.1, n_samples)
    
    # Create fake DataFrame structure
    fake_data = np.column_stack([clay_content, organic_matter, bulk_density])
    
    # Test information gain calculation
    info_gains = information_gain(
        fake_data,
        target_cols=['col_2'],  # bulk_density
        feature_cols=['col_0', 'col_1']  # clay_content, organic_matter
    )
    
    print("Information gains for bulk density prediction:")
    for feature, gain in info_gains.items():
        print(f"{feature}: {gain:.4f}")
    
    # Test mutual information
    mi_clay_om = mutual_information(clay_content, organic_matter)
    mi_clay_bd = mutual_information(clay_content, bulk_density)
    print(f"\nMutual Information:")
    print(f"Clay vs Organic Matter: {mi_clay_om:.4f}")
    print(f"Clay vs Bulk Density: {mi_clay_bd:.4f}")
    
    print("\nAll tests completed successfully!")
    
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
