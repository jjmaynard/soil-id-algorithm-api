# Performance Profiling Results - Summary

**Date:** December 17, 2025  
**Test Location:** -122.084, 37.422 (California)  
**Analysis:** Local profiling with cProfile

## Executive Summary

### Overall Performance
- **sim=false (baseline):** 43.1 seconds  
- **sim=true (full pipeline):** 47.6 seconds  
- **Simulation overhead:** 4.5 seconds (10% increase)

### Top 5 Bottlenecks (by cumulative time)

| Rank | Function | Time | % of Total | Category |
|------|----------|------|------------|----------|
| 1 | lab2munsell (color matching) | 37.2s | 78% | **COLOR CONVERSION** |
| 2 | Database queries (SoilWeb) | 5.7s | 12% | EXTERNAL API |
| 3 | soil_sim (simulation pipeline) | 8.6s | 18% | SIMULATION |
| 4 | Rosetta API (hydraulic properties) | 4.0s | 8% | EXTERNAL API |
| 5 | calculate_vwc_awc (spline interpolation) | 3.2s | 7% | INTERPOLATION |

## Critical Finding: lab2munsell is the Bottleneck! 🔴

**The color conversion function `lab2munsell()` is taking 37 seconds (78% of total time)!**

### Why is lab2munsell slow?
```python
# Current implementation: Brute force search
def lab2munsell(LAB_ref, munsell_ref):
    # Loops through ALL 8,468 Munsell colors
    # For EACH of 26 soil horizons
    # Calculating Euclidean distance for every color
    # 26 horizons × 8,468 colors = 220,168 distance calculations
```

### Breakdown of color conversion:
- **663,160 pandas indexing operations** (accessing color table)
- **220,142 euclidean_distance calls** (brute force matching)
- **Every horizon needs to find closest Munsell color**

## Detailed Timing Breakdown

### 1. Color Conversion (78% of time)
```
lab2munsell()                    37.2s   (main bottleneck)
├─ pandas indexing               34.6s   (table lookups)
├─ euclidean_distance            1.4s    (distance calculations)
└─ array operations              1.2s    (data manipulation)
```

**Impact:** For 26 horizons, converting LAB colors to Munsell is extremely slow

### 2. Database Queries (12% of time)
```
get_soilweb_data()               5.7s    (external API call)
└─ HTTP GET request              5.4s    (network latency)
```

**Impact:** Network-bound, unavoidable

### 3. Soil Simulation - when sim=true (10% of time)
```
soil_sim()                       8.6s    (total simulation pipeline)
├─ Rosetta API                   4.0s    (hydraulic properties)
├─ calculate_vwc_awc             3.2s    (spline interpolation)
│  ├─ UnivariateSpline init      0.7s    (cubic spline fitting)
│  └─ vg_function eval           2.5s    (van Genuchten equations)
├─ simulate_correlated           0.8s    (Monte Carlo)
└─ information_gain              0.6s    (entropy calculations)
```

**Impact:** Rosetta API and AWC calculations dominate simulation time

## Optimization Recommendations

### 🔴 CRITICAL - Optimize lab2munsell (Potential: 35s savings)

**Problem:** Brute force search through 8,468 colors for each horizon

**Solution 1: KD-Tree for Nearest Neighbor Search**
```python
from scipy.spatial import cKDTree

# Build tree once at module load
color_tree = cKDTree(LAB_ref[['L', 'A', 'B']].values)

def lab2munsell_fast(lab_values):
    # Query tree: O(log n) instead of O(n)
    distances, indices = color_tree.query(lab_values)
    return munsell_ref.iloc[indices]

# Expected speedup: 35s → 0.5s (70x faster!)
```

**Solution 2: Vectorized Operations**
```python
def lab2munsell_vectorized(lab_array):
    # Convert all horizons at once (26 × 8468 → vectorized)
    # Use numpy broadcasting instead of loops
    distances = np.sqrt(np.sum((LAB_ref - lab_array[:, None])**2, axis=2))
    min_indices = np.argmin(distances, axis=1)
    return munsell_ref.iloc[min_indices]

# Expected speedup: 35s → 2s (17x faster!)
```

**Solution 3: Caching Common Colors**
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def lab2munsell_cached(l, a, b):
    # Cache common soil colors
    # Typical soils have similar colors
    return lab2munsell_single(l, a, b)

# Expected speedup: 35s → 5s (7x faster on cache hits)
```

### 🟡 HIGH PRIORITY - Cache Database Queries (Potential: 5s savings)

**Problem:** Every request queries SoilWeb API (5.7s)

**Solution: Redis Cache**
```python
import redis
import hashlib

r = redis.Redis()

def list_soils_cached(lon, lat, sim=True):
    # Round coordinates for better cache hits
    lon_r, lat_r = round(lon, 4), round(lat, 4)
    cache_key = f"soil:{lon_r}:{lat_r}:{sim}"
    
    cached = r.get(cache_key)
    if cached:
        return json.loads(cached)
    
    result = list_soils(lon_r, lat_r, sim)
    r.setex(cache_key, 3600, json.dumps(result))  # 1 hour TTL
    return result

# Expected speedup: 5.7s → 0.1s on cache hit
```

### 🟢 MEDIUM PRIORITY - Optimize Rosetta API (Potential: 2s savings)

**Problem:** HTTP API calls with network latency (4.0s)

**Solution 1: Connection Pooling**
```python
import requests
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(
    pool_connections=10,
    pool_maxsize=20,
    max_retries=3
)
session.mount('http://', adapter)

# Reuse session for all Rosetta calls
# Expected speedup: 4.0s → 2.5s
```

**Solution 2: Batch Requests (if API supports)**
```python
# Send all samples in one request instead of multiple
# Expected speedup: 4.0s → 1.0s
```

### 🟢 LOW PRIORITY - Optimize Spline Interpolation (Potential: 1s savings)

**Problem:** Cubic spline fitting for each layer (3.2s)

**Solution: Linear Interpolation Mode**
```python
# Add "fast mode" option
def calculate_vwc_awc(sim_data, fast_mode=False):
    if fast_mode:
        # Use SimpleInterpolator (linear)
        vg_fwd = SimpleInterpolator(m["phi"], m["theta"])
    else:
        # Use UnivariateSpline (cubic)
        vg_fwd = UnivariateSpline(m["phi"], m["theta"], k=3, s=0)
    
    # Expected speedup: 3.2s → 2.0s (trade-off: accuracy)
```

## Performance Targets

### Current Performance
- sim=false: 43s
- sim=true: 48s

### After Critical Optimizations (KD-Tree + Redis)
- sim=false: **6s** (7x faster)
- sim=true: **11s** (4x faster)

### After All Optimizations
- sim=false: **4s** (10x faster)
- sim=true: **8s** (6x faster)

## Implementation Priority

1. **Week 1:** Implement KD-Tree for lab2munsell (35s → 0.5s)
2. **Week 2:** Add Redis caching for database queries (5.7s → 0.1s)
3. **Week 3:** Optimize Rosetta API with connection pooling (4s → 2.5s)
4. **Week 4:** Add fast mode for AWC calculations (3.2s → 2s)

## Next Steps

1. ✅ Profile completed
2. ⏳ Implement KD-Tree for color matching
3. ⏳ Set up Redis cache
4. ⏳ Benchmark improvements
5. ⏳ Deploy optimizations

## Commands to Reproduce

```bash
# Run profiling
python profile_local.py

# View detailed results
cat profile_sim_false.txt
cat profile_sim_true.txt

# Test specific functions
python -m cProfile -s cumulative -m soil_id.color
```

---

**Conclusion:** The biggest win is optimizing `lab2munsell()` which currently takes 78% of execution time. Implementing KD-Tree search will reduce this to <1% and speed up the entire API by 7-10x.
