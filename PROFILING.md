# Performance Profiling Guide

This guide helps you identify and optimize bottlenecks in the Soil ID API.

## Quick Start

### 1. Run Basic Profiling

```bash
# Profile locally with cProfile
python profile_local.py

# This will create:
# - profile_sim_false.txt: Baseline performance without simulation
# - profile_sim_true.txt: Full pipeline performance
```

### 2. Enable Debug Logging for Timing

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now all timing prints will show:
# ⏱️  Step 2 (Simulation): 2.456s
# ⏱️  Rosetta API: 1.234s
# ⏱️  AWC Calculation: 0.789s
# ⏱️  Information Gain: 0.456s
```

### 3. Add Custom Timing

```python
from soil_id.timing_utils import timeit, timer, print_timing_summary

# Decorate functions
@timeit("My Function")
def my_function():
    pass

# Or use context manager
with timer("Expensive Operation"):
    result = expensive_operation()

# Print summary at end
print_timing_summary()
```

## Profiling Tools

### Tool 1: cProfile (Function-Level)

**Best for:** Identifying which functions take the most time

```bash
python profile_local.py
```

**Output Analysis:**
- `ncalls`: Number of times function was called
- `tottime`: Total time spent in function (excluding subcalls)
- `cumtime`: Cumulative time (including subcalls)
- `percall`: Average time per call

**Look for:**
- Functions with high `cumtime` (total impact)
- Functions with high `tottime` (internal bottlenecks)
- Functions called many times (optimization opportunity)

### Tool 2: line_profiler (Line-Level)

**Best for:** Finding slow lines within a function

```bash
# Install
pip install line-profiler

# Add @profile to functions in soil_sim.py, utils.py
# Then run:
kernprof -l -v profile_line_by_line.py
```

**Look for:**
- Lines with high `% Time`
- Loops with high `Hits` count
- I/O operations (file, network, database)

### Tool 3: Built-in Timing (Already Added)

**Enable debug logging:**

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Timing points added:**
- Step 2 (Monte Carlo Simulation)
- Rosetta API calls
- AWC calculation (spline interpolation)
- Information Gain calculation

### Tool 4: Memory Profiling

```bash
# Install
pip install memory-profiler

# Add @profile to functions
# Run:
python -m memory_profiler profile_line_by_line.py
```

## Known Bottlenecks & Solutions

### 1. Database Queries (14s baseline)

**Bottleneck:**
- `get_soilweb_data()`: External API call
- `sda_return()`: SSURGO/STATSGO queries

**Solutions:**
- ✅ Cache results by (lon, lat) with Redis/Memcached
- ✅ Pre-fetch common locations
- ❌ Can't optimize external APIs

**Caching Example:**
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def list_soils_cached(lon, lat, sim=True):
    # Round coordinates to reduce cache misses
    lon_round = round(lon, 4)
    lat_round = round(lat, 4)
    return list_soils(lon_round, lat_round, sim)
```

### 2. Rosetta API Calls (~1-2s)

**Bottleneck:**
- HTTP REST API calls to Rosetta
- One call per simulation batch

**Solutions:**
- ✅ Batch multiple requests
- ✅ Add retry logic with exponential backoff
- ✅ Use connection pooling
- ❌ Already using fastest method (REST API)

### 3. Monte Carlo Simulation (~2-3s)

**Bottleneck:**
- `simulate_correlated_triangular()`: 1000+ samples per component
- ILR transformations on large arrays
- Cholesky decomposition: O(n³)

**Solutions:**
- ✅ Vectorize numpy operations (already done)
- ✅ Reduce simulation count for low-probability components
- ⚠️ Parallelize across components (requires multiprocessing)
- ❌ Can't reduce mathematical complexity

**Parallelization Example:**
```python
from multiprocessing import Pool

def simulate_component(row):
    # Run simulation for one component
    return simulated_data

# Parallelize across components
with Pool(4) as pool:
    sim_results = pool.map(simulate_component, components)
```

### 4. Spline Interpolation (~0.5-1s)

**Bottleneck:**
- `calculate_vwc_awc()`: Cubic spline per layer
- Called for each simulated sample

**Solutions:**
- ✅ Already using cubic spline (smooth but slower)
- ⚠️ Use linear interpolation for speed (less accurate)
- ⚠️ Pre-compute splines and reuse

**Fast Mode:**
```python
# In calculate_vwc_awc:
# Replace: vg_fwd = UnivariateSpline(m["phi"], m["theta"], k=3, s=0)
# With:    vg_fwd = SimpleInterpolator(m["phi"], m["theta"])
# Saves ~30% time but less accurate for soil water curves
```

### 5. Information Gain (~0.3-0.5s)

**Bottleneck:**
- Entropy calculations across all simulated samples
- Grouping and aggregation

**Solutions:**
- ✅ Already using vectorized entropy
- ⚠️ Sample fewer data points for entropy
- ⚠️ Use approximate entropy methods

## Optimization Priority

Based on performance testing:

1. **🔴 HIGH PRIORITY** (14s / 70% of time)
   - Database queries: Add caching layer
   - Solution: Redis cache with 1-hour TTL

2. **🟡 MEDIUM PRIORITY** (2-3s / 15% of time)
   - Monte Carlo simulations: Reduce sample count for low-prob components
   - Solution: Adaptive sampling based on cond_prob

3. **🟢 LOW PRIORITY** (2s / 10% of time)
   - Rosetta API: Already optimized
   - AWC/Info Gain: Already vectorized

## Target Performance

Current:
- sim=false: ~14s (database)
- sim=true: ~19s (database + 5s simulation)

With caching:
- sim=false: ~2s (cache hit)
- sim=true: ~7s (cache hit + 5s simulation)

With adaptive sampling:
- sim=true: ~5s (cache hit + 3s simulation)

## Monitoring in Production

Add to FastAPI app:

```python
import time
from fastapi import Request

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    print(f"{request.url.path}: {process_time:.3f}s")
    return response
```

## Next Steps

1. Run `python profile_local.py` to get baseline
2. Review output files for hotspots
3. Enable debug logging: `logging.basicConfig(level=logging.DEBUG)`
4. Implement caching for database queries
5. Consider adaptive sampling for simulations
6. Re-profile and measure improvements
