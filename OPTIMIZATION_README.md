# Performance Optimization Suite

Complete toolkit for profiling and optimizing the Soil ID Algorithm API.

## 📊 Current Performance

| Endpoint | Current | Optimized Target | Speedup |
|----------|---------|------------------|---------|
| list-soils (sim=false) | 43s | 6s | 7x |
| list-soils (sim=true) | 48s | 11s | 4x |

## 🔍 Profiling Tools

### 1. Quick Performance Test
```bash
python test_api_performance.py
```
- Tests all endpoints
- Measures response times
- Creates JSON report

### 2. Deep Profiling (Local)
```bash
python profile_local.py
```
- Identifies bottlenecks
- Function-level timing
- Creates detailed reports

### 3. Line-by-Line Profiling
```bash
pip install line-profiler
kernprof -l -v profile_line_by_line.py
```
- Line-level timing
- Find slow loops
- Optimize hot paths

## 📈 Key Findings

### Critical Bottleneck: Color Matching (78% of time!)

**Problem:** `lab2munsell()` takes 37 seconds
- Brute force search through 8,468 colors
- For each of 26 soil horizons
- 220,000+ distance calculations

**Solution:** KD-Tree spatial indexing
```bash
python optimize_color_matching.py
```
Expected: **37s → 0.5s (70x faster!)**

### Other Bottlenecks:
1. **Database Queries** (12%): Add Redis caching
2. **Rosetta API** (8%): Connection pooling
3. **AWC Calculation** (7%): Optional fast mode

## 🚀 Quick Wins

### Implement KD-Tree Color Matching
```bash
# Run benchmark
python optimize_color_matching.py

# Expected output:
# SPEEDUP: 70x faster!
# Projected savings for full API: 36s
```

### Add Redis Cache (optional)
```python
import redis
r = redis.Redis()

def list_soils_cached(lon, lat, sim):
    key = f"soil:{round(lon,4)}:{round(lat,4)}:{sim}"
    cached = r.get(key)
    if cached:
        return json.loads(cached)
    result = list_soils(lon, lat, sim)
    r.setex(key, 3600, json.dumps(result))
    return result
```

## 📝 Documentation

- **[PROFILING.md](PROFILING.md)**: Complete profiling guide
- **[PROFILING_RESULTS.md](PROFILING_RESULTS.md)**: Detailed analysis
- **[optimize_color_matching.py](optimize_color_matching.py)**: KD-Tree implementation

## 🎯 Implementation Roadmap

### Phase 1: Color Optimization (Week 1)
- [x] Profile and identify bottleneck
- [ ] Implement KD-Tree matcher
- [ ] Benchmark improvements
- [ ] Deploy to production

**Expected:** 43s → 8s

### Phase 2: Caching (Week 2)  
- [ ] Set up Redis
- [ ] Implement cache layer
- [ ] Add cache warming

**Expected:** 8s → 3s (cache hits)

### Phase 3: Fine-tuning (Week 3)
- [ ] Optimize Rosetta API calls
- [ ] Add fast mode option
- [ ] Connection pooling

**Expected:** 3s → 2s

## 📊 Monitoring

### Enable Timing in Production
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Output:
```
⏱️  Step 2 (Simulation): 2.456s
⏱️  Rosetta API: 1.234s
⏱️  AWC Calculation: 0.789s
⏱️  Information Gain: 0.456s
```

### Add Response Time Headers
```python
@app.middleware("http")
async def add_timing_header(request, call_next):
    start = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{time.time()-start:.3f}"
    return response
```

## 🔧 Tools Included

| File | Purpose |
|------|---------|
| `test_api_performance.py` | Quick endpoint testing |
| `profile_local.py` | Deep profiling with cProfile |
| `profile_line_by_line.py` | Line-level profiling setup |
| `optimize_color_matching.py` | KD-Tree implementation |
| `soil_id/timing_utils.py` | Timing decorators/context managers |
| `PROFILING.md` | Complete guide |
| `PROFILING_RESULTS.md` | Analysis & recommendations |

## 💡 Quick Commands

```bash
# Profile API
python profile_local.py

# Test performance
python test_api_performance.py

# Benchmark color matching
python optimize_color_matching.py

# Line profiling (after adding @profile)
kernprof -l -v profile_line_by_line.py

# View detailed stats
cat profile_sim_true.txt | head -100
```

## 📞 Support

Questions? Check:
1. [PROFILING.md](PROFILING.md) - Detailed guide
2. [PROFILING_RESULTS.md](PROFILING_RESULTS.md) - Analysis
3. Profile output files - Raw data

---

**Next Step:** Run `python optimize_color_matching.py` to see 70x speedup demo!
