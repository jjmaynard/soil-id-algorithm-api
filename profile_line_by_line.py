#!/usr/bin/env python3
"""
Line-by-line profiling setup
Install: pip install line-profiler
Run: kernprof -l -v profile_line_by_line.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from soil_id.us_soil import list_soils
from soil_id.soil_sim import soil_sim
from soil_id.utils import calculate_vwc_awc, process_data_with_rosetta

def profile_main():
    """Main profiling function - add @profile to functions you want to profile"""
    print("Running profiling...")
    
    # This will profile list_soils
    result = list_soils(-122.084, 37.422, sim=True)
    
    print(f"Result type: {type(result)}")
    return result

if __name__ == "__main__":
    profile_main()

"""
USAGE:

1. Install line_profiler:
   pip install line-profiler

2. Add @profile decorator to functions you want to profile:
   
   In soil_id/soil_sim.py:
   @profile
   def soil_sim(muhorzdata_pd):
       ...
   
   In soil_id/utils.py:
   @profile
   def calculate_vwc_awc(sim_data, phi_min=1e-6, phi_max=1e8, pts=100):
       ...
   
   @profile
   def process_data_with_rosetta(df, vars, v=3):
       ...

3. Run with line profiler:
   kernprof -l -v profile_line_by_line.py
   
4. Or run with function-level profiling:
   python -m cProfile -o output.prof profile_line_by_line.py
   python -m pstats output.prof
   
5. View results:
   - Line-by-line timing for each decorated function
   - Identify which lines are slowest
   - Focus optimization efforts there

EXAMPLE OUTPUT:
Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   100                                           @profile
   101                                           def calculate_vwc_awc(sim_data):
   102         1        150.0    150.0      0.5      phi = np.logspace(...)
   103        10      12000.0   1200.0     40.0      for _, row in sim_data.iterrows():
   104        10       8000.0    800.0     26.7          vg_fwd = UnivariateSpline(...)
   105        10       5000.0    500.0     16.7          data = {...}
"""
