#!/usr/bin/env python3
"""
Local Performance Profiling Script
Profiles the soil_sim function to identify bottlenecks
"""

import cProfile
import pstats
import io
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from soil_id.us_soil import list_soils

def profile_list_soils_with_sim():
    """Profile list_soils with sim=True"""
    print("Profiling list_soils with sim=True...")
    result = list_soils(-122.084, 37.422, sim=True)
    return result

def profile_list_soils_without_sim():
    """Profile list_soils with sim=False"""
    print("Profiling list_soils with sim=False...")
    result = list_soils(-122.084, 37.422, sim=False)
    return result

def run_with_cprofile(func, output_file=None):
    """Run function with cProfile and print results"""
    profiler = cProfile.Profile()
    
    start = time.time()
    profiler.enable()
    result = func()
    profiler.disable()
    elapsed = time.time() - start
    
    print(f"\n{'='*80}")
    print(f"Execution completed in {elapsed:.2f} seconds")
    print(f"{'='*80}\n")
    
    # Print stats to console
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)  # Top 30 functions
    print(s.getvalue())
    
    # Also save detailed stats to file
    if output_file:
        with open(output_file, 'w') as f:
            ps = pstats.Stats(profiler, stream=f).sort_stats('cumulative')
            f.write(f"{'='*80}\n")
            f.write(f"Total execution time: {elapsed:.2f} seconds\n")
            f.write(f"{'='*80}\n\n")
            
            f.write("="*80 + "\n")
            f.write("TOP FUNCTIONS BY CUMULATIVE TIME\n")
            f.write("="*80 + "\n")
            ps.print_stats(50)
            
            f.write("\n" + "="*80 + "\n")
            f.write("TOP FUNCTIONS BY INTERNAL TIME (excluding subcalls)\n")
            f.write("="*80 + "\n")
            ps.sort_stats('time')
            ps.print_stats(50)
            
            f.write("\n" + "="*80 + "\n")
            f.write("CALLERS (who called what)\n")
            f.write("="*80 + "\n")
            ps.sort_stats('cumulative')
            ps.print_callers(30)
        
        print(f"\n✓ Detailed profile saved to: {output_file}")
    
    return result

def analyze_bottlenecks():
    """Analyze and print likely bottlenecks"""
    print("\n" + "="*80)
    print("BOTTLENECK ANALYSIS")
    print("="*80)
    print("""
Based on the code structure, likely bottlenecks are:

1. DATABASE QUERIES (list_soils early stages)
   - get_soilweb_data(): External API call to SoilWeb
   - sda_return(): SSURGO/STATSGO database queries
   - These are network-bound and can't be easily optimized
   
2. SOIL SIMULATION (when sim=True)
   - simulate_correlated_triangular(): Monte Carlo simulations
   - ilr/ilr_inv transformations: Matrix operations
   - Cholesky decomposition: O(n³) complexity
   
3. ROSETTA API (hydraulic properties)
   - process_data_with_rosetta(): HTTP REST API calls
   - Can be optimized with batching or caching
   
4. INTERPOLATION (calculate_vwc_awc)
   - UnivariateSpline: Cubic spline fitting per layer
   - Called for each simulated layer
   
5. INFORMATION GAIN (variable importance)
   - entropy calculations across all simulated samples
   - Grouping and aggregation operations
   
Run the profiler to see actual time distribution!
""")

def main():
    print("="*80)
    print("SOIL ID API - PERFORMANCE PROFILING")
    print("="*80)
    
    analyze_bottlenecks()
    
    # Profile with sim=False first (baseline)
    print("\n" + "="*80)
    print("PROFILING: sim=False (database queries only)")
    print("="*80)
    run_with_cprofile(
        profile_list_soils_without_sim,
        output_file="profile_sim_false.txt"
    )
    
    # Profile with sim=True (full pipeline)
    print("\n" + "="*80)
    print("PROFILING: sim=True (full simulation pipeline)")
    print("="*80)
    run_with_cprofile(
        profile_list_soils_with_sim,
        output_file="profile_sim_true.txt"
    )
    
    print("\n" + "="*80)
    print("PROFILING COMPLETE")
    print("="*80)
    print("""
Next steps:
1. Review profile_sim_false.txt for database bottlenecks
2. Review profile_sim_true.txt for simulation bottlenecks
3. Focus optimization on functions with high cumulative time
4. Consider:
   - Caching database queries
   - Batching Rosetta API calls
   - Vectorizing numpy operations
   - Parallel processing for independent simulations
""")

if __name__ == "__main__":
    main()
