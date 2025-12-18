"""
Performance timing decorator and context manager
Add these to your code to measure execution time of specific functions
"""

import time
import functools
from contextlib import contextmanager

# Global storage for timing data
_timing_data = {}

def timeit(label=None):
    """
    Decorator to time function execution
    
    Usage:
        @timeit("My Function")
        def my_function():
            ...
    """
    def decorator(func):
        func_label = label or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            
            # Store timing data
            if func_label not in _timing_data:
                _timing_data[func_label] = []
            _timing_data[func_label].append(elapsed)
            
            # Print timing info
            print(f"⏱️  {func_label}: {elapsed:.3f}s")
            
            return result
        return wrapper
    return decorator

@contextmanager
def timer(label):
    """
    Context manager to time code blocks
    
    Usage:
        with timer("Database Query"):
            result = expensive_query()
    """
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        
        # Store timing data
        if label not in _timing_data:
            _timing_data[label] = []
        _timing_data[label].append(elapsed)
        
        print(f"⏱️  {label}: {elapsed:.3f}s")

def print_timing_summary():
    """Print summary of all timed operations"""
    if not _timing_data:
        print("No timing data collected")
        return
    
    print("\n" + "="*80)
    print("TIMING SUMMARY")
    print("="*80)
    print(f"{'Operation':<50} {'Count':<10} {'Total':<12} {'Avg':<12}")
    print("-"*80)
    
    # Sort by total time descending
    sorted_data = sorted(
        _timing_data.items(),
        key=lambda x: sum(x[1]),
        reverse=True
    )
    
    total_time = sum(sum(times) for times in _timing_data.values())
    
    for label, times in sorted_data:
        count = len(times)
        total = sum(times)
        avg = total / count
        pct = (total / total_time * 100) if total_time > 0 else 0
        
        print(f"{label:<50} {count:<10} {total:>10.3f}s  {avg:>10.3f}s  ({pct:>5.1f}%)")
    
    print("-"*80)
    print(f"{'TOTAL':<50} {'':<10} {total_time:>10.3f}s")
    print("="*80)

def reset_timing_data():
    """Clear all timing data"""
    global _timing_data
    _timing_data = {}

# Example usage in code:
"""
from soil_id.timing_utils import timeit, timer, print_timing_summary

# As decorator
@timeit("Database Query")
def get_soil_data(lon, lat):
    ...

# As context manager
def process_soil_data():
    with timer("ILR Transformation"):
        ilr_coords = ilr(compositions)
    
    with timer("Rosetta API Call"):
        rosetta_data = process_data_with_rosetta(...)
    
    with timer("AWC Calculation"):
        awc = calculate_vwc_awc(rosetta_data)

# Print summary at end
print_timing_summary()
"""
