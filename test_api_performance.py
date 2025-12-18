#!/usr/bin/env python3
"""
API Performance Testing Script
Tests response times for all endpoints with various scenarios
"""

import requests
import time
import statistics
from typing import Dict, List
import json

BASE_URL = "https://soil-id-algorithm-api.vercel.app"

def test_endpoint(name: str, method: str, url: str, data: dict = None, iterations: int = 5) -> Dict:
    """Test an endpoint multiple times and return timing statistics"""
    times = []
    success_count = 0
    
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    
    for i in range(iterations):
        try:
            start = time.time()
            
            if method == "GET":
                response = requests.get(url, timeout=60)
            elif method == "POST":
                response = requests.post(url, json=data, timeout=60)
            
            elapsed = time.time() - start
            times.append(elapsed)
            
            if response.status_code == 200:
                success_count += 1
                status = "✓"
            else:
                status = f"✗ ({response.status_code})"
            
            print(f"  Run {i+1}: {elapsed:.3f}s {status}")
            
            # Small delay between requests
            if i < iterations - 1:
                time.sleep(0.5)
                
        except requests.exceptions.Timeout:
            print(f"  Run {i+1}: TIMEOUT")
        except Exception as e:
            print(f"  Run {i+1}: ERROR - {str(e)}")
    
    if times:
        results = {
            "endpoint": name,
            "iterations": iterations,
            "successful": success_count,
            "min": min(times),
            "max": max(times),
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "stdev": statistics.stdev(times) if len(times) > 1 else 0
        }
        
        print(f"\n  Results:")
        print(f"    Success Rate: {success_count}/{iterations} ({success_count/iterations*100:.1f}%)")
        print(f"    Min:    {results['min']:.3f}s")
        print(f"    Max:    {results['max']:.3f}s")
        print(f"    Mean:   {results['mean']:.3f}s")
        print(f"    Median: {results['median']:.3f}s")
        if results['stdev'] > 0:
            print(f"    StdDev: {results['stdev']:.3f}s")
        
        return results
    else:
        return None


def main():
    print("API Performance Testing")
    print(f"Base URL: {BASE_URL}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    
    # Test 1: Health check endpoint
    result = test_endpoint(
        "Health Check (GET /)",
        "GET",
        f"{BASE_URL}/",
        iterations=3
    )
    if result:
        results.append(result)
    
    # Test 2: Debug endpoint
    result = test_endpoint(
        "Debug (GET /api/debug)",
        "GET",
        f"{BASE_URL}/api/debug",
        iterations=3
    )
    if result:
        results.append(result)
    
    # Test 3: List soils with sim=false
    result = test_endpoint(
        "List Soils (sim=false)",
        "POST",
        f"{BASE_URL}/api/list-soils",
        data={"lon": -122.084, "lat": 37.422, "sim": False},
        iterations=5
    )
    if result:
        results.append(result)
    
    # Test 4: List soils with sim=true
    result = test_endpoint(
        "List Soils (sim=true)",
        "POST",
        f"{BASE_URL}/api/list-soils",
        data={"lon": -122.084, "lat": 37.422, "sim": True},
        iterations=5
    )
    if result:
        results.append(result)
    
    # Test 5: Different location with sim=true
    result = test_endpoint(
        "List Soils Texas (sim=true)",
        "POST",
        f"{BASE_URL}/api/list-soils",
        data={"lon": -101.9733687, "lat": 33.81246789, "sim": True},
        iterations=3
    )
    if result:
        results.append(result)
    
    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Endpoint':<40} {'Mean':<10} {'Median':<10} {'Min':<10} {'Max':<10}")
    print(f"{'-'*80}")
    
    for r in results:
        print(f"{r['endpoint']:<40} {r['mean']:.3f}s    {r['median']:.3f}s    {r['min']:.3f}s    {r['max']:.3f}s")
    
    # Save results to JSON
    output_file = "api_performance_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "base_url": BASE_URL,
            "results": results
        }, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")


if __name__ == "__main__":
    main()
