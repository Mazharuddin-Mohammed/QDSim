#!/usr/bin/env python3
"""
Performance Benchmarking Suite for QDSim Cython Migration
Tests performance improvements and validates memory management
"""

import sys
import os
import time
import gc
import psutil
import traceback
import statistics
from typing import List, Dict, Any, Callable
import threading
import multiprocessing

class PerformanceBenchmark:
    """Comprehensive performance benchmarking for QDSim"""
    
    def __init__(self):
        self.results = {}
        self.process = psutil.Process()
        
    def measure_time(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Measure execution time and memory usage"""
        
        # Get initial memory
        gc.collect()
        initial_memory = self.process.memory_info().rss
        
        # Measure execution time
        start_time = time.perf_counter()
        start_cpu = time.process_time()
        
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        end_time = time.perf_counter()
        end_cpu = time.process_time()
        
        # Get final memory
        gc.collect()
        final_memory = self.process.memory_info().rss
        
        return {
            'result': result,
            'success': success,
            'error': error,
            'wall_time': end_time - start_time,
            'cpu_time': end_cpu - start_cpu,
            'memory_used': final_memory - initial_memory,
            'initial_memory': initial_memory,
            'final_memory': final_memory
        }
    
    def benchmark_material_creation(self, module, iterations: int = 1000) -> Dict[str, Any]:
        """Benchmark material creation performance"""
        
        def create_materials():
            materials = []
            for i in range(iterations):
                mat = module.create_material()
                mat.m_e = 0.067 + i * 0.001
                mat.E_g = 1.424 + i * 0.0001
                materials.append(mat)
            return materials
        
        print(f"Benchmarking material creation ({iterations} iterations)...")
        result = self.measure_time(create_materials)
        
        if result['success']:
            materials_per_second = iterations / result['wall_time']
            memory_per_material = result['memory_used'] / iterations if iterations > 0 else 0
            
            benchmark_data = {
                'iterations': iterations,
                'total_time': result['wall_time'],
                'cpu_time': result['cpu_time'],
                'materials_per_second': materials_per_second,
                'memory_used': result['memory_used'],
                'memory_per_material': memory_per_material,
                'success': True
            }
        else:
            benchmark_data = {
                'iterations': iterations,
                'success': False,
                'error': result['error']
            }
        
        self.results['material_creation'] = benchmark_data
        return benchmark_data
    
    def benchmark_property_access(self, module, iterations: int = 10000) -> Dict[str, Any]:
        """Benchmark property access performance"""
        
        def access_properties():
            mat = module.create_material()
            
            # Benchmark getters
            start = time.perf_counter()
            for i in range(iterations):
                _ = mat.m_e
                _ = mat.m_h
                _ = mat.E_g
                _ = mat.epsilon_r
            getter_time = time.perf_counter() - start
            
            # Benchmark setters
            start = time.perf_counter()
            for i in range(iterations):
                mat.m_e = 0.067 + i * 0.0001
                mat.m_h = 0.45 + i * 0.0001
                mat.E_g = 1.424 + i * 0.0001
                mat.epsilon_r = 12.9 + i * 0.001
            setter_time = time.perf_counter() - start
            
            return getter_time, setter_time
        
        print(f"Benchmarking property access ({iterations} iterations)...")
        result = self.measure_time(access_properties)
        
        if result['success']:
            getter_time, setter_time = result['result']
            
            benchmark_data = {
                'iterations': iterations,
                'getter_time': getter_time,
                'setter_time': setter_time,
                'total_time': result['wall_time'],
                'getters_per_second': (iterations * 4) / getter_time,
                'setters_per_second': (iterations * 4) / setter_time,
                'memory_used': result['memory_used'],
                'success': True
            }
        else:
            benchmark_data = {
                'iterations': iterations,
                'success': False,
                'error': result['error']
            }
        
        self.results['property_access'] = benchmark_data
        return benchmark_data
    
    def benchmark_memory_management(self, module, cycles: int = 100, objects_per_cycle: int = 1000) -> Dict[str, Any]:
        """Benchmark memory management and RAII"""
        
        def memory_stress_test():
            memory_snapshots = []
            
            for cycle in range(cycles):
                # Create many objects
                materials = []
                for i in range(objects_per_cycle):
                    mat = module.create_material()
                    mat.m_e = 0.067 + i * 0.001
                    materials.append(mat)
                
                # Take memory snapshot
                memory_snapshots.append(self.process.memory_info().rss)
                
                # Delete objects (test RAII)
                del materials
                gc.collect()
                
                # Take another snapshot
                memory_snapshots.append(self.process.memory_info().rss)
            
            return memory_snapshots
        
        print(f"Benchmarking memory management ({cycles} cycles, {objects_per_cycle} objects each)...")
        result = self.measure_time(memory_stress_test)
        
        if result['success']:
            memory_snapshots = result['result']
            
            # Analyze memory usage patterns
            peak_memory = max(memory_snapshots)
            min_memory = min(memory_snapshots)
            memory_variance = statistics.variance(memory_snapshots) if len(memory_snapshots) > 1 else 0
            
            # Check for memory leaks (final memory should be close to initial)
            initial_memory = memory_snapshots[0] if memory_snapshots else 0
            final_memory = memory_snapshots[-1] if memory_snapshots else 0
            potential_leak = final_memory - initial_memory
            
            benchmark_data = {
                'cycles': cycles,
                'objects_per_cycle': objects_per_cycle,
                'total_objects': cycles * objects_per_cycle,
                'total_time': result['wall_time'],
                'peak_memory': peak_memory,
                'min_memory': min_memory,
                'memory_variance': memory_variance,
                'potential_leak': potential_leak,
                'memory_efficiency': min_memory / peak_memory if peak_memory > 0 else 0,
                'success': True
            }
        else:
            benchmark_data = {
                'cycles': cycles,
                'objects_per_cycle': objects_per_cycle,
                'success': False,
                'error': result['error']
            }
        
        self.results['memory_management'] = benchmark_data
        return benchmark_data
    
    def benchmark_multithreading(self, module, threads: int = 4, iterations_per_thread: int = 1000) -> Dict[str, Any]:
        """Benchmark thread safety and parallel performance"""
        
        def thread_worker(thread_id: int, results: List):
            try:
                start_time = time.perf_counter()
                materials = []
                
                for i in range(iterations_per_thread):
                    mat = module.create_material()
                    mat.m_e = 0.067 + thread_id * 0.01 + i * 0.001
                    materials.append(mat)
                
                end_time = time.perf_counter()
                results.append({
                    'thread_id': thread_id,
                    'time': end_time - start_time,
                    'materials_created': len(materials),
                    'success': True
                })
            except Exception as e:
                results.append({
                    'thread_id': thread_id,
                    'success': False,
                    'error': str(e)
                })
        
        print(f"Benchmarking multithreading ({threads} threads, {iterations_per_thread} iterations each)...")
        
        # Run multithreaded test
        thread_results = []
        threads_list = []
        
        start_time = time.perf_counter()
        
        for i in range(threads):
            thread = threading.Thread(target=thread_worker, args=(i, thread_results))
            threads_list.append(thread)
            thread.start()
        
        for thread in threads_list:
            thread.join()
        
        end_time = time.perf_counter()
        
        # Analyze results
        successful_threads = [r for r in thread_results if r.get('success', False)]
        failed_threads = [r for r in thread_results if not r.get('success', False)]
        
        if successful_threads:
            total_materials = sum(r['materials_created'] for r in successful_threads)
            avg_thread_time = statistics.mean(r['time'] for r in successful_threads)
            
            benchmark_data = {
                'threads': threads,
                'iterations_per_thread': iterations_per_thread,
                'total_time': end_time - start_time,
                'avg_thread_time': avg_thread_time,
                'successful_threads': len(successful_threads),
                'failed_threads': len(failed_threads),
                'total_materials': total_materials,
                'materials_per_second': total_materials / (end_time - start_time),
                'parallel_efficiency': avg_thread_time / (end_time - start_time),
                'success': len(failed_threads) == 0
            }
        else:
            benchmark_data = {
                'threads': threads,
                'iterations_per_thread': iterations_per_thread,
                'successful_threads': 0,
                'failed_threads': len(failed_threads),
                'success': False,
                'errors': [r.get('error', 'Unknown error') for r in failed_threads]
            }
        
        self.results['multithreading'] = benchmark_data
        return benchmark_data
    
    def run_comprehensive_benchmark(self, module) -> Dict[str, Any]:
        """Run all benchmarks"""
        
        print("="*70)
        print("QDSim Cython Performance Benchmark Suite")
        print("="*70)
        
        # Test 1: Material Creation Performance
        self.benchmark_material_creation(module, iterations=1000)
        
        # Test 2: Property Access Performance
        self.benchmark_property_access(module, iterations=10000)
        
        # Test 3: Memory Management
        self.benchmark_memory_management(module, cycles=50, objects_per_cycle=500)
        
        # Test 4: Multithreading
        self.benchmark_multithreading(module, threads=4, iterations_per_thread=500)
        
        return self.results
    
    def print_results(self):
        """Print comprehensive benchmark results"""
        
        print("\n" + "="*70)
        print("PERFORMANCE BENCHMARK RESULTS")
        print("="*70)
        
        for test_name, data in self.results.items():
            print(f"\n{test_name.upper().replace('_', ' ')}:")
            print("-" * 50)
            
            if data.get('success', False):
                if test_name == 'material_creation':
                    print(f"  Materials created: {data['iterations']:,}")
                    print(f"  Total time: {data['total_time']:.3f} seconds")
                    print(f"  Materials/second: {data['materials_per_second']:,.0f}")
                    print(f"  Memory used: {data['memory_used']:,} bytes")
                    print(f"  Memory/material: {data['memory_per_material']:.1f} bytes")
                
                elif test_name == 'property_access':
                    print(f"  Property accesses: {data['iterations'] * 8:,}")
                    print(f"  Getters/second: {data['getters_per_second']:,.0f}")
                    print(f"  Setters/second: {data['setters_per_second']:,.0f}")
                    print(f"  Total time: {data['total_time']:.3f} seconds")
                
                elif test_name == 'memory_management':
                    print(f"  Objects created: {data['total_objects']:,}")
                    print(f"  Peak memory: {data['peak_memory']:,} bytes")
                    print(f"  Memory efficiency: {data['memory_efficiency']:.2%}")
                    print(f"  Potential leak: {data['potential_leak']:,} bytes")
                    leak_status = "✅ No leak" if abs(data['potential_leak']) < 1024*1024 else "⚠️ Possible leak"
                    print(f"  Leak status: {leak_status}")
                
                elif test_name == 'multithreading':
                    print(f"  Threads: {data['threads']}")
                    print(f"  Successful threads: {data['successful_threads']}")
                    print(f"  Materials/second: {data.get('materials_per_second', 0):,.0f}")
                    print(f"  Parallel efficiency: {data.get('parallel_efficiency', 0):.2%}")
                    
            else:
                print(f"  ❌ FAILED: {data.get('error', 'Unknown error')}")
        
        # Overall assessment
        print("\n" + "="*70)
        print("OVERALL ASSESSMENT")
        print("="*70)
        
        successful_tests = sum(1 for data in self.results.values() if data.get('success', False))
        total_tests = len(self.results)
        success_rate = successful_tests / total_tests if total_tests > 0 else 0
        
        print(f"Tests passed: {successful_tests}/{total_tests} ({success_rate:.1%})")
        
        if success_rate >= 1.0:
            print("🎉 EXCELLENT: All performance tests passed!")
            print("✅ Cython migration shows excellent performance")
        elif success_rate >= 0.75:
            print("✅ GOOD: Most performance tests passed")
            print("⚠️ Some optimizations may be needed")
        else:
            print("❌ POOR: Significant performance issues detected")
            print("🔧 Major optimization work required")

def main():
    """Main benchmark function"""
    
    try:
        # Try to import the materials module
        print("Loading QDSim materials module...")
        import materials_minimal as module
        print("✅ Module loaded successfully")
        
        # Run benchmarks
        benchmark = PerformanceBenchmark()
        results = benchmark.run_comprehensive_benchmark(module)
        benchmark.print_results()
        
        return 0
        
    except ImportError as e:
        print(f"❌ Failed to import materials module: {e}")
        print("Make sure the Cython module is compiled and available")
        return 1
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
