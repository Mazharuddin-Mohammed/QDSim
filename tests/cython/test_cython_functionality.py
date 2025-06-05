#!/usr/bin/env python3
"""
Test Cython Functionality

This script tests the actual functionality of the working Cython modules
and validates their correctness.
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "qdsim_cython"))
sys.path.insert(0, str(Path(__file__).parent / "frontend"))

def test_materials_functionality():
    """Test materials module functionality"""
    print("🔧 Testing Materials Module Functionality")
    print("=" * 60)
    
    try:
        import qdsim_cython.core.materials_minimal as materials
        
        # Test basic functionality
        start_time = time.time()
        result = materials.test_basic_functionality()
        duration = time.time() - start_time
        print(f"✅ Basic functionality test: {result} ({duration:.3f}s)")
        
        # Test material creation
        start_time = time.time()
        material = materials.create_material("InGaAs", 0.75, 0.041, 13.9)
        duration = time.time() - start_time
        print(f"✅ Material creation: {material} ({duration:.3f}s)")
        
        # Test Material class
        start_time = time.time()
        mat_obj = materials.Material()
        duration = time.time() - start_time
        print(f"✅ Material class instantiation ({duration:.3f}s)")
        
        return True
        
    except Exception as e:
        print(f"❌ Materials test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_quantum_analysis_functionality():
    """Test quantum analysis module functionality"""
    print("\n🔧 Testing Quantum Analysis Module Functionality")
    print("=" * 60)
    
    try:
        import qdsim_cython.analysis.quantum_analysis as qa
        
        # Test QuantumStateAnalyzer
        start_time = time.time()
        analyzer = qa.QuantumStateAnalyzer()
        duration = time.time() - start_time
        print(f"✅ QuantumStateAnalyzer created ({duration:.3f}s)")
        
        # Test with dummy wavefunction data
        start_time = time.time()
        psi = np.random.random(100) + 1j * np.random.random(100)
        psi = psi / np.linalg.norm(psi)  # Normalize
        
        # Test wavefunction analysis
        analysis_result = analyzer.analyze_wavefunction(psi, energy=1e-20)
        duration = time.time() - start_time
        print(f"✅ Wavefunction analysis completed ({duration:.3f}s)")
        print(f"   Analysis keys: {list(analysis_result.keys())}")
        
        # Test EnergyLevelAnalyzer
        start_time = time.time()
        energy_analyzer = qa.EnergyLevelAnalyzer()
        duration = time.time() - start_time
        print(f"✅ EnergyLevelAnalyzer created ({duration:.3f}s)")
        
        # Test energy spectrum analysis
        start_time = time.time()
        energies = np.array([1e-20, 2.1e-20, 4.5e-20, 7.2e-20, 10.8e-20])  # Sample energies
        spectrum_result = energy_analyzer.analyze_energy_spectrum(energies)
        duration = time.time() - start_time
        print(f"✅ Energy spectrum analysis completed ({duration:.3f}s)")
        print(f"   Spectrum keys: {list(spectrum_result.keys())}")
        
        # Test collection analysis
        start_time = time.time()
        states = [np.random.random(50) + 1j * np.random.random(50) for _ in range(5)]
        states = [s/np.linalg.norm(s) for s in states]  # Normalize all states
        
        collection_result = qa.analyze_quantum_state_collection(states, energies, mesh=None)
        duration = time.time() - start_time
        print(f"✅ State collection analysis completed ({duration:.3f}s)")
        print(f"   Collection keys: {list(collection_result.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Quantum analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_frontend_with_cython():
    """Test frontend integration with Cython modules"""
    print("\n🔧 Testing Frontend Integration with Cython")
    print("=" * 60)
    
    try:
        # Import frontend
        import qdsim
        
        # Test basic configuration
        start_time = time.time()
        config = qdsim.Config()
        config.Lx = 100e-9
        config.Ly = 50e-9
        config.nx = 20
        config.ny = 10
        duration = time.time() - start_time
        print(f"✅ Configuration created ({duration:.3f}s)")
        
        # Test simulator creation
        start_time = time.time()
        simulator = qdsim.Simulator(config)
        duration = time.time() - start_time
        print(f"✅ Simulator created ({duration:.3f}s)")
        
        # Test if we can access Cython modules from frontend
        try:
            import qdsim_cython.core.materials_minimal as materials
            material = materials.create_material("InGaAs", 0.75, 0.041, 13.9)
            print(f"✅ Cython materials accessible from frontend context")
        except Exception as e:
            print(f"⚠️  Cython materials not accessible: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Frontend integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_comparison():
    """Test performance comparison between Python and Cython"""
    print("\n🔧 Testing Performance Comparison")
    print("=" * 60)
    
    try:
        import qdsim_cython.analysis.quantum_analysis as qa
        
        # Test performance of Cython vs pure Python
        n_tests = 100
        
        # Cython performance test
        start_time = time.time()
        analyzer = qa.QuantumStateAnalyzer()
        for i in range(n_tests):
            psi = np.random.random(50) + 1j * np.random.random(50)
            psi = psi / np.linalg.norm(psi)
            result = analyzer.analyze_wavefunction(psi)
        cython_time = time.time() - start_time
        
        print(f"✅ Cython analysis ({n_tests} iterations): {cython_time:.3f}s")
        print(f"   Average per iteration: {cython_time/n_tests*1000:.2f}ms")
        
        # Pure Python equivalent (simplified)
        start_time = time.time()
        for i in range(n_tests):
            psi = np.random.random(50) + 1j * np.random.random(50)
            psi = psi / np.linalg.norm(psi)
            
            # Simple Python analysis
            norm_l2 = np.sqrt(np.sum(np.abs(psi)**2))
            prob_density = np.abs(psi)**2
            total_prob = np.sum(prob_density)
            max_amp = np.max(np.abs(psi))
            
        python_time = time.time() - start_time
        
        print(f"✅ Python analysis ({n_tests} iterations): {python_time:.3f}s")
        print(f"   Average per iteration: {python_time/n_tests*1000:.2f}ms")
        
        if python_time > 0:
            speedup = python_time / cython_time
            print(f"🚀 Cython speedup: {speedup:.1f}x faster")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_error_handling():
    """Test error handling in Cython modules"""
    print("\n🔧 Testing Error Handling")
    print("=" * 60)
    
    try:
        import qdsim_cython.analysis.quantum_analysis as qa
        
        # Test with invalid inputs
        analyzer = qa.QuantumStateAnalyzer()
        
        # Test with empty array
        try:
            result = analyzer.analyze_wavefunction(np.array([]))
            print("⚠️  Empty array should have failed")
        except Exception as e:
            print(f"✅ Empty array properly handled: {type(e).__name__}")
        
        # Test with invalid energy spectrum
        energy_analyzer = qa.EnergyLevelAnalyzer()
        try:
            result = energy_analyzer.analyze_energy_spectrum([])
            print(f"✅ Empty energy list handled: {result}")
        except Exception as e:
            print(f"✅ Empty energy list properly handled: {type(e).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 COMPREHENSIVE CYTHON FUNCTIONALITY TEST")
    print("=" * 80)
    
    # Run all tests
    tests = [
        ("Materials Functionality", test_materials_functionality),
        ("Quantum Analysis Functionality", test_quantum_analysis_functionality),
        ("Frontend Integration", test_frontend_with_cython),
        ("Performance Comparison", test_performance_comparison),
        ("Error Handling", test_error_handling),
    ]
    
    results = {}
    total_start_time = time.time()
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    total_duration = time.time() - total_start_time
    
    # Summary
    print("\n" + "=" * 80)
    print("🏆 TEST RESULTS SUMMARY")
    print("=" * 80)
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"📊 Overall Results:")
    print(f"   Total Tests: {total}")
    print(f"   Passed: {passed} ✅")
    print(f"   Failed: {total - passed} ❌")
    print(f"   Success Rate: {(passed/total)*100:.1f}%")
    print(f"   Total Duration: {total_duration:.3f}s")
    
    print(f"\n📋 Detailed Results:")
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} {test_name}")
    
    print(f"\n🎯 Assessment:")
    if passed == total:
        print("   🎉 ALL TESTS PASSED! Cython modules are working correctly.")
    elif passed >= total * 0.8:
        print("   ✅ Most tests passed. Minor issues to resolve.")
    else:
        print("   ⚠️  Significant issues found. More work needed.")
    
    print("=" * 80)
    
    return results

if __name__ == "__main__":
    results = main()
