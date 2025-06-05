#!/usr/bin/env python3
"""
Simple Visualization Test

This script tests visualization with minimal dependencies to identify the issue.
"""

import sys
import os

def test_basic_python():
    """Test basic Python functionality"""
    print("🔧 Testing basic Python functionality...")
    
    try:
        import numpy as np
        print("✅ NumPy imported")
        
        # Test basic numpy operations
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        print("✅ NumPy operations working")
        
        return True
    except Exception as e:
        print(f"❌ Basic Python test failed: {e}")
        return False

def test_matplotlib_minimal():
    """Test matplotlib with minimal setup"""
    print("\n🔧 Testing matplotlib minimal setup...")
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        print("✅ Matplotlib imported with Agg backend")
        
        # Test basic plot creation
        import numpy as np
        x = np.linspace(0, 10, 50)
        y = np.sin(x)
        
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_title("Test Plot")
        print("✅ Basic plot created")
        
        # Test saving (without showing)
        fig.savefig('test_plot.png')
        print("✅ Plot saved to file")
        
        plt.close(fig)  # Clean up
        
        return True
    except Exception as e:
        print(f"❌ Matplotlib test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_scipy_minimal():
    """Test scipy with minimal functionality"""
    print("\n🔧 Testing scipy minimal setup...")
    
    try:
        import scipy
        print("✅ Scipy imported")
        
        from scipy.interpolate import griddata
        print("✅ Scipy interpolation imported")
        
        # Test basic interpolation
        import numpy as np
        points = np.random.rand(10, 2)
        values = np.random.rand(10)
        grid_x, grid_y = np.mgrid[0:1:10j, 0:1:10j]
        
        result = griddata(points, values, (grid_x, grid_y), method='linear')
        print("✅ Basic interpolation working")
        
        return True
    except Exception as e:
        print(f"❌ Scipy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_visualization_import():
    """Test importing the visualization module"""
    print("\n🔧 Testing visualization module import...")
    
    try:
        # Add path
        sys.path.insert(0, 'qdsim_cython')
        print("✅ Path added")
        
        # Check if file exists
        viz_file = 'qdsim_cython/visualization/wavefunction_plotter.py'
        if os.path.exists(viz_file):
            print("✅ Visualization file exists")
        else:
            print(f"❌ Visualization file missing: {viz_file}")
            return False
        
        # Try import
        from qdsim_cython.visualization.wavefunction_plotter import WavefunctionPlotter
        print("✅ WavefunctionPlotter imported")
        
        # Try creating instance
        plotter = WavefunctionPlotter()
        print("✅ WavefunctionPlotter instance created")
        
        return True
    except Exception as e:
        print(f"❌ Visualization import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_plot_creation():
    """Test creating a simple plot with the visualization module"""
    print("\n🔧 Testing simple plot creation...")
    
    try:
        # Set matplotlib backend
        import matplotlib
        matplotlib.use('Agg')
        
        # Import visualization
        sys.path.insert(0, 'qdsim_cython')
        from qdsim_cython.visualization.wavefunction_plotter import WavefunctionPlotter
        
        # Create plotter
        plotter = WavefunctionPlotter()
        
        # Create simple test data
        import numpy as np
        
        # Simple 1D-like data for testing
        x = np.linspace(0, 20e-9, 8)
        y = np.linspace(0, 15e-9, 6)
        X, Y = np.meshgrid(x, y)
        
        nodes_x = X.flatten()
        nodes_y = Y.flatten()
        
        # Simple Gaussian wavefunction
        x0, y0 = 10e-9, 7.5e-9
        sigma = 3e-9
        wavefunction = np.exp(-((nodes_x - x0)**2 + (nodes_y - y0)**2) / (2 * sigma**2))
        
        print("✅ Test data created")
        
        # Test energy level plot (simplest)
        eigenvalues = np.array([-0.05, -0.03]) * 1.602176634e-19
        fig1 = plotter.plot_energy_levels(eigenvalues, "Simple Test Energy Levels")
        print("✅ Energy level plot created")
        
        # Test 2D wavefunction plot
        fig2 = plotter.plot_wavefunction_2d(nodes_x, nodes_y, wavefunction, "Simple Test Wavefunction")
        print("✅ 2D wavefunction plot created")
        
        return True
    except Exception as e:
        print(f"❌ Simple plot creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 SIMPLE VISUALIZATION DIAGNOSTIC TEST")
    print("=" * 60)
    
    tests = [
        ("Basic Python", test_basic_python),
        ("Matplotlib Minimal", test_matplotlib_minimal),
        ("Scipy Minimal", test_scipy_minimal),
        ("Visualization Import", test_visualization_import),
        ("Simple Plot Creation", test_simple_plot_creation)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"RUNNING: {test_name}")
        print('='*60)
        
        try:
            result = test_func()
            results[test_name] = result
            
            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
                
        except Exception as e:
            print(f"❌ {test_name}: EXCEPTION - {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*60}")
    print("🏆 DIAGNOSTIC TEST SUMMARY")
    print('='*60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")
    
    print(f"\nOVERALL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - VISUALIZATION IS WORKING!")
    elif passed >= 4:
        print("\n✅ MOST TESTS PASSED - VISUALIZATION MOSTLY WORKING")
    elif passed >= 2:
        print("\n⚠️  SOME TESTS PASSED - PARTIAL FUNCTIONALITY")
    else:
        print("\n❌ MOST TESTS FAILED - MAJOR ISSUES")
    
    return passed >= 4

if __name__ == "__main__":
    success = main()
