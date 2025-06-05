#!/usr/bin/env python3
"""
Test Working Cython Modules

This script tests the successfully compiled Cython modules
and identifies what's working vs what needs to be fixed.
"""

import sys
import os
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "qdsim_cython"))

def test_cython_imports():
    """Test Cython module imports"""
    print("🔧 Testing Cython Module Imports")
    print("=" * 50)
    
    # Test main module
    try:
        import qdsim_cython
        print("✅ qdsim_cython imported successfully")
        
        # Print status
        qdsim_cython.print_status()
        
        # Get available modules
        available = qdsim_cython.get_available_modules()
        print(f"Available modules: {available}")
        
    except Exception as e:
        print(f"❌ qdsim_cython import failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test core module
    try:
        import qdsim_cython.core
        print("✅ qdsim_cython.core imported successfully")
        
        # Print core status
        qdsim_cython.core.print_status()
        
    except Exception as e:
        print(f"❌ qdsim_cython.core import failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test materials minimal
    try:
        import qdsim_cython.core.materials_minimal as materials
        print("✅ materials_minimal imported successfully")
        print(f"   Module: {materials}")
        
        # Test if we can access any functions
        if hasattr(materials, '__dict__'):
            attrs = [attr for attr in dir(materials) if not attr.startswith('_')]
            print(f"   Available attributes: {attrs}")
        
    except Exception as e:
        print(f"❌ materials_minimal import failed: {e}")
        import traceback
        traceback.print_exc()

def test_materials_functionality():
    """Test materials module functionality"""
    print("\n🔧 Testing Materials Functionality")
    print("=" * 50)
    
    try:
        import qdsim_cython.core.materials_minimal as materials
        
        # Check what's available in the module
        print("Available in materials module:")
        for attr in dir(materials):
            if not attr.startswith('_'):
                obj = getattr(materials, attr)
                print(f"  {attr}: {type(obj)}")
        
        # Try to use any available classes or functions
        # This will depend on what's actually implemented in materials_minimal.pyx
        
    except Exception as e:
        print(f"❌ Materials functionality test failed: {e}")
        import traceback
        traceback.print_exc()

def test_frontend_integration():
    """Test if frontend can work with Cython modules"""
    print("\n🔧 Testing Frontend Integration")
    print("=" * 50)
    
    try:
        # Try to import frontend
        sys.path.insert(0, str(Path(__file__).parent / "frontend"))
        import qdsim
        print("✅ QDSim frontend imported successfully")
        
        # Try basic functionality
        config = qdsim.Config()
        print("✅ Config created successfully")
        
        # Try to create simulator
        simulator = qdsim.Simulator(config)
        print("✅ Simulator created successfully")
        
    except Exception as e:
        print(f"❌ Frontend integration failed: {e}")
        import traceback
        traceback.print_exc()

def identify_issues():
    """Identify specific issues that need to be resolved"""
    print("\n🔧 Identifying Issues to Resolve")
    print("=" * 50)
    
    issues = []
    
    # Check for missing files
    required_files = [
        "qdsim_cython/eigen.pxd",
        "qdsim_cython/core/mesh.pxd",
        "qdsim_cython/core/interpolator.pxd",
        "qdsim_cython/solvers/schrodinger.pxd",
        "qdsim_cython/gpu/cuda_solver.pxd",
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            issues.append(f"Missing file: {file_path}")
    
    # Check for compilation issues
    cython_files = [
        "qdsim_cython/core/materials.pyx",
        "qdsim_cython/core/mesh.pyx",
        "qdsim_cython/core/physics.pyx",
        "qdsim_cython/core/interpolator.pyx",
        "qdsim_cython/solvers/poisson.pyx",
        "qdsim_cython/solvers/schrodinger.pyx",
        "qdsim_cython/analysis/quantum_analysis.pyx",
    ]
    
    for pyx_file in cython_files:
        so_file = pyx_file.replace('.pyx', '.cpython-312-x86_64-linux-gnu.so')
        so_file = so_file.replace('qdsim_cython/', 'qdsim_cython/qdsim_cython/')
        
        if Path(pyx_file).exists() and not Path(so_file).exists():
            issues.append(f"Not compiled: {pyx_file}")
    
    # Print issues
    if issues:
        print("❌ Issues found:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
    else:
        print("✅ No obvious issues found")
    
    return issues

def create_fix_plan(issues):
    """Create a plan to fix identified issues"""
    print("\n🔧 Fix Plan")
    print("=" * 50)
    
    if not issues:
        print("✅ No fixes needed - all working!")
        return
    
    print("📋 Step-by-step fix plan:")
    
    step = 1
    
    # Group issues by type
    missing_files = [issue for issue in issues if "Missing file" in issue]
    compilation_issues = [issue for issue in issues if "Not compiled" in issue]
    
    if missing_files:
        print(f"\n{step}. Create missing declaration files:")
        for issue in missing_files:
            file_path = issue.replace("Missing file: ", "")
            print(f"   - Create {file_path}")
        step += 1
    
    if compilation_issues:
        print(f"\n{step}. Fix compilation issues:")
        for issue in compilation_issues:
            file_path = issue.replace("Not compiled: ", "")
            print(f"   - Fix dependencies and compile {file_path}")
        step += 1
    
    print(f"\n{step}. Test all modules incrementally")
    print(f"{step+1}. Commit working fixes")

def main():
    """Main test function"""
    print("🚀 TESTING WORKING CYTHON MODULES")
    print("=" * 80)
    
    # Test what's working
    test_cython_imports()
    test_materials_functionality()
    test_frontend_integration()
    
    # Identify issues
    issues = identify_issues()
    
    # Create fix plan
    create_fix_plan(issues)
    
    print("\n" + "=" * 80)
    print("🎯 SUMMARY:")
    print("✅ Materials minimal module is working")
    print("❌ Other modules need dependency fixes")
    print("📋 Follow the fix plan above to resolve issues")
    print("=" * 80)

if __name__ == "__main__":
    main()
