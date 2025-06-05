#!/bin/bash

# Test QDSim Backend Outside VSCode
# Run this script in a system terminal (not VSCode terminal)

echo "========================================================================"
echo "QDSim Backend Test - Outside VSCode Environment"
echo "========================================================================"

# Navigate to project directory
cd /home/madmax/Documents/dev/projects/QDSim

echo "Current directory: $(pwd)"
echo "Python location: $(which python3)"
echo "Python version: $(python3 --version)"
echo

# Test 1: Basic Python execution
echo "=== TEST 1: Basic Python Execution ==="
timeout 5 python3 -c "print('✅ Basic Python works!')" && echo "SUCCESS" || echo "FAILED - Python hanging"
echo

# Test 2: Simple import test
echo "=== TEST 2: Simple Import Test ==="
timeout 5 python3 -c "import sys; print('✅ Imports work!')" && echo "SUCCESS" || echo "FAILED - Import hanging"
echo

# Test 3: Backend module existence
echo "=== TEST 3: Backend Module File Check ==="
if [ -f "backend/build/fe_interpolator_module.cpython-312-x86_64-linux-gnu.so" ]; then
    echo "✅ Backend module file exists"
    ls -la backend/build/fe_interpolator_module.cpython-312-x86_64-linux-gnu.so
    echo "File size: $(stat -c%s backend/build/fe_interpolator_module.cpython-312-x86_64-linux-gnu.so) bytes"
else
    echo "❌ Backend module file missing"
fi
echo

# Test 4: Backend module dependencies
echo "=== TEST 4: Backend Module Dependencies ==="
echo "Checking shared library dependencies..."
ldd backend/build/fe_interpolator_module.cpython-312-x86_64-linux-gnu.so | head -10
echo
missing_deps=$(ldd backend/build/fe_interpolator_module.cpython-312-x86_64-linux-gnu.so 2>&1 | grep "not found" | wc -l)
if [ "$missing_deps" -eq 0 ]; then
    echo "✅ No missing dependencies"
else
    echo "❌ $missing_deps missing dependencies found"
fi
echo

# Test 5: Backend import test (with timeout)
echo "=== TEST 5: Backend Import Test ==="
echo "Attempting to import backend module..."
timeout 10 python3 -c "
import sys
sys.path.insert(0, 'backend/build')
print('Importing fe_interpolator_module...')
import fe_interpolator_module as fem
print('✅ Backend imported successfully!')
print('Available classes:', [x for x in dir(fem) if not x.startswith('_')])
" && echo "IMPORT SUCCESS" || echo "IMPORT FAILED - Module hanging"
echo

# Test 6: Materials module test
echo "=== TEST 6: Materials Module Test ==="
echo "Testing materials module..."
timeout 10 python3 -c "
import materials_minimal
print('✅ Materials module imported!')
mat = materials_minimal.create_material()
print('✅ Material created:', mat)
mat.m_e = 0.041
print('✅ Property modified:', mat.m_e)
" && echo "MATERIALS SUCCESS" || echo "MATERIALS FAILED"
echo

# Test 7: Combined test
echo "=== TEST 7: Combined Backend + Materials Test ==="
timeout 15 python3 -c "
import sys
sys.path.insert(0, 'backend/build')

print('Testing combined import...')
import fe_interpolator_module as fem
import materials_minimal

print('✅ Both modules imported!')

# Test FEM functionality
mesh = fem.Mesh(50e-9, 50e-9, 16, 16, 1)
print(f'✅ Mesh created: {mesh.get_num_nodes()} nodes')

# Test materials functionality  
mat = materials_minimal.create_material()
mat.m_e = 0.041
print(f'✅ Material created with m_e = {mat.m_e}')

# Check for eigensolvers
if hasattr(fem, 'SchrodingerSolver'):
    print('🎉 SchrodingerSolver found!')
else:
    print('❌ SchrodingerSolver not found')

if hasattr(fem, 'EigenSolver'):
    print('🎉 EigenSolver found!')
else:
    print('❌ EigenSolver not found')

print('✅ Combined test completed successfully!')
" && echo "COMBINED SUCCESS" || echo "COMBINED FAILED"
echo

# Summary
echo "========================================================================"
echo "TEST SUMMARY"
echo "========================================================================"
echo "If all tests pass, the backend is working and ready for quantum simulations."
echo "If tests fail, there may be a compilation or environment issue."
echo
echo "Next steps if successful:"
echo "1. Test real SchrodingerSolver"
echo "2. Create quantum device mesh"
echo "3. Solve actual Schrödinger equation"
echo "4. Validate physics results"
echo
echo "Run this script outside VSCode for best results:"
echo "  bash test_outside_vscode.sh"
