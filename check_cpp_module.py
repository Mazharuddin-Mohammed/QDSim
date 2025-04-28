import sys
import os
import importlib.util

# Add the build directory to the Python path
build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build')
print(f"Build directory: {build_dir}")
print(f"Exists: {os.path.exists(build_dir)}")

# List all files in the build directory
print("Files in build directory:")
for file in os.listdir(build_dir):
    print(f"  {file}")

# Try to import the C++ extension directly
cpp_module_path = os.path.join(build_dir, "qdsim.cpython-312-x86_64-linux-gnu.so")
if os.path.exists(cpp_module_path):
    print(f"Found C++ module at: {cpp_module_path}")
    
    # Load the module directly
    spec = importlib.util.spec_from_file_location("qdsim_cpp", cpp_module_path)
    qdsim_cpp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qdsim_cpp)
    
    print(f"qdsim_cpp module: {dir(qdsim_cpp)}")
    
    # Check if the required classes and functions exist
    required_attrs = [
        'MaterialDatabase', 'Mesh', 'PoissonSolver', 'FEMSolver', 'EigenSolver',
        'effective_mass', 'potential', 'epsilon_r', 'charge_density', 'cap'
    ]
    
    for attr in required_attrs:
        if hasattr(qdsim_cpp, attr):
            print(f"✓ {attr} exists")
        else:
            print(f"✗ {attr} does not exist")
else:
    print(f"C++ module not found at: {cpp_module_path}")
