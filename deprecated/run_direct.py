import sys
import os
import importlib.util

# Get the absolute path to the C++ extension
cpp_module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build', 'qdsim.cpython-312-x86_64-linux-gnu.so')
print(f"C++ module path: {cpp_module_path}")
print(f"Exists: {os.path.exists(cpp_module_path)}")

# Try to import the C++ extension directly
try:
    # Use a different name to avoid conflicts with the Python package
    spec = importlib.util.spec_from_file_location("qdsim_direct", cpp_module_path)
    qdsim_direct = importlib.util.module_from_spec(spec)
    sys.modules["qdsim_direct"] = qdsim_direct
    spec.loader.exec_module(qdsim_direct)
    
    print("Successfully imported C++ extension")
    print(f"Available classes and functions: {dir(qdsim_direct)}")
    
    # Try to create a simple mesh
    try:
        mesh = qdsim_direct.Mesh(1.0, 1.0, 10, 10, 1)
        print(f"Created mesh with {mesh.getNumNodes()} nodes and {mesh.getNumElements()} elements")
    except Exception as e:
        print(f"Error creating mesh: {e}")
        
except Exception as e:
    print(f"Error importing C++ extension: {e}")
