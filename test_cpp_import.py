import sys
import os

# Add the build directory to the Python path
build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build')
print(f"Build directory: {build_dir}")
print(f"Exists: {os.path.exists(build_dir)}")
sys.path.append(build_dir)

# List all files in the build directory
print("Files in build directory:")
for file in os.listdir(build_dir):
    print(f"  {file}")

# Try to import the C++ extension
try:
    import qdsim_cpp
    print("Successfully imported qdsim_cpp")
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
            
    # Try to create instances of the classes
    try:
        if hasattr(qdsim_cpp, 'Mesh'):
            mesh = qdsim_cpp.Mesh(1.0, 1.0, 10, 10, 1)
            print(f"✓ Created Mesh instance with {mesh.getNumNodes()} nodes and {mesh.getNumElements()} elements")
    except Exception as e:
        print(f"✗ Failed to create Mesh instance: {e}")
        
except ImportError as e:
    print(f"Error importing qdsim_cpp: {e}")
