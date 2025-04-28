import sys
import os

# Add the build directory to the Python path
build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build')
print(f"Build directory: {build_dir}")
print(f"Exists: {os.path.exists(build_dir)}")
sys.path.append(build_dir)

# Try to import the C++ extension
try:
    import qdsim
    print("Successfully imported qdsim")
    print(f"qdsim module: {dir(qdsim)}")
    
    # Check if the required classes and functions exist
    required_attrs = [
        'MaterialDatabase', 'Mesh', 'PoissonSolver', 'FEMSolver', 'EigenSolver',
        'effective_mass', 'potential', 'epsilon_r', 'charge_density', 'cap'
    ]
    
    for attr in required_attrs:
        if hasattr(qdsim, attr):
            print(f"✓ {attr} exists")
        else:
            print(f"✗ {attr} does not exist")
            
    # Check if we can create instances of the classes
    try:
        if hasattr(qdsim, 'Mesh'):
            mesh = qdsim.Mesh(1.0, 1.0, 10, 10, 1)
            print("✓ Created Mesh instance")
    except Exception as e:
        print(f"✗ Failed to create Mesh instance: {e}")
        
except ImportError as e:
    print(f"Error importing qdsim: {e}")
