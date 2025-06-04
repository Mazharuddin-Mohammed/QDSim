import sys
import os

# Add the build directory to the Python path
build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build')
sys.path.append(build_dir)

# Import the C++ extension directly
import qdsim

def main():
    # Print the available classes and functions
    print("Available classes and functions:")
    for item in dir(qdsim):
        if not item.startswith('__'):
            print(f"  {item}")
    
    # Try to create a simple mesh
    try:
        mesh = qdsim.Mesh(1.0, 1.0, 10, 10, 1)
        print(f"Created mesh with {mesh.getNumNodes()} nodes and {mesh.getNumElements()} elements")
    except Exception as e:
        print(f"Error creating mesh: {e}")

if __name__ == "__main__":
    main()
