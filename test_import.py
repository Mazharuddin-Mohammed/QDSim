import sys
import os

# Add the build directory to the Python path
build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build')
print(f"Build directory: {build_dir}")
print(f"Exists: {os.path.exists(build_dir)}")
sys.path.append(build_dir)

# List files in the build directory
print("Files in build directory:")
for file in os.listdir(build_dir):
    print(f"  {file}")

# Try to import the C++ extension
try:
    import libqdsim
    print("Successfully imported libqdsim")
    print(f"libqdsim module: {dir(libqdsim)}")
except ImportError as e:
    print(f"Error importing libqdsim: {e}")

# Try to import the Python package
try:
    import qdsim
    print("Successfully imported qdsim")
    print(f"qdsim module: {dir(qdsim)}")
except ImportError as e:
    print(f"Error importing qdsim: {e}")
