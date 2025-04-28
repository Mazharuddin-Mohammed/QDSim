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
except ImportError as e:
    print(f"Error importing qdsim: {e}")

# Try to import the Python package
try:
    from frontend.qdsim import qdsim_cpp
    print("Successfully imported qdsim_cpp")
    print(f"qdsim_cpp module: {dir(qdsim_cpp)}")
except ImportError as e:
    print(f"Error importing qdsim_cpp: {e}")
