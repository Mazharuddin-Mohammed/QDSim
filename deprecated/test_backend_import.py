
import sys
sys.path.insert(0, "backend/build")
print("Importing backend...")
import fe_interpolator_module as fem
print("Backend imported successfully!")
print("Available classes:", [x for x in dir(fem) if not x.startswith("_")])

