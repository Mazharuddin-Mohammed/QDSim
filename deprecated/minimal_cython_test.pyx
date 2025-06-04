# cython: language_level = 3

"""
Absolute minimal Cython test
No dependencies, just basic functionality
"""

def hello_world():
    """Simple function that returns a string"""
    return "Hello from Cython!"

def add_numbers(double a, double b):
    """Simple math function"""
    return a + b

cdef class SimpleTest:
    """Minimal Cython class"""
    
    cdef double value
    
    def __init__(self, double val=42.0):
        self.value = val
    
    def get_value(self):
        return self.value
    
    def set_value(self, double val):
        self.value = val
    
    def __str__(self):
        return f"SimpleTest(value={self.value})"
