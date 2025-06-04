# cython: language_level = 3

"""
Simplest possible Cython test
"""

def hello():
    """Simple test function"""
    return "Hello from Cython!"

def add_numbers(double a, double b):
    """Simple math function"""
    return a + b

cdef class SimpleClass:
    """Simple Cython class"""
    
    cdef double value
    
    def __cinit__(self, double val=1.0):
        self.value = val
    
    def get_value(self):
        return self.value
    
    def set_value(self, double val):
        self.value = val
