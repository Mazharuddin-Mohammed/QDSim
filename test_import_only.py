#!/usr/bin/env python3
"""
Minimal test script that just imports the module.
"""

import sys
import os

# Add the backend build directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend/build'))

# Import the C++ module directly
import qdsim_cpp as cpp

def main():
    """
    Main function to test importing the module.
    """
    print("Successfully imported the module.")
    print("Available classes:")
    for attr in dir(cpp):
        if not attr.startswith('__'):
            print(f"- {attr}")
    
    print("Test completed successfully!")

if __name__ == "__main__":
    main()
