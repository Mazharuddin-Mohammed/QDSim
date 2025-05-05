#!/bin/bash
# Script to check for "Author: Dr. Mazharuddin Mohammed" in source files

echo "Checking for missing author attribution..."
echo "----------------------------------------"

# Check C++ header files
echo "Checking C++ header files..."
for file in $(find backend/include -name "*.h"); do
    if ! grep -q "Author: Dr. Mazharuddin Mohammed" "$file"; then
        echo "Missing in: $file"
    fi
done

# Check C++ source files
echo "Checking C++ source files..."
for file in $(find backend/src -name "*.cpp" -o -name "*.cu"); do
    if ! grep -q "Author: Dr. Mazharuddin Mohammed" "$file"; then
        echo "Missing in: $file"
    fi
done

# Check Python files
echo "Checking Python files..."
for file in $(find frontend -name "*.py" | grep -v "__pycache__"); do
    if ! grep -q "Author: Dr. Mazharuddin Mohammed" "$file"; then
        echo "Missing in: $file"
    fi
done

# Check example files
echo "Checking example files..."
for file in $(find examples -name "*.py"); do
    if ! grep -q "Author: Dr. Mazharuddin Mohammed" "$file"; then
        echo "Missing in: $file"
    fi
done

echo "----------------------------------------"
echo "Check complete."
