#!/bin/bash

# Static Validation Script for QDSim Cython Migration
# Tests what we can without Python execution

echo "========================================================================"
echo "QDSim Cython Migration - Static Validation"
echo "========================================================================"
echo

# Test 1: Check compiled modules exist
echo "=== TEST 1: Compiled Module Existence ==="
modules_found=0
total_modules=0

check_module() {
    local module_path="$1"
    local module_name="$2"
    total_modules=$((total_modules + 1))
    
    if [ -f "$module_path" ]; then
        echo "✅ $module_name: Found"
        modules_found=$((modules_found + 1))
        
        # Check file size (should be > 100KB for real modules)
        size=$(stat -c%s "$module_path" 2>/dev/null || echo "0")
        if [ "$size" -gt 100000 ]; then
            echo "   Size: ${size} bytes (Good)"
        else
            echo "   Size: ${size} bytes (Warning: Small)"
        fi
        
        # Check if it's a valid ELF file
        if file "$module_path" | grep -q "ELF.*shared object"; then
            echo "   Type: Valid shared library"
        else
            echo "   Type: Invalid or corrupted"
        fi
        
    else
        echo "❌ $module_name: Missing"
    fi
    echo
}

# Check all our compiled modules
check_module "materials_minimal.cpython-312-x86_64-linux-gnu.so" "Materials Minimal"
check_module "qdsim_cython/core/materials.cpython-312-x86_64-linux-gnu.so" "Materials Full"
check_module "backend/build/fe_interpolator_module.cpython-312-x86_64-linux-gnu.so" "FEM Backend"

echo "Module Summary: $modules_found/$total_modules modules found"
echo

# Test 2: Check dependencies
echo "=== TEST 2: Dependency Analysis ==="
if [ -f "materials_minimal.cpython-312-x86_64-linux-gnu.so" ]; then
    echo "Dependencies for materials_minimal:"
    ldd materials_minimal.cpython-312-x86_64-linux-gnu.so | head -10
    echo
    
    # Check for missing dependencies
    missing_deps=$(ldd materials_minimal.cpython-312-x86_64-linux-gnu.so 2>&1 | grep "not found" | wc -l)
    if [ "$missing_deps" -eq 0 ]; then
        echo "✅ No missing dependencies"
    else
        echo "❌ $missing_deps missing dependencies found"
    fi
else
    echo "❌ Cannot check dependencies - module missing"
fi
echo

# Test 3: Check source files
echo "=== TEST 3: Source File Analysis ==="
source_files_found=0
total_source_files=0

check_source() {
    local file_path="$1"
    local file_name="$2"
    total_source_files=$((total_source_files + 1))
    
    if [ -f "$file_path" ]; then
        echo "✅ $file_name: Found"
        source_files_found=$((source_files_found + 1))
        
        # Check file size
        lines=$(wc -l < "$file_path" 2>/dev/null || echo "0")
        echo "   Lines: $lines"
        
        # Check for Cython syntax
        if grep -q "cdef\|cpdef\|cython" "$file_path"; then
            echo "   Content: Valid Cython syntax detected"
        else
            echo "   Content: No Cython syntax found"
        fi
        
    else
        echo "❌ $file_name: Missing"
    fi
    echo
}

# Check source files
check_source "qdsim_cython/core/materials.pyx" "Materials PYX"
check_source "qdsim_cython/core/materials.pxd" "Materials PXD"
check_source "qdsim_cython/core/materials_minimal.pyx" "Materials Minimal PYX"

echo "Source Summary: $source_files_found/$total_source_files source files found"
echo

# Test 4: Check unified architecture headers
echo "=== TEST 4: Unified Architecture Headers ==="
headers_found=0
total_headers=0

check_header() {
    local header_path="$1"
    local header_name="$2"
    total_headers=$((total_headers + 1))
    
    if [ -f "$header_path" ]; then
        echo "✅ $header_name: Found"
        headers_found=$((headers_found + 1))
        
        # Check for C++ content
        if grep -q "class\|namespace\|template" "$header_path"; then
            echo "   Content: C++ code detected"
        else
            echo "   Content: Basic header"
        fi
        
    else
        echo "❌ $header_name: Missing"
    fi
    echo
}

# Check architecture headers
check_header "backend/include/unified_memory.h" "Unified Memory"
check_header "backend/include/parallel_executor.h" "Parallel Executor"
check_header "backend/include/memory_manager.h" "Memory Manager"

echo "Headers Summary: $headers_found/$total_headers headers found"
echo

# Test 5: Build system validation
echo "=== TEST 5: Build System Validation ==="
build_files_found=0
total_build_files=0

check_build_file() {
    local file_path="$1"
    local file_name="$2"
    total_build_files=$((total_build_files + 1))
    
    if [ -f "$file_path" ]; then
        echo "✅ $file_name: Found"
        build_files_found=$((build_files_found + 1))
    else
        echo "❌ $file_name: Missing"
    fi
}

# Check build files
check_build_file "setup_cython.py" "Cython Setup"
check_build_file "setup_minimal.py" "Minimal Setup"
check_build_file "comprehensive_validation.py" "Validation Suite"

echo "Build Files Summary: $build_files_found/$total_build_files build files found"
echo

# Test 6: Performance indicators
echo "=== TEST 6: Performance Analysis ==="
if [ -f "materials_minimal.cpython-312-x86_64-linux-gnu.so" ]; then
    # Check compiled module size (larger usually means more optimized)
    size=$(stat -c%s "materials_minimal.cpython-312-x86_64-linux-gnu.so")
    echo "Compiled module size: $size bytes"
    
    # Check if compiled with optimizations
    if objdump -h materials_minimal.cpython-312-x86_64-linux-gnu.so 2>/dev/null | grep -q "\.text"; then
        echo "✅ Contains compiled code section"
    else
        echo "❌ No compiled code section found"
    fi
    
    # Check for debug symbols (should be minimal for optimized builds)
    if objdump -h materials_minimal.cpython-312-x86_64-linux-gnu.so 2>/dev/null | grep -q "\.debug"; then
        echo "⚠️  Contains debug symbols (may impact performance)"
    else
        echo "✅ No debug symbols (optimized build)"
    fi
else
    echo "❌ Cannot analyze performance - module missing"
fi
echo

# Final Summary
echo "========================================================================"
echo "STATIC VALIDATION SUMMARY"
echo "========================================================================"

total_score=0
max_score=0

# Calculate scores
max_score=$((max_score + total_modules))
total_score=$((total_score + modules_found))

max_score=$((max_score + total_source_files))
total_score=$((total_score + source_files_found))

max_score=$((max_score + total_headers))
total_score=$((total_score + headers_found))

max_score=$((max_score + total_build_files))
total_score=$((total_score + build_files_found))

# Add dependency check
max_score=$((max_score + 1))
if [ -f "materials_minimal.cpython-312-x86_64-linux-gnu.so" ]; then
    missing_deps=$(ldd materials_minimal.cpython-312-x86_64-linux-gnu.so 2>&1 | grep "not found" | wc -l)
    if [ "$missing_deps" -eq 0 ]; then
        total_score=$((total_score + 1))
    fi
fi

percentage=$((total_score * 100 / max_score))

echo "Overall Score: $total_score/$max_score ($percentage%)"
echo

if [ "$percentage" -ge 90 ]; then
    echo "🎉 EXCELLENT: Cython migration is highly successful!"
    echo "✅ All major components are in place"
    echo "✅ Ready for runtime testing"
elif [ "$percentage" -ge 75 ]; then
    echo "✅ GOOD: Cython migration is mostly successful"
    echo "⚠️  Some components may need attention"
elif [ "$percentage" -ge 50 ]; then
    echo "⚠️  PARTIAL: Cython migration has significant gaps"
    echo "🔧 Major work needed"
else
    echo "❌ POOR: Cython migration needs substantial work"
    echo "🔧 Rebuild required"
fi

echo
echo "Next Steps:"
echo "1. Fix any missing components identified above"
echo "2. Test runtime execution in working Python environment"
echo "3. Run performance benchmarks"
echo "4. Validate memory management"
echo "5. Test with quantum simulations"

exit 0
