#!/usr/bin/env python3
"""
Comprehensive QDSim Cython Validation Suite
Tests all aspects of the Cython migration when execution environment works
"""

import sys
import os
import time
import gc
import traceback

class QDSimValidator:
    """Comprehensive validation for QDSim Cython implementation"""
    
    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
        self.results = []
    
    def log_result(self, test_name, success, message=""):
        """Log test result"""
        status = "✅ PASS" if success else "❌ FAIL"
        result = f"{status}: {test_name}"
        if message:
            result += f" - {message}"
        
        self.results.append(result)
        print(result)
        
        if success:
            self.tests_passed += 1
        else:
            self.tests_failed += 1
    
    def test_basic_import(self):
        """Test 1: Basic module import"""
        try:
            import materials_minimal
            self.log_result("Basic Import", True, "materials_minimal imported successfully")
            return materials_minimal
        except Exception as e:
            self.log_result("Basic Import", False, f"Import failed: {e}")
            return None
    
    def test_material_creation(self, materials_minimal):
        """Test 2: Material creation and basic functionality"""
        if not materials_minimal:
            self.log_result("Material Creation", False, "Module not available")
            return None
        
        try:
            mat = materials_minimal.create_material()
            self.log_result("Material Creation", True, f"Created: {mat}")
            return mat
        except Exception as e:
            self.log_result("Material Creation", False, f"Creation failed: {e}")
            return None
    
    def test_property_access(self, mat):
        """Test 3: Property access and modification"""
        if not mat:
            self.log_result("Property Access", False, "Material not available")
            return False
        
        try:
            # Test getter
            original_me = mat.m_e
            original_eg = mat.E_g
            
            # Test setter
            mat.m_e = 0.123
            mat.E_g = 1.5
            
            # Verify changes
            if mat.m_e == 0.123 and mat.E_g == 1.5:
                self.log_result("Property Access", True, 
                               f"Properties modified: m_e={mat.m_e}, E_g={mat.E_g}")
                return True
            else:
                self.log_result("Property Access", False, "Property modification failed")
                return False
                
        except Exception as e:
            self.log_result("Property Access", False, f"Property access failed: {e}")
            return False
    
    def test_performance(self, materials_minimal):
        """Test 4: Performance benchmarking"""
        if not materials_minimal:
            self.log_result("Performance", False, "Module not available")
            return False
        
        try:
            start_time = time.time()
            
            # Create 1000 materials
            materials = []
            for i in range(1000):
                mat = materials_minimal.create_material()
                mat.m_e = 0.067 + i * 0.001
                materials.append(mat)
            
            end_time = time.time()
            duration = end_time - start_time
            avg_time = duration * 1000  # ms per 1000 materials
            
            if duration < 1.0:  # Should be fast
                self.log_result("Performance", True, 
                               f"1000 materials in {duration:.3f}s ({avg_time:.3f}ms total)")
                return True
            else:
                self.log_result("Performance", False, 
                               f"Too slow: {duration:.3f}s for 1000 materials")
                return False
                
        except Exception as e:
            self.log_result("Performance", False, f"Performance test failed: {e}")
            return False
    
    def test_memory_management(self, materials_minimal):
        """Test 5: Memory management and RAII"""
        if not materials_minimal:
            self.log_result("Memory Management", False, "Module not available")
            return False
        
        try:
            # Get initial object count
            gc.collect()
            initial_objects = len(gc.get_objects())
            
            # Create and destroy many objects
            for cycle in range(10):
                materials = []
                for i in range(100):
                    mat = materials_minimal.create_material()
                    materials.append(mat)
                del materials
                gc.collect()
            
            # Check final object count
            final_objects = len(gc.get_objects())
            leaked_objects = final_objects - initial_objects
            
            if leaked_objects < 50:  # Allow some variance
                self.log_result("Memory Management", True, 
                               f"Minimal leaks: {leaked_objects} objects")
                return True
            else:
                self.log_result("Memory Management", False, 
                               f"Memory leak detected: {leaked_objects} objects")
                return False
                
        except Exception as e:
            self.log_result("Memory Management", False, f"Memory test failed: {e}")
            return False
    
    def test_backend_module(self):
        """Test 6: Backend FEM module"""
        try:
            sys.path.insert(0, 'backend/build')
            import fe_interpolator_module
            
            # Test basic functionality
            available_attrs = [x for x in dir(fe_interpolator_module) if not x.startswith('_')]
            
            self.log_result("Backend Module", True, 
                           f"FEM backend loaded, {len(available_attrs)} attributes")
            return fe_interpolator_module
            
        except Exception as e:
            self.log_result("Backend Module", False, f"Backend import failed: {e}")
            return None
    
    def test_full_materials_module(self):
        """Test 7: Full materials module"""
        try:
            from qdsim_cython.core import materials
            
            # Test material creation
            mat = materials.create_material("GaAs")
            
            self.log_result("Full Materials Module", True, 
                           f"Full module working: {mat}")
            return materials
            
        except Exception as e:
            self.log_result("Full Materials Module", False, 
                           f"Full materials import failed: {e}")
            return None
    
    def test_unified_architecture(self):
        """Test 8: Unified memory architecture headers"""
        try:
            # Check if unified architecture headers exist
            unified_headers = [
                'backend/include/unified_memory.h',
                'backend/include/parallel_executor.h',
                'backend/include/memory_manager.h'
            ]
            
            existing_headers = []
            for header in unified_headers:
                if os.path.exists(header):
                    existing_headers.append(header)
            
            if len(existing_headers) >= 2:
                self.log_result("Unified Architecture", True, 
                               f"{len(existing_headers)}/{len(unified_headers)} headers exist")
                return True
            else:
                self.log_result("Unified Architecture", False, 
                               f"Only {len(existing_headers)}/{len(unified_headers)} headers found")
                return False
                
        except Exception as e:
            self.log_result("Unified Architecture", False, 
                           f"Architecture test failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run comprehensive validation suite"""
        print("="*70)
        print("QDSim Cython Migration Validation Suite")
        print("="*70)
        print()
        
        # Test 1: Basic import
        materials_minimal = self.test_basic_import()
        
        # Test 2: Material creation
        mat = self.test_material_creation(materials_minimal)
        
        # Test 3: Property access
        self.test_property_access(mat)
        
        # Test 4: Performance
        self.test_performance(materials_minimal)
        
        # Test 5: Memory management
        self.test_memory_management(materials_minimal)
        
        # Test 6: Backend module
        self.test_backend_module()
        
        # Test 7: Full materials module
        self.test_full_materials_module()
        
        # Test 8: Unified architecture
        self.test_unified_architecture()
        
        # Summary
        print()
        print("="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        
        total_tests = self.tests_passed + self.tests_failed
        success_rate = (self.tests_passed / total_tests * 100) if total_tests > 0 else 0
        
        print(f"Tests Passed: {self.tests_passed}")
        print(f"Tests Failed: {self.tests_failed}")
        print(f"Success Rate: {success_rate:.1f}%")
        print()
        
        if self.tests_failed == 0:
            print("🎉 ALL TESTS PASSED!")
            print("✅ Cython migration is successful")
            print("✅ Ready for quantum simulations")
            return 0
        elif success_rate >= 75:
            print("⚠️  MOSTLY SUCCESSFUL")
            print("✅ Core functionality working")
            print("🔧 Some issues need attention")
            return 1
        else:
            print("❌ SIGNIFICANT ISSUES")
            print("🔧 Major debugging required")
            return 2

def main():
    """Main validation function"""
    validator = QDSimValidator()
    return validator.run_all_tests()

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n❌ Validation interrupted by user")
        sys.exit(3)
    except Exception as e:
        print(f"\n❌ Validation failed with exception: {e}")
        traceback.print_exc()
        sys.exit(4)
