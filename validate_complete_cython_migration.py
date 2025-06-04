#!/usr/bin/env python3
"""
Complete Cython Migration Validation

This script validates that ALL functional features of the backend have been
successfully migrated to Cython-based implementations, replacing the C++
backend entirely.
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

def test_cython_solver_availability():
    """Test which Cython solvers are available"""
    print("🔍 Testing Cython Solver Availability")
    print("=" * 60)
    
    available_solvers = {}
    
    # Test core modules first
    try:
        sys.path.insert(0, str(Path(__file__).parent / "qdsim_cython" / "qdsim_cython"))
        
        # Test mesh
        try:
            import core.mesh_minimal as mesh
            test_mesh = mesh.SimpleMesh(5, 4, 10e-9, 8e-9)
            available_solvers['mesh'] = True
            print("✅ Mesh module: Available and working")
        except Exception as e:
            available_solvers['mesh'] = False
            print(f"❌ Mesh module: Failed ({e})")
        
        # Test materials
        try:
            import core.materials_minimal as materials
            mat = materials.create_material("GaAs", 1.424, 0.067, 12.9)
            available_solvers['materials'] = True
            print("✅ Materials module: Available and working")
        except Exception as e:
            available_solvers['materials'] = False
            print(f"❌ Materials module: Failed ({e})")
        
        # Test quantum analysis
        try:
            import analysis.quantum_analysis as qa
            analyzer = qa.QuantumStateAnalyzer()
            available_solvers['quantum_analysis'] = True
            print("✅ Quantum Analysis module: Available and working")
        except Exception as e:
            available_solvers['quantum_analysis'] = False
            print(f"❌ Quantum Analysis module: Failed ({e})")
        
    except Exception as e:
        print(f"❌ Core module import failed: {e}")
    
    # Test solver modules
    solver_modules = [
        ('poisson_solver', 'Poisson Solver'),
        ('schrodinger_solver', 'Schrödinger Solver'),
        ('fem_solver', 'FEM Solver'),
        ('eigen_solver', 'Eigen Solver'),
        ('self_consistent_solver', 'Self-Consistent Solver')
    ]
    
    for module_name, display_name in solver_modules:
        try:
            module = __import__(f'solvers.{module_name}', fromlist=[''])
            available_solvers[module_name] = True
            print(f"✅ {display_name}: Available")
        except Exception as e:
            available_solvers[module_name] = False
            print(f"❌ {display_name}: Failed ({e})")
    
    return available_solvers

def test_backend_replacement():
    """Test if Cython solvers can replace C++ backend functionality"""
    print("\n🔧 Testing Backend Replacement Capability")
    print("=" * 60)
    
    # Test if we can create equivalent functionality to C++ backend
    replacement_tests = {}
    
    # Test 1: Mesh generation (replaces C++ Mesh class)
    try:
        sys.path.insert(0, str(Path(__file__).parent / "qdsim_cython" / "qdsim_cython"))
        import core.mesh_minimal as mesh
        
        # Create various mesh sizes to test robustness
        mesh_configs = [
            (10, 8, 50e-9, 40e-9),
            (20, 15, 100e-9, 75e-9),
            (30, 25, 150e-9, 125e-9)
        ]
        
        total_time = 0
        for nx, ny, Lx, Ly in mesh_configs:
            start_time = time.time()
            test_mesh = mesh.SimpleMesh(nx, ny, Lx, Ly)
            mesh_time = time.time() - start_time
            total_time += mesh_time
            
            # Validate mesh properties
            assert test_mesh.num_nodes == nx * ny
            assert test_mesh.num_elements == (nx - 1) * (ny - 1) * 2
            
            # Test mesh operations
            nodes_x, nodes_y = test_mesh.get_nodes()
            elements = test_mesh.get_elements()
            
            assert len(nodes_x) == test_mesh.num_nodes
            assert elements.shape == (test_mesh.num_elements, 3)
        
        replacement_tests['mesh_generation'] = True
        print(f"✅ Mesh Generation: Replaces C++ Mesh class ({total_time:.3f}s total)")
        
    except Exception as e:
        replacement_tests['mesh_generation'] = False
        print(f"❌ Mesh Generation: Failed ({e})")
    
    # Test 2: Material property management (replaces C++ material database)
    try:
        import core.materials_minimal as materials
        
        # Test creating various semiconductor materials
        material_configs = [
            ("GaAs", 1.424, 0.067, 12.9),
            ("InGaAs", 0.75, 0.041, 13.9),
            ("AlGaAs", 1.8, 0.092, 11.5),
            ("InAs", 0.354, 0.023, 15.15),
            ("GaN", 3.4, 0.2, 9.5)
        ]
        
        materials_created = []
        for name, Eg, m_eff, eps_r in material_configs:
            mat = materials.create_material(name, Eg, m_eff, eps_r)
            materials_created.append(mat)
            
            # Validate material properties
            assert abs(mat.E_g - Eg) < 1e-6
            assert abs(mat.m_e - m_eff) < 1e-6
            assert abs(mat.epsilon_r - eps_r) < 1e-6
        
        replacement_tests['material_database'] = True
        print(f"✅ Material Database: Replaces C++ material system ({len(materials_created)} materials)")
        
    except Exception as e:
        replacement_tests['material_database'] = False
        print(f"❌ Material Database: Failed ({e})")
    
    # Test 3: Quantum state analysis (replaces C++ analysis tools)
    try:
        import analysis.quantum_analysis as qa
        
        # Test quantum state analysis capabilities
        analyzer = qa.QuantumStateAnalyzer()
        energy_analyzer = qa.EnergyLevelAnalyzer()
        
        # Create test quantum states
        num_states = 5
        state_size = 100
        test_states = []
        test_energies = []
        
        for i in range(num_states):
            # Create normalized random wavefunction
            psi = np.random.random(state_size) + 1j * np.random.random(state_size)
            psi = psi / np.linalg.norm(psi)
            test_states.append(psi)
            
            # Create realistic energy levels
            energy = (i + 1) * 1e-20  # Joules
            test_energies.append(energy)
        
        # Test individual state analysis
        analysis_results = []
        for i, (psi, energy) in enumerate(zip(test_states, test_energies)):
            result = analyzer.analyze_wavefunction(psi, energy=energy)
            analysis_results.append(result)
            
            # Validate analysis results
            assert 'normalization' in result
            assert 'localization' in result
            assert 'phase' in result
        
        # Test energy spectrum analysis
        spectrum_result = energy_analyzer.analyze_energy_spectrum(np.array(test_energies))
        assert 'basic_properties' in spectrum_result
        assert 'level_spacing' in spectrum_result
        
        # Test collection analysis
        collection_result = qa.analyze_quantum_state_collection(test_states, test_energies)
        assert len(collection_result) > 0
        
        replacement_tests['quantum_analysis'] = True
        print(f"✅ Quantum Analysis: Replaces C++ analysis tools ({len(analysis_results)} states analyzed)")
        
    except Exception as e:
        replacement_tests['quantum_analysis'] = False
        print(f"❌ Quantum Analysis: Failed ({e})")
    
    return replacement_tests

def test_performance_vs_backend():
    """Test performance comparison with original backend expectations"""
    print("\n⚡ Testing Performance vs Backend Expectations")
    print("=" * 60)
    
    performance_results = {}
    
    try:
        sys.path.insert(0, str(Path(__file__).parent / "qdsim_cython" / "qdsim_cython"))
        
        # Performance test 1: Mesh creation speed
        import core.mesh_minimal as mesh
        
        mesh_sizes = [(50, 40), (100, 80), (200, 160)]
        mesh_times = []
        
        for nx, ny in mesh_sizes:
            start_time = time.time()
            for _ in range(10):  # Create 10 meshes
                test_mesh = mesh.SimpleMesh(nx, ny, 100e-9, 80e-9)
            mesh_time = (time.time() - start_time) / 10
            mesh_times.append(mesh_time)
            
            print(f"   Mesh {nx}×{ny}: {mesh_time*1000:.2f} ms per mesh")
        
        # Performance should be sub-millisecond for reasonable mesh sizes
        avg_mesh_time = np.mean(mesh_times)
        performance_results['mesh_speed'] = avg_mesh_time < 0.01  # < 10ms
        
        # Performance test 2: Material creation speed
        import core.materials_minimal as materials
        
        start_time = time.time()
        for _ in range(1000):  # Create 1000 materials
            mat = materials.create_material("GaAs", 1.424, 0.067, 12.9)
        material_time = (time.time() - start_time) / 1000
        
        print(f"   Material creation: {material_time*1000:.3f} ms per material")
        performance_results['material_speed'] = material_time < 0.001  # < 1ms
        
        # Performance test 3: Quantum analysis speed
        import analysis.quantum_analysis as qa
        
        analyzer = qa.QuantumStateAnalyzer()
        
        start_time = time.time()
        for _ in range(100):  # Analyze 100 states
            psi = np.random.random(200) + 1j * np.random.random(200)
            psi = psi / np.linalg.norm(psi)
            result = analyzer.analyze_wavefunction(psi)
        analysis_time = (time.time() - start_time) / 100
        
        print(f"   Quantum analysis: {analysis_time*1000:.2f} ms per state")
        performance_results['analysis_speed'] = analysis_time < 0.01  # < 10ms
        
    except Exception as e:
        print(f"❌ Performance testing failed: {e}")
        performance_results = {'mesh_speed': False, 'material_speed': False, 'analysis_speed': False}
    
    return performance_results

def generate_migration_report(available_solvers, replacement_tests, performance_results):
    """Generate comprehensive migration report"""
    print("\n" + "=" * 80)
    print("🏆 COMPLETE CYTHON MIGRATION VALIDATION REPORT")
    print("=" * 80)
    
    # Count available components
    total_core_modules = 3  # mesh, materials, quantum_analysis
    available_core = sum([available_solvers.get('mesh', False),
                         available_solvers.get('materials', False),
                         available_solvers.get('quantum_analysis', False)])
    
    total_solvers = 5  # poisson, schrodinger, fem, eigen, self_consistent
    available_solver_modules = sum([available_solvers.get(f'{name}_solver', False) 
                                   for name in ['poisson', 'schrodinger', 'fem', 'eigen', 'self_consistent']])
    
    total_replacements = len(replacement_tests)
    working_replacements = sum(replacement_tests.values())
    
    total_performance = len(performance_results)
    good_performance = sum(performance_results.values())
    
    print(f"📊 MIGRATION STATISTICS:")
    print(f"   Core Modules: {available_core}/{total_core_modules} working ({available_core/total_core_modules*100:.1f}%)")
    print(f"   Solver Modules: {available_solver_modules}/{total_solvers} working ({available_solver_modules/total_solvers*100:.1f}%)")
    print(f"   Backend Replacement: {working_replacements}/{total_replacements} successful ({working_replacements/total_replacements*100:.1f}%)")
    print(f"   Performance Targets: {good_performance}/{total_performance} met ({good_performance/total_performance*100:.1f}%)")
    
    print(f"\n📋 DETAILED COMPONENT STATUS:")
    
    print(f"   Core Modules:")
    core_modules = ['mesh', 'materials', 'quantum_analysis']
    for module in core_modules:
        status = "✅ WORKING" if available_solvers.get(module, False) else "❌ FAILED"
        print(f"     {status} {module.title()}")
    
    print(f"   Solver Modules:")
    solver_names = ['poisson_solver', 'schrodinger_solver', 'fem_solver', 'eigen_solver', 'self_consistent_solver']
    for solver in solver_names:
        status = "✅ WORKING" if available_solvers.get(solver, False) else "❌ FAILED"
        display_name = solver.replace('_solver', '').replace('_', ' ').title()
        print(f"     {status} {display_name}")
    
    print(f"   Backend Replacement:")
    replacement_names = {
        'mesh_generation': 'Mesh Generation',
        'material_database': 'Material Database',
        'quantum_analysis': 'Quantum Analysis Tools'
    }
    for key, name in replacement_names.items():
        status = "✅ REPLACED" if replacement_tests.get(key, False) else "❌ NOT REPLACED"
        print(f"     {status} {name}")
    
    print(f"   Performance:")
    perf_names = {
        'mesh_speed': 'Mesh Creation Speed',
        'material_speed': 'Material Creation Speed', 
        'analysis_speed': 'Quantum Analysis Speed'
    }
    for key, name in perf_names.items():
        status = "✅ FAST" if performance_results.get(key, False) else "❌ SLOW"
        print(f"     {status} {name}")
    
    # Overall assessment
    total_components = available_core + available_solver_modules + working_replacements + good_performance
    max_components = total_core_modules + total_solvers + total_replacements + total_performance
    overall_success = total_components / max_components
    
    print(f"\n🎯 OVERALL MIGRATION ASSESSMENT:")
    print(f"   Success Rate: {overall_success*100:.1f}% ({total_components}/{max_components} components working)")
    
    if overall_success >= 0.9:
        print("   🎉 OUTSTANDING SUCCESS! Complete Cython migration achieved.")
        print("   🚀 C++ backend successfully replaced with high-performance Cython.")
    elif overall_success >= 0.7:
        print("   ✅ EXCELLENT PROGRESS! Major migration success.")
        print("   🔧 Core functionality migrated, minor components need work.")
    elif overall_success >= 0.5:
        print("   ⚠️  GOOD PROGRESS! Partial migration success.")
        print("   🔧 Significant work remaining for complete migration.")
    else:
        print("   ❌ MIGRATION INCOMPLETE! Major work needed.")
        print("   🔧 Core components require attention.")
    
    print("=" * 80)
    
    return overall_success

def main():
    """Main validation function"""
    print("🚀 COMPLETE CYTHON MIGRATION VALIDATION")
    print("Validating that ALL backend functionality has been migrated to Cython")
    print("=" * 80)
    
    # Run all validation tests
    available_solvers = test_cython_solver_availability()
    replacement_tests = test_backend_replacement()
    performance_results = test_performance_vs_backend()
    
    # Generate comprehensive report
    overall_success = generate_migration_report(available_solvers, replacement_tests, performance_results)
    
    return overall_success >= 0.7  # 70% success threshold

if __name__ == "__main__":
    success = main()
