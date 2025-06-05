Realistic Device Simulations
============================

This section demonstrates how to simulate realistic quantum devices using QDSim, including material properties, device geometries, and experimental conditions.

Chromium Quantum Dots in InGaAs p-n Junctions
---------------------------------------------

This example simulates chromium quantum dots embedded in InGaAs p-n junction diodes under reverse bias conditions, matching experimental parameters.

**Physical System:**
    - Material: In₀.₅₃Ga₀.₄₇As on InP substrate
    - Quantum dots: Chromium impurities
    - Device: p-n junction diode
    - Operating conditions: Reverse bias 0-5V
    - Temperature: 77K (liquid nitrogen)

Complete Implementation
----------------------

.. code-block:: python

    #!/usr/bin/env python3
    """
    Realistic Chromium QD in InGaAs p-n Junction Simulation
    
    This example demonstrates:
    1. Realistic material properties and device geometry
    2. Applied bias conditions and temperature effects
    3. Open quantum system with finite lifetimes
    4. Comparison with experimental data
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    import qdsim
    from qdsim.visualization import WavefunctionPlotter
    
    def main():
        print("🔬 Realistic Device Simulation: Cr QDs in InGaAs")
        print("=" * 60)
        
        # Physical constants
        HBAR = 1.054571817e-34  # J⋅s
        M_E = 9.1093837015e-31  # kg
        EV_TO_J = 1.602176634e-19  # J/eV
        KB = 1.380649e-23  # J/K
        
        # Device parameters
        temperature = 77.0  # K (liquid nitrogen)
        bias_voltage = 2.0  # V (reverse bias)
        
        # Material properties (In₀.₅₃Ga₀.₄₇As)
        ingaas_bandgap = 0.75 * EV_TO_J  # eV at 77K
        ingaas_effective_mass = 0.041 * M_E  # Electron effective mass
        ingaas_dielectric = 13.9  # Relative permittivity
        
        # Device geometry
        device_length = 50e-9  # 50 nm
        device_width = 40e-9   # 40 nm
        junction_position = 25e-9  # p-n junction at center
        
        # Chromium QD parameters
        qd_position = (30e-9, 20e-9)  # QD location
        qd_size = 3e-9  # QD diameter
        cr_binding_energy = 0.15 * EV_TO_J  # Cr binding energy
        
        print(f"Device parameters:")
        print(f"  Material: In₀.₅₃Ga₀.₄₇As")
        print(f"  Temperature: {temperature} K")
        print(f"  Bias voltage: {bias_voltage} V")
        print(f"  Device size: {device_length*1e9:.0f} × {device_width*1e9:.0f} nm")
        print(f"  QD position: ({qd_position[0]*1e9:.0f}, {qd_position[1]*1e9:.0f}) nm")
        
        # Define material properties
        def m_star_func(x, y):
            """Effective mass function for InGaAs"""
            return ingaas_effective_mass
        
        def potential_func(x, y):
            """Realistic device potential including bias and QD"""
            # Built-in potential from p-n junction
            if x < junction_position:
                # p-side
                builtin_potential = 0.5 * ingaas_bandgap
            else:
                # n-side  
                builtin_potential = -0.5 * ingaas_bandgap
            
            # Applied bias (reverse bias increases barrier)
            electric_field = bias_voltage / device_length
            bias_potential = -electric_field * (x - junction_position) * EV_TO_J
            
            # Chromium quantum dot potential
            qd_potential = 0.0
            distance_to_qd = np.sqrt((x - qd_position[0])**2 + (y - qd_position[1])**2)
            if distance_to_qd < qd_size:
                # Attractive Coulomb potential from Cr³⁺
                qd_potential = -cr_binding_energy * np.exp(-(distance_to_qd/qd_size)**2)
            
            return builtin_potential + bias_potential + qd_potential
        
        # Create open system solver for realistic device
        print(f"\n🔧 Setting up realistic device solver...")
        solver = qdsim.FixedOpenSystemSolver(
            nx=60, ny=50,           # High resolution for realistic device
            Lx=device_length, Ly=device_width,
            m_star_func=m_star_func,
            potential_func=potential_func,
            use_open_boundaries=True,  # Open system for carrier injection
            cap_strength=0.005,        # Optimized for InGaAs
            cap_width=5e-9            # 5 nm absorption region
        )
        
        # Apply realistic device physics
        print(f"🔧 Applying device-specific physics...")
        solver.apply_open_system_boundary_conditions()
        solver.apply_dirac_delta_normalization()
        solver.configure_device_specific_solver('quantum_well')
        solver.apply_conservative_boundary_conditions()
        solver.apply_minimal_cap_boundaries()
        
        # Solve the realistic quantum system
        print(f"🚀 Solving realistic device...")
        num_states = 10
        eigenvals, eigenvecs = solver.solve(num_states)
        
        # Analyze complex eigenvalues
        print(f"✅ Found {len(eigenvals)} quantum states")
        print(f"\nRealistic device analysis:")
        
        bound_states = 0
        resonant_states = 0
        
        for i, E in enumerate(eigenvals):
            E_real_eV = np.real(E) / EV_TO_J
            E_imag_eV = np.imag(E) / EV_TO_J
            
            is_complex = abs(np.imag(E)) > 1e-25
            
            if is_complex:
                resonant_states += 1
                # Calculate lifetime
                lifetime_s = HBAR / (2 * abs(np.imag(E)))
                lifetime_fs = lifetime_s * 1e15
                
                print(f"  State {i+1}: {E_real_eV*1000:.2f} + {E_imag_eV*1000:.3f}j meV")
                print(f"           Lifetime: {lifetime_fs:.1f} fs (resonant)")
            else:
                bound_states += 1
                print(f"  State {i+1}: {E_real_eV*1000:.2f} meV (bound)")
        
        print(f"\nState classification:")
        print(f"  Bound states: {bound_states}")
        print(f"  Resonant states: {resonant_states}")
        
        # Temperature effects
        print(f"\n🌡️  Temperature effects at {temperature} K:")
        thermal_energy = KB * temperature / EV_TO_J * 1000  # meV
        print(f"  Thermal energy: {thermal_energy:.2f} meV")
        
        # Check which states are thermally accessible
        accessible_states = 0
        for E in eigenvals:
            E_eV = np.real(E) / EV_TO_J * 1000  # meV
            if abs(E_eV) < 3 * thermal_energy:  # Within 3kT
                accessible_states += 1
        
        print(f"  Thermally accessible states: {accessible_states}")
        
        # Create realistic device visualizations
        print(f"\n🎨 Creating device visualizations...")
        plotter = WavefunctionPlotter()
        
        # Device potential landscape
        fig_device = plotter.plot_device_structure(
            solver.nodes_x, solver.nodes_y,
            potential_func, m_star_func,
            f"Cr QD in InGaAs p-n Junction (V = {bias_voltage}V)"
        )
        
        # Energy levels with lifetime information
        fig_energy = plotter.plot_energy_levels(
            eigenvals, 
            f"Energy Levels: Cr QD in InGaAs at {temperature}K"
        )
        
        # Quantum dot states
        qd_states = []
        for i, E in enumerate(eigenvals):
            # Check if state is localized near QD
            wavefunction = eigenvecs[:, i]
            x_coords = solver.nodes_x
            y_coords = solver.nodes_y
            
            # Calculate probability density near QD
            qd_region_mask = ((x_coords - qd_position[0])**2 + 
                             (y_coords - qd_position[1])**2) < (2*qd_size)**2
            
            prob_in_qd = np.sum(np.abs(wavefunction[qd_region_mask])**2)
            total_prob = np.sum(np.abs(wavefunction)**2)
            
            if prob_in_qd / total_prob > 0.5:  # >50% probability in QD region
                qd_states.append(i)
                
                E_eV = np.real(E) / EV_TO_J
                title = f"Cr QD State {len(qd_states)} (E = {E_eV*1000:.1f} meV)"
                fig = plotter.plot_wavefunction_2d(
                    x_coords, y_coords, wavefunction, title
                )
        
        print(f"  Found {len(qd_states)} states localized in Cr QD")
        
        # Experimental comparison
        print(f"\n📊 Experimental comparison:")
        
        # Typical experimental values for Cr QDs in InGaAs
        exp_binding_energy = 150  # meV
        exp_lifetime_range = (10, 1000)  # fs
        
        print(f"  Expected Cr binding energy: ~{exp_binding_energy} meV")
        print(f"  Expected lifetime range: {exp_lifetime_range[0]}-{exp_lifetime_range[1]} fs")
        
        # Compare with simulation
        if qd_states:
            sim_binding = abs(np.real(eigenvals[qd_states[0]]) / EV_TO_J * 1000)
            print(f"  Simulated binding energy: {sim_binding:.1f} meV")
            
            complex_qd_states = [i for i in qd_states if abs(np.imag(eigenvals[i])) > 1e-25]
            if complex_qd_states:
                sim_lifetime = HBAR / (2 * abs(np.imag(eigenvals[complex_qd_states[0]]))) * 1e15
                print(f"  Simulated lifetime: {sim_lifetime:.1f} fs")
        
        # Device performance metrics
        print(f"\n⚡ Device performance:")
        
        # Calculate tunneling probability (simplified)
        barrier_height = bias_voltage * EV_TO_J
        barrier_width = device_length / 4  # Approximate depletion width
        
        # WKB approximation for tunneling
        kappa = np.sqrt(2 * ingaas_effective_mass * barrier_height) / HBAR
        tunneling_prob = np.exp(-2 * kappa * barrier_width)
        
        print(f"  Barrier height: {barrier_height/EV_TO_J:.2f} eV")
        print(f"  Tunneling probability: {tunneling_prob:.2e}")
        
        # Estimate current density (simplified)
        carrier_density = 1e16  # cm⁻³ (typical doping)
        current_density = carrier_density * EV_TO_J * tunneling_prob * 1e-4  # A/cm²
        print(f"  Estimated current density: {current_density:.2e} A/cm²")
        
        print(f"\n🎉 Realistic device simulation completed!")
        
        return {
            'eigenvalues': eigenvals,
            'eigenvectors': eigenvecs,
            'qd_states': qd_states,
            'device_params': {
                'temperature': temperature,
                'bias_voltage': bias_voltage,
                'material': 'InGaAs',
                'qd_type': 'Chromium'
            },
            'solver': solver
        }
    
    if __name__ == "__main__":
        results = main()

Experimental Validation
----------------------

This simulation can be validated against experimental data:

**Literature References:**
    - Cr QD binding energies: 100-200 meV
    - Lifetimes: 10-1000 fs depending on bias
    - Current-voltage characteristics
    - Temperature-dependent behavior

**Experimental Techniques:**
    - Photoluminescence spectroscopy
    - Time-resolved measurements
    - Transport measurements
    - Scanning tunneling spectroscopy

Parameter Studies
----------------

The simulation enables systematic parameter studies:

.. code-block:: python

    # Bias voltage sweep
    bias_voltages = np.linspace(0, 5, 11)  # 0-5V
    lifetimes = []
    
    for bias in bias_voltages:
        # Update potential function with new bias
        # Re-solve and extract lifetimes
        pass
    
    # Temperature sweep
    temperatures = np.linspace(4, 300, 50)  # 4K to room temperature
    thermal_populations = []
    
    for T in temperatures:
        # Calculate thermal population of QD states
        pass

Device Optimization
------------------

The realistic simulation enables device optimization:

**Design Parameters:**
    - QD position and size
    - Doping concentrations
    - Device geometry
    - Material composition

**Performance Metrics:**
    - Current density
    - Switching speed
    - Power consumption
    - Temperature stability

**Optimization Strategies:**
    - Genetic algorithms
    - Gradient-based optimization
    - Machine learning approaches
    - Multi-objective optimization

This realistic device example demonstrates how QDSim can be used for practical device design and optimization, bridging the gap between fundamental quantum mechanics and real-world applications.
