class Config:
    def __init__(self):
        # Physical constants
        self.e_charge = 1.602e-19      # Electron charge (C)
        self.hbar = 1.054e-34          # Reduced Planck's constant (J·s)
        self.m_e = 9.11e-31            # Electron mass (kg)
        self.k_B = 1.381e-23           # Boltzmann constant (J/K)
        self.epsilon_0 = 8.854e-12     # Vacuum permittivity (F/m)

        # Simulation domain
        self.Lx = 100e-9               # Domain width (m)
        self.Ly = 100e-9               # Domain height (m)
        self.nx = 50                   # Initial x-grid points
        self.ny = 50                   # Initial y-grid points

        # Mesh and solver settings
        self.element_order = 3         # Element order: 1 (P1), 2 (P2), 3 (P3)
        self.potential_type = "gaussian"  # Potential type: "square" or "gaussian"
        self.cache_dir = "data/meshes"    # Mesh cache directory
        self.adaptive_threshold = 1e-2    # Refinement threshold
        self.max_refinements = 2          # Max refinement iterations

        # Complex absorbing potential
        self.eta = 0.1 * self.e_charge    # CAP strength (J)

        # Material properties
        self.diode_p_material = "GaAs"    # p-type material
        self.diode_n_material = "GaAs"    # n-type material
        self.qd_material = "InAs"         # QD material
        self.matrix_material = "AlGaAs"   # Matrix material

        # Doping and bias
        self.N_A = 1e24                   # Acceptor doping (m^-3)
        self.N_D = 1e24                   # Donor doping (m^-3)
        self.V_r = 0.0                    # Reverse bias voltage (V)

        # Quantum dot geometry
        self.R = 10e-9                    # QD radius (m)

        # Parallel processing
        self.use_mpi = True               # Enable/disable MPI

        # Additional parameters for pn junction
        self.depletion_width = 50e-9      # Depletion width (m)
        self.junction_position = 0.0      # Junction position (m)

        # Potential depth (in eV, will be converted to J when used)
        self.V_0 = 0.5                    # QD potential depth (eV)
        self.V_bi = 1.0                   # Built-in potential (eV)

        # Temperature
        self.temperature = 300            # Temperature (K)

        self.tolerance = 1e-6  # Self-consistency tolerance
        self.max_iter = 500    # Max self-consistent iterations

    def validate(self):
        """Validate configuration parameters for physical consistency."""
        # Basic validation
        assert self.element_order in [1, 2, 3], "Element order must be 1, 2, or 3"
        assert self.potential_type in ["square", "gaussian"], "Invalid potential type"
        assert self.N_A > 0 and self.N_D > 0, "Doping concentrations must be positive"
        assert self.R > 0, "QD radius must be positive"

        # Physical parameter validation
        assert self.Lx > 0 and self.Ly > 0, "Domain dimensions must be positive"
        assert self.nx > 0 and self.ny > 0, "Grid dimensions must be positive"
        assert self.R < min(self.Lx, self.Ly) / 2, "QD radius must be smaller than half the domain size"

        # Potential validation
        assert self.V_0 >= 0 and self.V_0 <= 100, "QD potential depth must be between 0 and 100 eV"
        assert self.V_bi >= 0 and self.V_bi <= 100, "Built-in potential must be between 0 and 100 eV"
        assert abs(self.V_r) <= 100, "Reverse bias voltage must be between -100 and 100 V"

        # Temperature validation
        assert self.temperature > 0 and self.temperature <= 1000, "Temperature must be between 0 and 1000 K"

        # CAP validation
        assert self.eta >= 0, "CAP strength must be non-negative"

        assert self.tolerance > 0, "Tolerance must be positive"
        assert self.max_iter > 0, "Max iterations must be positive"

        return True

    def get_V_0_J(self):
        """Get QD potential depth in Joules."""
        return self.V_0 * self.e_charge

    def get_V_bi_J(self):
        """Get built-in potential in Joules."""
        return self.V_bi * self.e_charge

    def get_V_r_J(self):
        """Get reverse bias voltage in Joules."""
        return self.V_r * self.e_charge

    def get_thermal_energy(self):
        """Get thermal energy (kT) in eV."""
        return self.k_B * self.temperature / self.e_charge