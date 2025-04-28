class Config:
    def __init__(self):
        self.Lx = 100e-9          # Domain width (m)
        self.Ly = 100e-9          # Domain height (m)
        self.nx = 50              # Initial x-grid points
        self.ny = 50              # Initial y-grid points
        self.element_order = 3    # Element order: 1 (P1), 2 (P2), 3 (P3)
        self.potential_type = "gaussian"  # Potential type: "square" or "gaussian"
        self.cache_dir = "data/meshes"    # Mesh cache directory
        self.adaptive_threshold = 1e-2    # Refinement threshold
        self.max_refinements = 2          # Max refinement iterations
        self.hbar = 1.054e-34             # Reduced Planck's constant (J·s)
        self.eta = 0.1 * 1.602e-19        # CAP strength (J)
        # New fields
        self.diode_p_material = "GaAs"    # p-type material
        self.diode_n_material = "GaAs"    # n-type material
        self.qd_material = "InAs"         # QD material
        self.matrix_material = "AlGaAs"   # Matrix material
        self.N_A = 1e24                   # Acceptor doping (m^-3)
        self.N_D = 1e24                   # Donor doping (m^-3)
        self.V_r = 0.0                    # Reverse bias voltage (V)
        self.R = 10e-9                    # QD radius (m)
        # MPI can be enabled/disabled at runtime
        self.use_mpi = True               # Enable/disable MPI

    def validate(self):
        assert self.element_order in [1, 2, 3], "Element order must be 1, 2, or 3"
        assert self.potential_type in ["square", "gaussian"], "Invalid potential type"
        assert self.N_A > 0 and self.N_D > 0, "Doping concentrations must be positive"
        assert self.R > 0, "QD radius must be positive"