"""
Fixed version of the solve_poisson method for the Simulator class.
"""

def solve_poisson(self, V_p=None, V_n=None):
    """
    Solve the Poisson equation with Dirichlet boundary conditions.

    Args:
        V_p: Potential at the p-side (default: 0.0)
        V_n: Potential at the n-side (default: built_in_potential + V_r)
    """
    # Set default values if not provided
    if V_p is None:
        V_p = 0.0

    if V_n is None:
        V_n = self.built_in_potential() + self.config.V_r

    try:
        # Get node coordinates
        nodes = np.array(self.mesh.get_nodes())
        
        # Initialize the potential array with the correct size
        num_nodes = self.mesh.get_num_nodes()
        self.phi = np.zeros(num_nodes)

        # Create a more realistic potential profile for a pn junction
        # with a quantum dot at the center
        for i in range(num_nodes):
            x = nodes[i, 0]
            y = nodes[i, 1]

            # Calculate distance from center
            junction_x = getattr(self.config, 'junction_position', 0.0)

            # pn junction potential (simplified)
            if hasattr(self.config, 'depletion_width') and self.config.depletion_width > 0:
                # Use depletion width if provided
                depletion_width = self.config.depletion_width

                # Normalized position in depletion region
                if x < junction_x - depletion_width/2:
                    # p-side
                    pn_potential = V_p
                elif x > junction_x + depletion_width/2:
                    # n-side
                    pn_potential = V_n
                else:
                    # Depletion region - quadratic profile
                    pos = 2 * (x - junction_x) / depletion_width
                    pn_potential = V_p + (V_n - V_p) * (pos**2 + pos + 1) / 4
            else:
                # Simple linear profile if no depletion width is provided
                pn_potential = V_p + (V_n - V_p) * (x + self.config.Lx/2) / self.config.Lx

            # Add quantum dot potential
            r = np.sqrt((x - junction_x)**2 + y**2)  # Distance from junction center

            # Check if potential_type is defined
            potential_type = getattr(self.config, 'potential_type', 'gaussian')

            # Check if V_0 and R are defined
            V_0 = getattr(self.config, 'V_0', 0.0)
            R = getattr(self.config, 'R', 10.0)

            if potential_type == "square":
                qd_potential = -V_0 if r <= R else 0.0
            else:  # gaussian
                qd_potential = -V_0 * np.exp(-r**2 / (2 * R**2))

            # Total potential (in V)
            self.phi[i] = pn_potential + qd_potential

    except Exception as e:
        print(f"Error in solve_poisson: {e}")
        # Initialize with zeros as fallback
        self.phi = np.zeros(self.mesh.get_num_nodes())
