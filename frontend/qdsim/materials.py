"""
Materials module for QDSim.
"""

class Material:
    """Base class for materials."""
    def __init__(self, name, epsilon_r, band_gap, electron_mass, hole_mass):
        self.name = name
        self.epsilon_r = epsilon_r
        self.band_gap = band_gap
        self.electron_mass = electron_mass
        self.hole_mass = hole_mass

class GaAs(Material):
    """GaAs material."""
    def __init__(self):
        super().__init__(
            name="GaAs",
            epsilon_r=12.9,
            band_gap=1.424,  # eV at 300K
            electron_mass=0.067,  # m0
            hole_mass=0.45  # m0
        )

class AlGaAs(Material):
    """AlGaAs material."""
    def __init__(self, x):
        """
        Initialize AlGaAs material with Al fraction x.
        
        Args:
            x: Al fraction (0 <= x <= 1)
        """
        self.x = x
        # Interpolate properties based on Al fraction
        epsilon_r = 12.9 - 2.84 * x
        band_gap = 1.424 + 1.247 * x  # eV at 300K (valid for x < 0.45)
        electron_mass = 0.067 + 0.083 * x
        hole_mass = 0.45 + 0.31 * x
        
        super().__init__(
            name=f"Al{x}Ga{1-x}As",
            epsilon_r=epsilon_r,
            band_gap=band_gap,
            electron_mass=electron_mass,
            hole_mass=hole_mass
        )

class InAs(Material):
    """InAs material."""
    def __init__(self):
        super().__init__(
            name="InAs",
            epsilon_r=15.15,
            band_gap=0.354,  # eV at 300K
            electron_mass=0.023,  # m0
            hole_mass=0.41  # m0
        )

class InP(Material):
    """InP material."""
    def __init__(self):
        super().__init__(
            name="InP",
            epsilon_r=12.5,
            band_gap=1.344,  # eV at 300K
            electron_mass=0.08,  # m0
            hole_mass=0.6  # m0
        )

class Si(Material):
    """Si material."""
    def __init__(self):
        super().__init__(
            name="Si",
            epsilon_r=11.7,
            band_gap=1.12,  # eV at 300K
            electron_mass=0.98,  # m0 (longitudinal)
            hole_mass=0.49  # m0 (light hole)
        )

class Ge(Material):
    """Ge material."""
    def __init__(self):
        super().__init__(
            name="Ge",
            epsilon_r=16.0,
            band_gap=0.66,  # eV at 300K
            electron_mass=0.55,  # m0 (longitudinal)
            hole_mass=0.37  # m0 (light hole)
        )
