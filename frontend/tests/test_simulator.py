import pytest
from qdsim import Simulator, Config
import numpy as np

def test_simulator_initialization():
    config = Config()
    config.diode_p_material = "GaAs"
    config.qd_material = "InAs"
    sim = Simulator(config)
    assert sim.config is not None
    assert sim.mesh is not None
    assert sim.poisson is not None
    assert sim.sc_solver is not None
    assert sim.fem is not None
    assert sim.solver is not None

def test_reverse_bias():
    config = Config()
    config.V_r = 1.0
    sim = Simulator(config)
    eigenvalues, _ = sim.run(num_eigenvalues=1)
    assert np.iscomplexobj(eigenvalues)