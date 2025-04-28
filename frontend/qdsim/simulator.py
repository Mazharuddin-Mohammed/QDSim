import qdsim
import numpy as np
from .config import Config

class Simulator:
    def __init__(self, config: Config):
        self.config = config
        self.db = qdsim.MaterialDatabase()
        self.mesh = qdsim.Mesh(config.Lx, config.Ly, config.nx, config.ny, config.element_order)
        self.poisson = qdsim.PoissonSolver(self.mesh, self.epsilon_r, self.charge_density)
        self.poisson.solve(0.0, self.built_in_potential() + config.V_r)
        self.fem = qdsim.FEMSolver(self.mesh, self.effective_mass, self.potential, self.cap,
                                   self.poisson, config.element_order, config.use_mpi)
        self.solver = qdsim.EigenSolver(self.fem)
        self.normalizer = qdsim.Normalizer(self.mesh, self.db.get_material(config.matrix_material)[0],
                                           config.hbar)

    def built_in_potential(self):
        kT = 8.617e-5 * 300  # eV at 300K
        n_i = 1e16  # Intrinsic carrier concentration (m^-3, approximate)
        return kT * np.log(self.config.N_A * self.config.N_D / n_i**2) / 1.602e-19

    def depletion_width(self):
        p_mat = self.db.get_material(self.config.diode_p_material)
        epsilon_r = p_mat[4]  # epsilon_r
        epsilon_0 = 8.854e-12  # F/m
        q = 1.602e-19  # C
        V_bi = self.built_in_potential()
        return np.sqrt(2 * epsilon_r * epsilon_0 * (V_bi + self.config.V_r) /
                       (q * (self.config.N_A * self.config.N_D / (self.config.N_A + self.config.N_D))))

    def effective_mass(self, x, y):
        qd_mat = self.db.get_material(self.config.qd_material)
        matrix_mat = self.db.get_material(self.config.matrix_material)
        return qdsim.effective_mass(x, y, qd_mat, matrix_mat, self.config.R)

    def potential(self, x, y):
        qd_mat = self.db.get_material(self.config.qd_material)
        matrix_mat = self.db.get_material(self.config.matrix_material)
        return qdsim.potential(x, y, qd_mat, matrix_mat, self.config.R,
                               self.config.potential_type, self.poisson.get_potential())

    def epsilon_r(self, x, y):
        p_mat = self.db.get_material(self.config.diode_p_material)
        n_mat = self.db.get_material(self.config.diode_n_material)
        return qdsim.epsilon_r(x, y, p_mat, n_mat)

    def charge_density(self, x, y):
        return qdsim.charge_density(x, y, self.config.N_A, self.config.N_D, self.depletion_width())

    def cap(self, x, y):
        return qdsim.cap(x, y, self.config.eta, self.config.Lx, self.config.Ly, self.config.Lx / 10)

    def run(self, num_eigenvalues, max_refinements=None, threshold=None, cache_dir=None):
        max_refinements = max_refinements if max_refinements is not None else self.config.max_refinements
        threshold = threshold if threshold is not None else self.config.adaptive_threshold
        cache_dir = cache_dir if cache_dir is not None else self.config.cache_dir

        eigenvalues, eigenvectors = None, []
        for i in range(max_refinements):
            self.poisson.solve(0.0, self.built_in_potential() + self.config.V_r)
            self.fem.assemble_matrices()
            self.solver.solve(num_eigenvalues)
            eigenvalues = self.solver.get_eigenvalues()
            eigenvectors = self.solver.get_eigenvectors()
            self.fem.adapt_mesh(eigenvectors[0], threshold, cache_dir)
        normalized_eigenvectors = [self.normalizer.delta_normalize(ev, e) for e, ev in zip(eigenvalues, eigenvectors)]
        return eigenvalues, normalized_eigenvectors