# Contributing to QDSim

Thank you for your interest in contributing to QDSim! This guide will help you get started with contributing to our advanced quantum dot simulator.

## 🚀 Quick Start for Contributors

### 1. Development Environment Setup

```bash
# Fork and clone the repository
git clone https://github.com/your-username/QDSim.git
cd QDSim

# Create development environment
python3 -m venv qdsim_dev_env
source qdsim_dev_env/bin/activate

# Install development dependencies
pip install -e .[dev]

# Install system dependencies
sudo apt-get install python3-dev python3-matplotlib python3-numpy python3-scipy
```

### 2. Build and Test

```bash
# Build Cython extensions
python setup.py build_ext --inplace

# Run tests
python -m pytest tests/ -v

# Run working examples
python working_integration_test.py
```

## 🎯 Types of Contributions

### **High Priority Areas**
- **Open System Physics**: Complex eigenvalue implementations
- **Visualization**: Advanced plotting and analysis tools
- **Performance**: GPU acceleration and memory optimization
- **Documentation**: Theory guides and tutorials

### **Code Contributions**
- Bug fixes and performance improvements
- New quantum physics features
- Enhanced visualization capabilities
- Better error handling and validation

### **Documentation Contributions**
- Theory explanations and derivations
- Tutorial examples and use cases
- API documentation improvements
- Scientific background materials

### **Testing Contributions**
- Unit tests for new features
- Integration tests for complex workflows
- Performance benchmarks
- Cross-platform validation

## 🔬 Scientific Standards

### **Physics Accuracy**
- All quantum mechanical implementations must be physically correct
- Include references to scientific literature
- Validate against analytical solutions where possible
- Document assumptions and approximations

### **Numerical Methods**
- Use stable and well-tested algorithms
- Include error analysis and convergence studies
- Document numerical parameters and their effects
- Provide performance benchmarks

## 💻 Development Guidelines

### **Code Style**

#### Python Code
```python
# Follow PEP 8 with these specifics:
# - 4 spaces for indentation
# - Line length: 88 characters (Black formatter)
# - Type hints for all public functions
# - Comprehensive docstrings

def solve_schrodinger_equation(
    hamiltonian: np.ndarray,
    num_states: int = 5,
    solver_type: str = "arpack"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve the time-independent Schrödinger equation.
    
    Args:
        hamiltonian: Complex Hamiltonian matrix
        num_states: Number of eigenvalues to compute
        solver_type: Eigenvalue solver algorithm
        
    Returns:
        Tuple of (eigenvalues, eigenvectors)
        
    Raises:
        ValueError: If hamiltonian is not square matrix
    """
    pass
```

#### Cython Code
```cython
# Use .pyx extension for Cython files
# Include proper memory management
# Add type declarations for performance

cdef class QuantumSolver:
    cdef:
        double complex[:, :] hamiltonian
        int num_states
        
    def __init__(self, hamiltonian, num_states=5):
        self.hamiltonian = hamiltonian
        self.num_states = num_states
```

### **Testing Requirements**

#### Unit Tests
```python
import pytest
import numpy as np
from qdsim.solvers import FixedOpenSystemSolver

def test_complex_eigenvalues():
    """Test that open system produces complex eigenvalues."""
    # Setup
    solver = create_test_solver()
    
    # Execute
    eigenvals, _ = solver.solve(3)
    
    # Verify
    assert len(eigenvals) == 3
    assert any(np.imag(E) != 0 for E in eigenvals)
    
    # Check physical constraints
    for E in eigenvals:
        assert np.real(E) < 0  # Bound states
        assert np.imag(E) <= 0  # Finite lifetimes
```

#### Integration Tests
```python
def test_full_quantum_simulation():
    """Test complete quantum simulation workflow."""
    # Test realistic quantum dot simulation
    # Verify complex eigenvalues with finite lifetimes
    # Check visualization output
    # Validate against known results
```

### **Documentation Standards**

#### Theory Documentation
- Include mathematical derivations
- Reference scientific literature
- Explain physical assumptions
- Provide validation examples

#### API Documentation
```python
class WavefunctionPlotter:
    """
    Advanced visualization for quantum wavefunctions.
    
    This class provides comprehensive plotting capabilities for quantum
    mechanical systems, including energy level diagrams, 2D/3D wavefunction
    plots, and device structure visualization.
    
    Examples:
        >>> plotter = WavefunctionPlotter()
        >>> plotter.plot_energy_levels(eigenvals, "Energy Levels")
        >>> plotter.plot_wavefunction_2d(x, y, psi, "Ground State")
    
    References:
        [1] Griffiths, D. J. (2018). Introduction to Quantum Mechanics.
        [2] Tannor, D. J. (2007). Introduction to Quantum Mechanics.
    """
```

## 🧪 Testing Your Contributions

### **Required Tests**
1. **Unit Tests**: Test individual functions and classes
2. **Integration Tests**: Test complete workflows
3. **Physics Validation**: Compare with analytical solutions
4. **Performance Tests**: Ensure no significant regressions

### **Running the Test Suite**
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/unit/ -v
python -m pytest tests/integration/ -v
python -m pytest tests/physics/ -v

# Run with coverage
python -m pytest tests/ --cov=qdsim --cov-report=html
```

### **Validation Examples**
```bash
# Test complex eigenvalue implementation
python working_complex_eigenvalue_example.py

# Test visualization system
python working_integration_test.py

# Test specific solvers
python validate_actual_solvers_working.py
```

## 📝 Pull Request Process

### **Before Submitting**
1. ✅ All tests pass
2. ✅ Code follows style guidelines
3. ✅ Documentation is updated
4. ✅ Physics validation completed
5. ✅ Performance impact assessed

### **Pull Request Template**
```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation update
- [ ] Breaking change

## Physics Validation
- [ ] Compared with analytical solutions
- [ ] Validated against literature
- [ ] Tested edge cases

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Performance benchmarks run

## Documentation
- [ ] API documentation updated
- [ ] Theory documentation added
- [ ] Examples provided
```

## 🐛 Reporting Issues

### **Bug Reports**
Include:
- Clear description of the problem
- Minimal reproducible example
- Expected vs. actual behavior
- System information (OS, Python version, dependencies)
- Error messages and stack traces

### **Feature Requests**
Include:
- Scientific motivation
- Proposed implementation approach
- References to relevant literature
- Potential impact on existing code

## 📚 Documentation Contributions

### **Theory Documentation**
- Mathematical derivations with proper notation
- Physical explanations and intuition
- References to scientific literature
- Validation against known results

### **Tutorial Examples**
- Step-by-step explanations
- Complete working code
- Physical interpretation of results
- Common pitfalls and solutions

## 🏆 Recognition

Contributors will be acknowledged in:
- README.md contributors section
- Documentation acknowledgments
- Release notes for significant contributions
- Academic publications (for major scientific contributions)

## 📞 Getting Help

- **GitHub Discussions**: For questions and design discussions
- **GitHub Issues**: For bug reports and feature requests
- **Documentation**: [qdsim.readthedocs.io](https://qdsim.readthedocs.io)
- **Email**: qdsim-dev@example.com

## 📄 License

By contributing to QDSim, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to QDSim and advancing open-source quantum simulation!**
