Installation Guide
==================

This guide provides detailed instructions for installing QDSim on various platforms and configurations.

Quick Installation
------------------

For most users, the quickest way to get started with QDSim is:

.. code-block:: bash

    # Install system dependencies
    sudo apt-get install python3-dev python3-matplotlib python3-numpy python3-scipy
    
    # Clone and install QDSim
    git clone https://github.com/your-username/QDSim.git
    cd QDSim
    pip install -e .

System Requirements
------------------

**Minimum Requirements**
    - Python 3.8 or higher
    - NumPy 1.19.0 or higher
    - SciPy 1.5.0 or higher
    - Matplotlib 3.3.0 or higher
    - 4 GB RAM
    - 2 GB disk space

**Recommended Requirements**
    - Python 3.9 or higher
    - NumPy 1.21.0 or higher
    - SciPy 1.7.0 or higher
    - Matplotlib 3.5.0 or higher
    - 16 GB RAM or more
    - 10 GB disk space
    - NVIDIA GPU with CUDA support (optional)

**Supported Platforms**
    - Linux (Ubuntu 20.04+, CentOS 8+, Fedora 34+)
    - macOS 10.15+ (Catalina and later)
    - Windows 10/11 (with WSL2 recommended)

Installation Methods
-------------------

Method 1: Standard Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the recommended method for most users:

.. code-block:: bash

    # Update system packages
    sudo apt-get update
    
    # Install Python development tools
    sudo apt-get install python3-dev python3-pip python3-venv
    
    # Install scientific computing libraries
    sudo apt-get install python3-numpy python3-scipy python3-matplotlib
    
    # Clone QDSim repository
    git clone https://github.com/your-username/QDSim.git
    cd QDSim
    
    # Create virtual environment (recommended)
    python3 -m venv qdsim_env
    source qdsim_env/bin/activate
    
    # Install QDSim
    pip install -e .

Method 2: Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For developers and advanced users who want to modify QDSim:

.. code-block:: bash

    # Install additional development dependencies
    sudo apt-get install build-essential cmake git
    
    # Clone repository
    git clone https://github.com/your-username/QDSim.git
    cd QDSim
    
    # Create development environment
    python3 -m venv qdsim_dev_env
    source qdsim_dev_env/bin/activate
    
    # Install development dependencies
    pip install -e .[dev]
    
    # Build Cython extensions
    python setup.py build_ext --inplace

Method 3: GPU-Accelerated Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For users with NVIDIA GPUs who want GPU acceleration:

.. code-block:: bash

    # Install CUDA toolkit (Ubuntu)
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
    sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
    wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
    sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
    sudo apt-get update
    sudo apt-get -y install cuda
    
    # Install QDSim with GPU support
    git clone https://github.com/your-username/QDSim.git
    cd QDSim
    python3 -m venv qdsim_gpu_env
    source qdsim_gpu_env/bin/activate
    pip install -e .[gpu]

Platform-Specific Instructions
------------------------------

Ubuntu/Debian
~~~~~~~~~~~~~

.. code-block:: bash

    # Update package list
    sudo apt-get update
    
    # Install dependencies
    sudo apt-get install python3-dev python3-pip python3-venv
    sudo apt-get install python3-numpy python3-scipy python3-matplotlib
    sudo apt-get install build-essential cmake git
    
    # For GPU support (optional)
    sudo apt-get install nvidia-cuda-toolkit
    
    # Install QDSim
    git clone https://github.com/your-username/QDSim.git
    cd QDSim
    python3 -m venv qdsim_env
    source qdsim_env/bin/activate
    pip install -e .

CentOS/RHEL/Fedora
~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Install dependencies (CentOS/RHEL)
    sudo yum install python3-devel python3-pip
    sudo yum install python3-numpy python3-scipy python3-matplotlib
    sudo yum install gcc gcc-c++ cmake git
    
    # Or for Fedora
    sudo dnf install python3-devel python3-pip
    sudo dnf install python3-numpy python3-scipy python3-matplotlib
    sudo dnf install gcc gcc-c++ cmake git
    
    # Install QDSim
    git clone https://github.com/your-username/QDSim.git
    cd QDSim
    python3 -m venv qdsim_env
    source qdsim_env/bin/activate
    pip install -e .

macOS
~~~~~

.. code-block:: bash

    # Install Homebrew (if not already installed)
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Install dependencies
    brew install python@3.9
    brew install cmake git
    
    # Install QDSim
    git clone https://github.com/your-username/QDSim.git
    cd QDSim
    python3 -m venv qdsim_env
    source qdsim_env/bin/activate
    pip install -e .

Windows (WSL2)
~~~~~~~~~~~~~~

.. code-block:: bash

    # Install WSL2 and Ubuntu
    # (Follow Microsoft's WSL2 installation guide)
    
    # In WSL2 Ubuntu terminal:
    sudo apt-get update
    sudo apt-get install python3-dev python3-pip python3-venv
    sudo apt-get install python3-numpy python3-scipy python3-matplotlib
    
    # Install QDSim
    git clone https://github.com/your-username/QDSim.git
    cd QDSim
    python3 -m venv qdsim_env
    source qdsim_env/bin/activate
    pip install -e .

Conda Installation
------------------

For users who prefer Conda:

.. code-block:: bash

    # Create conda environment
    conda create -n qdsim python=3.9 numpy scipy matplotlib
    conda activate qdsim
    
    # Install additional dependencies
    conda install -c conda-forge cython
    
    # Install QDSim
    git clone https://github.com/your-username/QDSim.git
    cd QDSim
    pip install -e .

Verification
-----------

After installation, verify that QDSim is working correctly:

.. code-block:: python

    # Test basic import
    import qdsim
    print(f"QDSim version: {qdsim.__version__}")
    
    # Test quantum simulation
    from qdsim.solvers import FixedOpenSystemSolver
    print("✅ Quantum solvers available")
    
    # Test visualization
    from qdsim.visualization import WavefunctionPlotter
    print("✅ Visualization tools available")
    
    # Run integration test
    import subprocess
    result = subprocess.run(['python', 'working_integration_test.py'], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ Integration test passed")
    else:
        print("❌ Integration test failed")

Troubleshooting
--------------

Common Installation Issues
~~~~~~~~~~~~~~~~~~~~~~~~~

**ImportError: No module named 'qdsim'**

.. code-block:: bash

    # Ensure you're in the correct environment
    source qdsim_env/bin/activate
    
    # Reinstall in development mode
    pip install -e .

**Compilation errors with Cython**

.. code-block:: bash

    # Install development tools
    sudo apt-get install build-essential python3-dev
    
    # Update Cython
    pip install --upgrade cython
    
    # Clean and rebuild
    python setup.py clean --all
    python setup.py build_ext --inplace

**CUDA-related errors**

.. code-block:: bash

    # Check CUDA installation
    nvcc --version
    
    # Verify GPU is detected
    nvidia-smi
    
    # Install CUDA-compatible PyTorch (if using GPU features)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

**Memory errors during large simulations**

.. code-block:: python

    # Reduce problem size or use iterative solvers
    solver = qdsim.FixedOpenSystemSolver(
        nx=6, ny=5,  # Smaller grid
        use_iterative_solver=True
    )

Performance Optimization
-----------------------

For optimal performance:

1. **Use virtual environments** to avoid package conflicts
2. **Install optimized BLAS/LAPACK** libraries:

   .. code-block:: bash

       sudo apt-get install libopenblas-dev liblapack-dev

3. **Enable GPU acceleration** if available
4. **Use appropriate compiler flags** for your system
5. **Monitor memory usage** for large simulations

Getting Help
-----------

If you encounter issues during installation:

1. **Check the FAQ**: :doc:`faq`
2. **Search existing issues**: `GitHub Issues <https://github.com/your-username/QDSim/issues>`_
3. **Ask for help**: `GitHub Discussions <https://github.com/your-username/QDSim/discussions>`_
4. **Contact support**: qdsim-support@example.com

Next Steps
----------

After successful installation:

1. **Read the Quick Start Guide**: :doc:`quickstart`
2. **Try the tutorials**: :doc:`tutorials/index`
3. **Explore examples**: :doc:`examples/index`
4. **Join the community**: `GitHub Discussions <https://github.com/your-username/QDSim/discussions>`_
