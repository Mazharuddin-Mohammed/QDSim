from setuptools import setup, find_packages

setup(
    name="qdsim",
    version="0.1.0",
    packages=find_packages(where="frontend"),
    package_dir={"": "frontend"},
    install_requires=[
        "numpy",
        "matplotlib",
    ],
)
