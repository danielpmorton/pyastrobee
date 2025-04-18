"""Sets up the current directory as a python package for easier imports

This is used in conjunction with "pip install -e ."

NOTE Pybullet is a key dependency of the project, but we want to be working with the most recent version of the code,
including some custom changes, which may not be available on pypi for some time. We'll use a manually built version
of pybullet from source instead.
"""

from setuptools import setup, find_packages

setup(
    name="pyastrobee",
    version="0.0.1",
    install_requires=[
        # "pybullet",  # Simulation. See notes about locally-built version
        "numpy<2",  # Needs to be installed before Pybullet
        "wheel",  # For helping build Pybullet
        "matplotlib",  # Plotting
        "pytransform3d",  # Manages rotations and transformations
        "pynput",  # For keyboard event listeners
        "cvxpy",  # Optimization
        "clarabel",  # More optimization
        "mosek",  # More optimization
        "stable-baselines3",  # Environments
        "gymnasium",  # Environments
    ],
    extras_require={
        "dev": [
            "pylint",  # Linting
            "black",  # Formatting
            "portray",  # Documentation
        ],
    },
    description="A simulation environment for Astrobee in Python",
    author="Daniel Morton",
    author_email="danielpmorton@gmail.com",
    url="https://github.com/danielpmorton/pyastrobee",
    packages=find_packages(exclude=["artifacts", "images"]),
)
