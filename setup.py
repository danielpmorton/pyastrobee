"""Sets up the current directory as a python package for easier imports

This is used in conjunction with "pip install -e ."

TODO add specific versions to the package requirements
TODO add obj2mjcf to the dev requirements?
TODO add beautifulsoup4 back in if parsing XML/URDFs
TODO add open3d back in if visualizing anything 3D outside of pybullet
TODO add mujoco back if needed
"""

from setuptools import setup

setup(
    name="pyastrobee",
    version="0.0.1",
    install_requires=[
        "numpy",  # Needs to be installed before Pybullet to enable speedup for matrix ops
        "wheel",  # For helping build Pybullet
        "pybullet",  # Simulation
        "opencv-python",  # Vision
        "matplotlib",  # Plotting
        "pytransform3d",  # Manages rotations and transformations
        "pynput",  # For keyboard event listeners
        "ahrs",  # For some quaternion math and test cases
        "pylint",  # Linting
        "black",  # Formatting
        "ipython",  # Interactive sessions
        "control",  # Control systems reference code
        "slycot",  # Used in conjunction with the control package
    ],
    description="Code for the IPRL Astrobee project",
    author="Daniel Morton",
    author_email="danielpmorton@gmail.com",
    url="https://github.com/danielpmorton/pyastrobee",
)
