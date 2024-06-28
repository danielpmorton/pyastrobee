"""Sets up the current directory as a python package for easier imports

This is used in conjunction with "pip install -e ."

NOTE Pybullet is a key dependency of the project, but we want to be working with the most recent version of the code,
including some custom changes, which may not be available on pypi for some time. We'll use a manually built version
of pybullet from source instead.
"""

# TODO add specific versions to the package requirements
# TODO add obj2mjcf to the dev requirements?
# TODO add beautifulsoup4 back in if parsing XML/URDFs
# TODO add open3d back in if visualizing anything 3D outside of pybullet
# TODO add mujoco back if needed
# TODO add slycot back in if the control package needs it. Seems to have installation problems

from setuptools import setup, find_packages

setup(
    name="pyastrobee",
    version="0.0.1",
    install_requires=[
        # "pybullet",  # Simulation. See notes about locally-built version
        "numpy==1.26.4",  # Needs to be installed before Pybullet to enable speedup for matrix ops
        "wheel",  # For helping build Pybullet
        "opencv-python",  # Vision
        "matplotlib",  # Plotting
        "pytransform3d",  # Manages rotations and transformations
        "pynput",  # For keyboard event listeners
        "ahrs",  # For some quaternion math and test cases
        "pylint",  # Linting
        "black",  # Formatting
        "ipython",  # Interactive sessions
        "control",  # Control systems reference code
        "cvxpy",  # Optimization
        "clarabel",  # More optimization
        "stable-baselines3",  # Environments
        "gymnasium",  # Environments
        "portray",  # Documentation
    ],
    description="Code for the IPRL Astrobee project",
    author="Daniel Morton",
    author_email="danielpmorton@gmail.com",
    url="https://github.com/danielpmorton/pyastrobee",
    packages=find_packages(exclude=["artifacts", "images"]),
)
