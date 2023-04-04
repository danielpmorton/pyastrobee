"""Sets up the current directory as a python package for easier imports

This is used in conjunction with "pip install -e ."

TODO add specific versions to the package requirements
TODO add obj2mjcf to the dev requirements?
TODO add beautifulsoup4 back in if parsing XML/URDFs
TODO add open3d back in if visualizing anything 3D outside of pybullet

Dependencies notes (outside of the usual suspects):
- pytransform3d: Manages rotations and transformation math
- Pylint/Black: For code formatting
- pynput: Manages the keyboard listener
"""

from setuptools import setup

setup(
    name="pyastrobee",
    version="0.0.1",
    # Note on install_requires: order matters! Keep wheel and numpy before pybullet
    install_requires=[
        "numpy",
        "wheel",
        "pybullet",
        "opencv-python",
        "matplotlib",
        "pytransform3d",
        "pynput",
    ],
    extras_require={"dev": ["pylint", "black", "ipython", "mujoco"]},
    description="Code for the IPRL Astrobee project",
    author="Daniel Morton",
    author_email="danielpmorton@gmail.com",
    url="https://github.com/danielpmorton/pyastrobee",
)
