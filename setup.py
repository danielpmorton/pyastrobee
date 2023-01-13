"""Sets up the current directory as a python package for easier imports

This is used in conjunction with "pip install -e ."

TODO add specific versions to the package requirements
"""

from setuptools import setup

setup(
    name="astrobee_pybullet",
    version="0.0.1",
    install_requires=["numpy", "pybullet", "opencv-python"],
    extras_require={"dev": ["pylint", "black"]},
    description="Code for the IPRL Astrobee project",
    author="Daniel Morton",
    author_email="danielpmorton@gmail.com",
    url="https://github.com/danielpmorton/astrobee_pybullet",
)
