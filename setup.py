"""Sets up the current directory as a python package for easier imports

This is used in conjunction with "pip install -e ."

TODO add specific versions to the package requirements
"""

from setuptools import setup, find_packages

setup(
    name="astrobee_pybullet",
    version="0.0.1",
    install_requires=["numpy", "pybullet==3.1.7", "opencv-python"],
    extras_require={"dev": ["pylint", "black", "ipython"]},
    description="Code for the IPRL Astrobee project",
    author="Daniel Morton",
    author_email="danielpmorton@gmail.com",
    url="https://github.com/danielpmorton/astrobee_pybullet",
    packages=find_packages(exclude=["astrobee_media"]),
)
