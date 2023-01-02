"""Sets up the current directory as a python package for easier imports

This is used in conjunction with setup.sh which will run the command "pip install -e ."
"""

from setuptools import setup

setup(
    name="astrobee_pybullet",
    version="0.0.1",
    install_requires=[],
    include_package_data=True,
    description="Code for the IPRL Astrobee project",
    author="Daniel Morton",
    author_email="danielpmorton@gmail.com",
    url="https://github.com/danielpmorton/astrobee_pybullet",
)
