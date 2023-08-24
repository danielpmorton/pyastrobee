# Getting started

*An overview of how to set up the repository, virtual environment, and pip dependencies*

## Prerequisites

Information on installing software prerequisites can be found [here](docs/../additional_installs.md)

Once these are installled, continue with the setup process

## Cloning the repo

```
cd $HOME
git clone https://github.com/danielpmorton/pyastrobee
```

## Virtual environment

A virtual environment is optional, but highly recommended. Pyenv was found to work a bit better than conda here. 

```
# pyenv install 3.10.8 (if not already installed)
pyenv virtualenv 3.10.8 astrobee
pyenv shell astrobee
```
## Install dependencies

Install all packages except for pybullet:

```
cd $HOME/pyastrobee
pip install -e .
```

Install pybullet from source:

```
export BULLET_DIR=$HOME/software/bullet3  # Can change this install location as desired
git clone https://github.com/danielpmorton/bullet3 $BULLET_DIR
cd $BULLET_DIR
git remote add upstream https://github.com/bulletphysics/bullet3
./build_cmake_pybullet_double.sh  # Locally build Bullet/Pybullet
pip install .
```

After doing this, open a python interpreter and run the following commands:

```
import pybullet
pybullet.isNumpyEnabled()
```

The `isNumpyEnabled()` line should return `1`. If not, `pip uninstall pybullet`, then make sure that numpy is installed in your current environment, and retry the pip installation seen above.


## Interacting with the Astrobee

With everything installed, confirm that things are loading properly:

- For an interactive keyboard-based demo of the Astrobee inside the ISS, run `pyastrobee/control/keyboard_controller.py`
- Alternatively, check out `pyastrobee/scripts/demo.py` for an example of how to control the Astrobee in a script
