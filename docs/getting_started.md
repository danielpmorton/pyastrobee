# Getting started

*An overview of how to set up the repository, virtual environment, and more*

## Cloning the repo

```
cd $HOME
git clone https://github.com/danielpmorton/astrobee_pybullet
cd astrobee_pybullet
cd astrobee_media
git submodule init
git submodule update
cd ..
```

## Pyenv

A virtual environment is optional, but recommended. Pyenv was found to work a bit better than conda here.

If pyenv is not already installed, run
```
curl https://pyenv.run | bash
```
Then, set up `~/.bashrc` -- Make sure the following lines are included
```
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

Set up the virtual environment
```
# pyenv install 3.10.8 if not already installed
pyenv virtualenv 3.10.8 astrobee
pyenv shell astrobee
```
## Install dependencies

The `[dev]` option will install additional packages for helping with developement. If you only want the minimal requirements, just run `pip install -e .`
```
pip install -e .[dev]
```

After doing this, open a python interpreter and run the following commands:
```
import pybullet
pybullet.isNumpyEnabled()
```
The `isNumpyEnabled()` line should return `1`. If not, `pip uninstall pybullet`, then make sure that numpy is installed in your current environment, and retry `pip install pybullet`

## Other stuff

To get a nice visual graph of Git history via `git graph`:
```
git config --global alias.graph "log --all --graph --decorate --oneline"
```

To make sure Ipython (via `ipython`) uses the same python version as your environment: in `~/.bashrc`, add:
```
alias ipython="python -m IPython"
```
