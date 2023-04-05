# Installing additional software

## Prerequisites:
- Ubuntu 20.04
  - WSL2 seems to work for some of the general pybullet functionality but the full Gazebo/ROS setup with NASA's astrobee code is (currently) untested
- No non-Noetic versions of ROS installed
- cmake
  - `sudo apt install cmake`
- numpy
  - `pip install numpy`


## Notes

If you're installing things on a low-memory machine, cmake may fail during the build process for Astrobee or Bullet. If this happens, try building it again with a `-j1` flag

Root access is required for both the ROS install and the Astrobee install. 

## Pyenv

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

If the Pyenv Python install fails and warns about things not being installed, run this command to make sure the dependencies are up to date. (Then, retry the command that failed)
```
sudo apt-get update; sudo apt-get install --no-install-recommends make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
```

## ROS Noetic

```
wget -c https://raw.githubusercontent.com/qboticslabs/ros_install_noetic/master/ros_install_noetic.sh && chmod +x ./ros_install_noetic.sh && ./ros_install_noetic.sh
```
Refer to http://wiki.ros.org/noetic/Installation/Ubuntu for more info

## OpenGL

```
sudo apt-get install libglu1-mesa-dev freeglut3-dev mesa-common-dev
```

## Bullet
Make sure numpy and OpenGL are installed first!
```
cd $HOME/software
git clone https://github.com/bulletphysics/bullet3
cd bullet3
./build_cmake_pybullet_double.sh
```

## Astrobee

This can be an extensive process with a lot of potential issues that may come up. There is a lot more info in a separate page [here](../docs/nasa_sim.md).

## Blender

```
sudo snap install blender --classic
```

If snap is not available, 
1. Go to https://www.blender.org/download/
2. Download the Linux file
3. Extract the file to `$HOME/software`
4. Run via the `blender` executable file in the blender folder (`chmod u+x blender` if it's not executable)


## V-HACD

```
cd $HOME/software
git clone https://github.com/kmammou/v-hacd
cd v-hacd/app
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Meshlab

1. Go to https://www.meshlab.net/#download
2. Download the Linux AppImage
3. Move the file to `$HOME/software`
4. `chmod u+x` the AppImage to make it executable


## Mujoco

Download the source code (mostly just to use as a reference for now):

```
cd $HOME/software
git clone https://github.com/deepmind/mujoco
```

Get the pre-built binaries:

1. Download the latest `linux-x86_64.tar.gz` file from  https://github.com/deepmind/mujoco/releases
2. Extract the contents to `$HOME/software`


## Assimp

This is only required if you'll be modifying/converting meshes outside of Blender
```
sudo apt install libassimp-dev
sudo apt install assimp-utils
```

## GMSH

This is needed for modifying/creating tetrahedral meshes
1. Go to https://gmsh.info/ and click on the Linux download link
2. Extract the contents to `$HOME/software`

## VSCode

```
sudo snap install --classic code
```

If snap is not available:
```
sudo apt-get install wget gpg
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
sudo install -D -o root -g root -m 644 packages.microsoft.gpg /etc/apt/keyrings/packages.microsoft.gpg
sudo sh -c 'echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list'
rm -f packages.microsoft.gpg

sudo apt install apt-transport-https
sudo apt update
sudo apt install code
```

Refer to https://code.visualstudio.com/docs/setup/linux for more details and alternate install methods

## Other miscellaneous tweaks

To get a nice visual graph of Git history via `git graph`:
```
git config --global alias.graph "log --all --graph --decorate --oneline"
```

To make sure Ipython (via `ipython`) uses the same python version as your environment: in `~/.bashrc`, add:
```
alias ipython="python -m IPython"
```
