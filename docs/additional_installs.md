# Installing additional software

## Prerequisites:

- Ubuntu >= 20.04
- cmake
  - `sudo apt install cmake`
- numpy
  - `pip install numpy`


## Notes

If you're installing things on a low-memory machine, cmake may fail during the build process for Astrobee or Bullet. If this happens, try building it again with a `-j1` flag

Root access is required for both the ROS install and the Astrobee install. 

## Pyenv

This is what I use to manage my virtual environments and python versions (`uv`, `conda`, and more also work)

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

This can sometimes be necessary when building Bullet from source

```
sudo apt-get install libglu1-mesa-dev freeglut3-dev mesa-common-dev
```

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

## Assimp

This is only required if you'll be modifying/converting meshes outside of Blender
```
sudo apt install libassimp-dev
sudo apt install assimp-utils
```

## GMSH

This is needed for modifying/creating tetrahedral meshes
1. Go to https://gmsh.info/ and click on the Linux download link
2. Extract the contents to `$HOME/software` or your preferred location
