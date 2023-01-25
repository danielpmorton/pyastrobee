# Astrobee Pybullet/ROS Integration


## Getting started:

### Pyenv

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
### Configure the repository as a python package

The `[dev]` option will install additional packages for helping with developement. If you only want the minimal requirements, just run `pip install -e .`
```
pip install -e .[dev]
```

### Other stuff

To get a nice visual graph of Git history via `git graph`:
```
git config --global alias.graph "log --all --graph --decorate --oneline"
```

To make sure Ipython (via `ipython`) uses the same python version as your environment: in `~/.bashrc`, add:
```
alias ipython="python -m IPython"
```

## Status
### TODOs:
- [ ] Try a manifold mesh with the handle being solid
- [ ] Try loading the iss as multiple objs (with the texture files). Or, see if the texture can be included in urdf
- [ ] Try out FEM deformables?
- [ ] Try out adding a small anchor object to the handle - see dedo anchor utils. Make it a small nonzero mass
- [ ] Figure out if it's the combination of nonmanifoldity and self-collision that cauese the issues
- [ ] Try flipping all of the face normals on the non-manifold mesh
- [ ] Change pybullet version back to most recent in the pyenv astrobee env?
- [ ] Get dedo pointcloud stuff working
- [ ] Work on improving bag meshes
- [ ] Import textures - see png files from dedo as examples
- [ ] Try out remeshing only half of a bag to see if a denser mesh in an area will give different properties in Bullet
- [ ] Figure out if it's possible to load arbitrary meshes into mujoco
- [ ] Test out `from mujoco import viewer` and `viewer.launch(...)`
- [ ] Get a texture file for astrobee and import it
- [ ] Organize meshes
- [ ] Solve the Cupola/Node 1 mesh issues
- [ ] Send cargo bag properties to ROS
- [ ] Get Astrobee ROS/simulation processes working in Pybullet
- [ ] Figure out how to send robot state from bullet to ROS
- [ ] Add tests
- [ ] Figure out more TODOs

### In Progress:
- [ ] Get correct physical properties for cargo bag (check on weird inertia?)

### Backlog/Optional:
- [ ] Figure out a better way of generating the astrobee/iss URDF with less manual modifications
- [ ] Figure out how to work with relative file paths in the urdf/xacro/xml files
- [ ] Consider switching from conda to pyenv
- [ ] If there is a need for multiple bullet clients in the future, add the "sim" parameters back in from dedo
- [ ] Add in debugging and exception handling
- [ ] Consider using pathlib Path with str(Path(filename))?

### Done:
- [X] Try out pybullet 3.1.7 to see if this is more stable at all
- [X] Try out MuJoCo
- [X] Merge in Dedo utility functions
- [X] Import Dedo bag assets
- [X] Create cargo bag URDF and import into Pybullet
- [X] Simplify/improve cargo bag mesh (try a thin mesh handle?)
- [X] Set up repository and packaging
- [X] Import Astobee resources into Pybullet
- [X] Fix the ISS interior collisions with V-HACD
- [X] Model cargo bag in CAD and export mesh
- [X] Set up virtual environment

### Ask NASA:
- Does the version of Python matter for any of your Python code? e.g. 3.8 vs 3.10
- Can we get some of the cad files used to make the meshes?
- What are the dimensions of the cargo bags / measurements for the handles? Do you have any CAD for these?

### Bugs/Issues:
- The dedo duffel bag cannot load as a soft body
- The joint of the astrobee nearest to the body does not seem to be moving properly in simulation - check the URDF to see if there is an issue with how this joint is defined
- Some softbodies have very strange inertia when moving them around in the Bullet gui (likely, some parameters need to be refined)
- The `cupola.dae` and `node_1.dae` files in the `astrobee_iss` meshes cannot be loaded into Blender, whereas all of the other ISS meshes can.

## Thoughts
- Mujoco is interesting because it models the deformables as composites - for instance, a soft box is a collection of spheres/capsules wrapped up in a skin. I wonder if we can change the properties of these smaller particles inside the composite?

## General notes
- Recall that to run the NASA Gazebo/Rviz code, go to `~/astrobee`, run `source devel/setup.bash`, then `roslaunch astrobee sim.launch dds:=false robot:=sim_pub rviz:=true sviz:=true`.
  - NOTE: this does not currently work if the conda `astrobee` environment you made is active, so run `conda deactivate` before running the command. This could be an issue in the future, but for now we will put this off. 
- Meshing steps: 
  - Construction: Create CAD model -> Export as OBJ
  - Refinement: Delete any extra faces near the handle (Meshmixer) -> fill any holes (Blender) -> perform remeshing to improve uniformity (Meshmixer)
- Softbody mesh behavior debugging and notes
  - If it collapses/shrinks in on itself, this could be an issue with self-collision. Try turning this to False
  - If it looks like it's having a seizure, try increasing the frequency -- 240 Hz might be too low, so try 350 or 500
  - If the triangles blow up when you drag on them, check to make sure you've specified the elastic parameters
  - The more sparse meshes seem to perform better (this makes sense)
  - With self collision turned off, the thin handles tend to enter the inside of the bag more than the thick handles, which hold their shape a little better
  - The thick handles might represent a thick duffel bag handle better than the thin handles anyways, which act more like a thin shirt


## Loading ISS meshes into Pybullet

For each iss module, do the following:

### VISUAL
- Create an empty directory with the module name
- Import the associated DAE into Blender
  - Click on Viewport Shading in the top right corner to confirm that the textures loaded properly
- Export the module from Blender as an OBJ with the following options specified:
  - Grouping -> Object Groups
  - Grouping -> Material Groups
  - Path Mode -> Relative
  - (Ensure that the save directory is the folder for the ISS module. This should add an OBJ and MTL file)
- Run the obj2sdf tool in Bullet on the OBJ
  ```
  cd ~/software/bullet3/build_cmake/Extras/obj2sdf
  ./App_obj2sdf --fileName=PATH_TO_OBJ_FILE
  ```
  - This will populate the directory with a number of OBJ files as well as an SDF
  - (This requires Bullet C++ to be locally built first)
- Update the paths in the obj2sdf results
  - The paths at the top of the OBJ files from this output (part0, part1, ...) will be *absolute* paths, but pybullet works best with these as *relative* paths. These will need to be manually modified, but a directory search/replace shouldn't take much effort. 
- Since importing the SDF does not appear to work properly on its own (no textures load), import each obj as a visual shape
- NOT DONE. Look into visualShapeArray?

### COLLISION


### MERGED




