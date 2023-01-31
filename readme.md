# Astrobee Pybullet/ROS Integration


## Getting started:

### Cloning the repo

```
cd $HOME
git clone https://github.com/danielpmorton/astrobee_pybullet
cd astrobee_pybullet
cd astrobee_media
git submodule init
git submodule update
cd ..
```

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
- [ ] Check if pyenv messes with the nasa ROS commands like the conda env did
- [ ] Clean up the debugging script and any files leftover from the `meshing` branch rebase that are no longer needed
- [ ] Set up camera 
- [ ] Set up pointcloud
- [ ] Add info to readme about setting up the submodule
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
- [ ] Load the ISS and astrobee with all textures applied
  - [ ] Document all steps too (finish vhacd info)
- [ ] Get correct physical properties for cargo bag (check on weird inertia?)

### Backlog/Optional:
- [ ] Figure out a better way of generating the astrobee/iss URDF with less manual modifications
- [ ] Figure out how to work with relative file paths in the urdf/xacro/xml files
- [ ] If there is a need for multiple bullet clients in the future, add the "sim" parameters back in from dedo
- [ ] Add in debugging and exception handling
- [ ] Consider using pathlib Path with str(Path(filename))?
- [ ] Reduced the amount of hardcoded directory/file locations (especially absolute paths)

### Done:
- [X] Switch from conda to pyenv
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

Loading complex meshes with lots of textures into pybullet is tricky. 
- If you load an OBJ file through createVisualShape(), this will look for an MTL file in the same directory and apply it to the OBJ. However, if the OBJ has multiple bodies each with different textures, pybullet will mess this up and apply only one of the textures to every body in the OBJ.
  - Note: createVisualShapeArray sounds like it might be useful here, but it isn't - this is limited to a single texture applied to all of the visual objects too. 
- To fix this problem, we have to load the complex mesh as multiple OBJs, one OBJ corresponding to a single texture. To do this, run the `obj2sdf` executable in the `bullet3/build_cmake/Extras/obj2sdf` folder, which will then populate the directory of the reference OBJ file with the split-up meshes, named like `part*.obj`
- Once we have these, we can start loading these into pybullet. We now have some number n different obj files which represent the visual components of the ISS module, but we only have 1 VHACD file for the module. Pybullet needs 1 collision object for each visual object (if we want to get the textures to load properly), so we need to do some hacky stuff to get this to work.
- The solution to this is to create "dummy" invisible objects outside of the ISS workspace area - these objects will form the collision body requirement, but won't affect the simulation at all other than allowing the visuals to be seen properly.

A full workflow of the steps required to go from the NASA-provided DAE meshes to a correct pybullet environment can be found below:

For each iss module, do the following:

### Visual
- Create an empty directory with the module name
- Import the associated DAE into Blender
  - Click on Viewport Shading in the top right corner to confirm that the textures loaded properly
- Export the module from Blender as an OBJ with the following options specified:
  - Grouping -> Object Groups
  - Grouping -> Material Groups
  - Path Mode -> Relative
  - (Ensure that the save directory is the folder for the ISS module. This should add an OBJ and MTL file)
- Run the `obj2sdf` tool in Bullet on the OBJ
  ```
  cd ~/software/bullet3/build_cmake/Extras/obj2sdf
  ./App_obj2sdf --fileName=PATH_TO_OBJ_FILE
  ```
  - This will populate the directory with a number of OBJ files as well as an SDF
  - (This requires Bullet C++ to be locally built first)
- Update the paths in the obj2sdf results
  - The paths at the top of the OBJ files from this output (part0, part1, ...) will be *absolute* paths, but pybullet works best with these as *relative* paths. These will need to be manually modified, but a directory search/replace shouldn't take much effort. 

### Collision
- If we just import an ISS module into pybullet, the collision body will not be the mesh itself, but rather the convex hull of the entire body (which is not useful for us, since we need to go inside the ISS)
- Running VHACD is the solution here - this will give us a decomposition of the module into multiple convex hulls and allow for it to be hollow. 
- Note: running `obj2sdf` in the previous section gave us multiple OBJs to work with, but NASA split these up in a weird way, so running VHACD on each individual one would be overly complex. VHACD should be run on just the main OBJ which contains the full mesh for the module.
- To run this on an OBJ file (for example, `us_lab.obj`):
  ```
  cd $HOME/software/v-hacd-4.1.0/app/build
  ./TestVHACD /home/dan/astrobee_pybullet/astrobee_media/astrobee_iss/meshes/us_lab/us_lab.obj
  ```
- This will save two files inside the `$HOME/software/v-hacd-4.1.0/app/build` folder: `decomp.obj` and `decomp.stl`. We only need the OBJ file, so rename this (for example, `vhacd_us_lab.obj`) and then move it into the correct folder (for example, `$HOME/astrobee_pybullet/astrobee_media/astrobee_iss/meshes/us_lab`)
- To double-check that things look correct, import the OBJ into MeshLab just to visualize the results. If it looks like it doesn't match up the source mesh, try changing some of the additional parameters mentioned in the [vhacd readme](https://github.com/kmammou/v-hacd). Note: the default parameters seemed to work fine for a first attempt
- Additional step: the VHACD result will append the object's path to the name of the objects in the OBJ file, and this isn't totally needed, so remove the path from the file.
  - For example, the line in the output OBJ `o /home/dan/astrobee_pybullet/astrobee_media/astrobee_iss/meshes/cupola/cupola000` can be simplified to just `cupola000`

### Merging and Importing
- Since importing the SDF does not appear to work properly on its own (no textures load), each obj will need to be loaded individually.
- To load this (see the explanation in the above section for the reasoning), 
  - For the first OBJ file, load it as a visual shape. Pybullet will see the associated MTL file in the same directory and apply the texture. For the collision body, load the full VHACD result for the entire module.
  - For each other OBJ file, load the visual shape as before, but set the collision body to a "dummy" object outside the bounds of the ISS workspace

The ISS module should be fully loaded at this point. Repeat for each other module. 


