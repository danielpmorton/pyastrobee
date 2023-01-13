# Astrobee Pybullet/ROS Integration

## Getting started
```
conda create --name astrobee # Optional
conda activate astrobee # Optional
pip install -e .[dev]
```
- A conda environment is optional but recommended
- The `[dev]` option will install additional packages for helping with developement. If you only want the minimal requirements, just run `pip install -e .`
- Recommended git trick: `git config --global alias.graph "log --all --graph --decorate --oneline"`, then `git graph` will work nicely

## Status
### TODOs:
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
- [ ] Try out pybullet 3.1.7 to see if this is more stable at all
- [ ] Figure out a better way of generating the astrobee/iss URDF with less manual modifications
- [ ] Figure out how to work with relative file paths in the urdf/xacro/xml files
- [ ] Consider switching from conda to pyenv
- [ ] If there is a need for multiple bullet clients in the future, add the "sim" parameters back in from dedo
- [ ] Add in debugging and exception handling
- [ ] Consider using pathlib Path with str(Path(filename))?

### Done:
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
