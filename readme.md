# Astrobee Pybullet/ROS Integration


## Documentation Overview
[Getting Started](docs/getting_started.md): Setting up the repository

[Loading ISS Meshes](docs/loading_iss_meshes.md): Dealing with complex meshes and textures in Pybullet

[Assorted Notes](docs/assorted_notes.md): Things I thought seemed important

## Status
### TODOs:
- [ ] Rename the new meshes folder to iss_meshes?
- [ ] Completely remove the astrobee_media submodule and any reference to it?
- [ ] Check if pyenv messes with the nasa ROS commands like the conda env did
- [ ] Clean up the debugging script and any files leftover from the `meshing` branch rebase that are no longer needed
- [ ] Set up camera (see dedo)
- [ ] Set up pointcloud (see dedo)
- [ ] Try a manifold mesh with the handle being solid
- [ ] Try out FEM deformables?
- [ ] Try out adding a small anchor object to the handle - see dedo anchor utils. Make it a small nonzero mass
- [ ] Try flipping all of the face normals on the non-manifold mesh
- [ ] Change pybullet version back to most recent in the pyenv astrobee env?
- [ ] Work on improving bag meshes
- [ ] Try out remeshing only half of a bag to see if a denser mesh in an area will give different properties in Bullet
- [ ] Figure out if it's possible to load arbitrary meshes into mujoco
- [ ] Test out `from mujoco import viewer` and `viewer.launch(...)`
- [ ] Get a texture file for astrobee and import it
- [ ] Organize meshes
- [ ] Send cargo bag properties to ROS
- [ ] Get Astrobee ROS/simulation processes working in Pybullet
- [ ] Figure out how to send robot state from bullet to ROS
- [ ] Add tests
- [ ] Figure out more TODOs

### In Progress:
- [ ] Load the ISS and astrobee with all textures applied
- [ ] See if it is possible to get URDFs working with multiple textures
- [ ] Get correct physical properties for cargo bag (check on weird inertia?)

### Backlog/Optional:
- [ ] Figure out a better way of generating the astrobee/iss URDF with less manual modifications
- [ ] Figure out how to work with relative file paths in the urdf/xacro/xml files
- [ ] If there is a need for multiple bullet clients in the future, add the "sim" parameters back in from dedo
- [ ] Add in debugging and exception handling
- [ ] Consider using pathlib Path with str(Path(filename))?
- [ ] Reduced the amount of hardcoded directory/file locations (especially absolute paths)

### Done:
- [X] Debugged weird collapsing/exploding/stuttering issues with the cargo bag meshes
- [X] Add info to readme about setting up the submodule
- [X] Import ISS modules with textures
- [X] Try loading the iss as multiple objs (with the texture files)
- [X] Solve the Cupola/Node 1 mesh issues
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
