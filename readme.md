![Pyastrobee](pyastrobee/assets/imgs/banner.png)
# Pyastrobee: A space robotics simulation environment in Python

## Documentation Overview
[Getting Started](docs/getting_started.md): Setting up the repository

[Additional Installs](docs/additional_installs.md): Additional installs important for working with the project

[Loading ISS Meshes](docs/loading_iss_meshes.md): Dealing with complex meshes and textures in Pybullet

[Re-texturing](docs/retexturing.md): How to modify Astrobee meshes to load the URDF with a single texture file

[Meshing](docs/meshing.md): Tips on creating new (triangular) meshes to load into Pybullet

[Tetrahedral Meshing](docs/tet_meshing.md): How to generate and modify tetrahedral meshes for soft bodies

[Bag Dynamics](docs/bag_dynamics.md): Some notes on defining mass/inertia values of the cargo bag

[Testing](docs/testing.md): Information about running test cases

[Using NASA's simulator](docs/nasa_sim.md): Helpful commands and installation debugging

[References](docs/references.md): Links to hepful resources

## Status
### TODOs:
- [ ] Update the VTK meshes so that we can just apply `astrobee_texture` directly? Might need to regenerate them based on the OBJ file that has the correct UV map
- [ ] FIX THE ORIENTATION ISSUE IN `load_rigid_object` and `load_deformable_object`
  - [ ] They seem to use fixed angles rather than euler angles in their own converters
  - [ ] Just make the input to the functions a quaternion
  - [ ] Might need some refactoring of the demo script
- [ ] When the Astrobee is loaded it steps the sim (since it opens the gripper), which may be unwanted behavior
- [ ] Collision information for the rigid bag looks pretty weird. (in wireframe mode). Will this be an issue? Should we vhacd this? Or make something out of simple geometry
- [ ] Try out determining inertia matrices in MeshLab
- [ ] See if it's possible to import the VTKs into Blender and adjust their texture maps
- [ ] Load the Astrobee with the anchored soft bag with NO CONSTRAINT, click + drag to see if the physics seems ok
- [ ] Turn the load_bag function from the demo into something a bit more robust
- [ ] Add min-jerk and screw path trajectory
  - [ ] See the NU Modern Robotics videos
    - [ ] https://modernrobotics.northwestern.edu/nu-gm-book-resource/9-1-and-9-2-point-to-point-trajectories-part-1-of-2/
- [ ] Check on quaternion normalization meaning
- [ ] Check out `humanoidMotionCapture` in pybullet examples!!
  - [ ] Especially how they use `pdControllerStable` and `pdControllerExplicit`
- [ ] Update the pose controller to use the new quaternion heading function
- [ ] Add test cases for batched quaternion operations
- [ ] Deal with stuff from demo
  - [ ] Delete the unnecessary code
  - [ ] Make the file self-contained with Rika's code
  - [ ] Rename the file with the demo date
- [ ] Figure out how to get textures to apply properly to VTK files
- [ ] Look into changeDynamics()
- [ ] Fix the weird orientation of the ISS meshes now that we know about the OBJ orientation export issues
- [ ] Add anything from the demo to the control section, if we want to keep it
- [ ] Make an "electromagnetic" snap to a wall when the bag is in the right position
- [ ] Add a way to pause the sim when looping in interactive mode (ipython / debugger)
- [ ] Idea: add a parameter on initialization of Astrobee() deciding on the control mode? 
  - Then we could initialize the constraint if it's in position mode (and not otherwise)
  - This would really only be useful if we did decide to eliminate Controller
- [ ] Add an overview of the repo structure to the docs
- [ ] Decide if all arm/gripper control should remain in Astrobee or not
- [ ] Figure out how to export just the volume mesh from gmsh
- [ ] Get velocity control integrated into the keyboard controller
- [ ] Decide if astrobee_geom should be merged with something else
- [ ] Test out soft contact forces in an old build of Bullet (or old pybullet version in new pyenv) 
  - [ ] https://github.com/bulletphysics/bullet3/issues/4406
- [ ] Decide if we need to modify/refine the VHACD results based on what's important to us
- [ ] Set up camera (see dedo)
- [ ] Set up pointcloud (see dedo)
- [ ] Try out remeshing only half of a bag to see if a denser mesh in an area will give different properties in Bullet
- [ ] Add more tests

### References to look into
- [ ] https://github.com/rock-learning/bolero
- [ ] Check out how this guy implemented his Robot, Trajectory, and Planner classes
  - [ ] https://github.com/sahandrez/jaco_control/blob/master/jaco_control/utils/robot.py
  - [ ] https://github.com/sahandrez/jaco_control/blob/master/jaco_control/utils/trajectory.py
  - [ ] https://github.com/sahandrez/jaco_control/blob/master/jaco_control/utils/planner.py
- [ ] See if there are any useful transformations here:
  - [ ] https://github.com/cgohlke/transformations/
- [ ] Check out quaternions in numpy? https://github.com/moble/quaternion

### In Progress:
- [ ] Create trajectories
  - [ ] Make a class
  - [X] Make a way to visualize these (see pytransform trajectories)
  - [X] Visualize trajectory inside pybullet (see `addUserDebugLine` / `addUserDebugPoints`)
  - [ ] Follow trajectory using PID controller
  - [ ] Visualize tracking error
- [ ] Implement force control / velocity control
  - [ ] Improve physics models for blower, drag, ...
  - [ ] Refine step sizes / timesteps / tolerances when stepping through sim loop
  - [ ] Check out the GNC/CTL section of nasa/astrobee, try to replicate their control method? And see Keenan's papers
  - [ ] Check out `calculateMassMatrix`
  - [ ] Try making a Jacobian mapping between desired forces and torques on the robot to the generalized forces we need to apply in Pybullet (t = J.T @ F)
- [ ] Experiment with tetrahedral meshes
  - [ ] Try out tetgen/pyvista/tetwild/gmsh python interfaces
  - [ ] Try out different mesh parameters:
    - [ ] Number of mesh elements in the handle
    - [ ] Handle geometry and size
    - [ ] Main compartment geometry
    - [ ] Number of mesh elements in the main compartment
    - [ ] One side of the main compartment having a denser mesh than the other
    - [ ] Figure out relationship between number of mesh elements / size of mesh elements to stiffness
- [ ] Improve the demo script
  - [ ] Load the tet mesh instead of the tri mesh
  - [ ] Improve waypoint positions
  - [ ] Debug excessive intermediate rotations (position controller stuff)

### Backlog:
- [ ] Look into some of the other modules in pytransform3d like urdf, camera, ...
- [ ] Create a unified sim loop in Astrobee or Controller?
- [ ] Test out setRealTimeSimulation?
- [ ] Fix the `setup.py` so that it actually installs pybullet after numpy/wheel
- [ ] Decide if the *exact* positioning of the ISS needs to match Gazebo, and if so, update meshes and blender file
- [ ] See if we can use the RPBI to plan the path in ROS, then communicate the constraint info back to pybullet
- [ ] Add threading once commands get too complicated
- [ ] Send cargo bag properties to ROS
- [ ] Get Astrobee ROS/simulation processes working in Pybullet
- [ ] Figure out how to send robot state from bullet to ROS
- [ ] Make other environments loadable from astrobee_media (Granite lab, ...)
- [ ] Figure out a better way of generating the astrobee/iss URDF with less manual modifications
- [ ] Add in debugging and exception handling

### Optional
- [ ] Decide if we need to reduce the size of the Astrobee meshes at all (they're quite complex for their simple geometry). This will require a remesh of all of the parts and then another retexturing process, which might take a little while
- [ ] If there is a need for multiple bullet clients in the future, add the "sim" parameters back in from dedo

### Done:
- [X] Move info about working with the NASA ROS sim out of "Assorted Notes" and into its own page in docs
- [X] Make a URDF with a rigid cargo bag attached to the Astrobee gripper
- [X] Fix incorrect rotation motions in position controller
- [X] Calibrate that gripper / arm distal joint transformation
- [X] Fix the Astrobee textures in Blender
- [X] Move the modified Honey skin file into the correct place
- [X] Clean up all of the mesh/urdf/resources organization confusion, make assets folder, delete old meshes
- [X] Add some notes about using the keyboard controller somewhere
- [X] Get a softbody anchor working between the astrobee gripper and the bag
- [X] Try out tetrahedral deformables
- [X] Test out maxForce parameter in changeConstraint?
- [X] Update the demo script now that the control interface has moved out of Astrobee()
- [X] Get demo script for Rika
- [X] Try a manifold mesh with the handle being solid
- [X] Make the debug visualizer camera position an input in `initialize_pybullet` so we can start the visualization inside the ISS
- [X] Load the astrobee inside the new ISS model just to confirm it works ok collision-wise
- [X] Quaternion test cases (functions + class)
- [X] Rotate the ISS so it's flat
- [X] Simple motion/trajectory planning
- [X] Integrate new pose and transformation code into astrobee class
- [X] Switch over to pytransform3d for rotations/transformations
- [X] Understand which joint indices on the astrobee correspond to which locations (and which are fixed / not usable)
- [X] Figure out a way to move the astrobee in space (not via controlling a joint)
- [X] Get a feel for how pybullet controls robot links. Write helper functions to control joints
- [X] Completely remove the astrobee_media submodule and any reference to it?
- [X] Reduce the amount of hardcoded directory/file locations (especially absolute paths)
- [X] Figure out how to work with relative file paths in the urdf/xacro/xml files
- [X] Load the ISS with all textures applied
- [X] Load the astrobee URDF
- [X] Change pybullet version back to most recent in the pyenv astrobee env?
- [X] Change repo name (and name of all local folder systems) to pyastrobee
- [X] Remove all absolute filepaths, and make sure any external resources are made available in this repo so that it can work on multiple computers
- [X] Rename the new meshes folder to iss_meshes?
- [X] Get the inertial properties for the base link updated
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
- What's the deal with their crazy positioning of the ISS in Gazebo? Why is world frame attached to a random handle in the middle of space?
- What are the dimensions of the cargo bags / measurements for the handles? Do you have any CAD for these?

### Bugs/Issues:

### Reminders
