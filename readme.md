# Astrobee Pybullet/ROS Integration

Run `./setup.sh` to get started

### TODOs:
- [ ] Merge in Dedo utility functions 
- [ ] Import Dedo bag assets
- [ ] Set up virtual environment
- [ ] Solve the Cupola/Node 1 mesh issues
- [ ] Figure out a better way of generating the astrobee/iss URDF with less manual modifications
- [ ] Figure out how to work with relative file paths in the urdf/xacro/xml files
- [ ] Send cargo bag properties to ROS
- [ ] Get Astrobee ROS/simulation processes working in Pybullet
- [ ] Figure out how to send robot state from bullet to ROS
- [ ] Figure out more TODOs

### In Progress:
- [ ] Create cargo bag URDF and import into Pybullet
- [ ] Get correct physical properties for cargo bag

### Done:
- [X] Set up repository and packaging
- [X] Import Astobee resources into Pybullet
- [X] Fix the ISS interior collisions with V-HACD
- [X] Model cargo bag in CAD and export mesh

### Ask NASA:
- Can we upgrade numpy and python to newer versions?
- Can we get some of the cad files used to make the meshes?
- What are the dimensions of the cargo bags / measurements for the handles? Do you have any CAD for these?

### Bugs/Issues:
- The `cupola.dae` and `node_1.dae` files in the `astrobee_iss` meshes cannot be loaded into Blender, whereas all of the other ISS meshes can.
  - Regarding this, I originally tested simulating the Astrobee in the cupola and was getting some weird collision behavior. I wonder if this is due to these meshes being messed-up somehow
