# Astrobee Pybullet/ROS Integration

Run `./setup.sh` to get started

TODOs:
- [X] Set up repository and packaging
- [ ] Set up virtual environment
- [X] Import Astobee resources into Pybullet
- [ ] Solve the Cupola/Node 1 mesh issues
- [ ] Figure out a better way of generating the URDF with less manual modifications
- [ ] Debug the ISS interior collisions - V_HACD? New meshes/CAD?
- [ ] Get Astrobee ROS/simulation processes working in Pybullet
- [ ] Model cargo bag in Pybullet
- [ ] Send cargo bag properties to ROS
- [ ] Figure out more TODOs

Ask NASA:
- Can we upgrade numpy and python to newer versions?
- Can we get some of the cad files used to make the meshes?

Bugs/Issues:
- The `cupola.dae` and `node_1.dae` files in the `astrobee_iss` meshes cannot be loaded into Blender, whereas all of the other ISS meshes can.
  - Regarding this, I originally tested simulating the Astrobee in the cupola and was getting some weird collision behavior. I wonder if this is due to these meshes being messed-up somehow
