# Astrobee Pybullet/ROS Integration

### Getting started

```
conda create --name astrobee # Optional
conda activate astrobee # Optional
pip install -e .
```

### TODOs:
- [ ] Try out MuJoCo just to see what it can do
- [ ] Try out pybullet 3.1.7 to see if this is more stable at all
- [ ] Solve the Cupola/Node 1 mesh issues
- [ ] Figure out a better way of generating the astrobee/iss URDF with less manual modifications
- [ ] Figure out how to work with relative file paths in the urdf/xacro/xml files
- [ ] Send cargo bag properties to ROS
- [ ] Get Astrobee ROS/simulation processes working in Pybullet
- [ ] Figure out how to send robot state from bullet to ROS
- [ ] Figure out more TODOs

### In Progress:
- [ ] Simplify/improve cargo bag mesh (try a thin mesh handle?)
- [ ] Merge in Dedo utility functions
- [ ] Import Dedo bag assets (remember to try the duffel bag attachment in Slack)
- [ ] Create cargo bag URDF and import into Pybullet
- [ ] Get correct physical properties for cargo bag

### Done:
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
- The `cupola.dae` and `node_1.dae` files in the `astrobee_iss` meshes cannot be loaded into Blender, whereas all of the other ISS meshes can.
  - Regarding this, I originally tested simulating the Astrobee in the cupola and was getting some weird collision behavior. I wonder if this is due to these meshes being messed-up somehow
