# Meshing tips and notes

This document will refer mainly to triangular meshes (OBJ), as opposed to tetrahedral meshes (VTK). For more info on tet meshes, see [here](../docs/tet_meshing.md). 

## Steps
1. Create CAD model (in whatever software works best)
   1. If round corners are desired over a purely rectangular body, use Fusion (they have a simple freeform modeler that works well for cargo-bag-like bodies)
   2. Try to design the part so that the center of mass lies at the origin. This is because if you load a (rigid) OBJ into Pybullet, it will specify its center of mass wherever the origin of the mesh is (which may be undesirable behavior depending on the part's geometry).
2. Export the CAD body as an OBJ
   1. If there is a "mesh refinement" parameter in the export options, this should generally be at the lowest value such that no significant detail is lost
3. Import the OBJ into MeshMixer
   1. This is generally the best software for remeshing that I've found, though MeshLab theoretically has similar tools (and works on more than just Windows)
4. Remesh the OBJ to increase uniformity
   1. All of the mesh elements in the final mesh should be about the same size, and the actually geometry of the mesh should not be significantly degraded during the remeshing
   2. In general, you will want to decrease the mesh density, but in some cases (such as if you have a large rectangular face), you will need to increase the mesh density by linear subdivision of the faces first. 
   3. The handle of the cargo bag should generally be handled separately from the rest of the body. To do this, select all of the faces on the handle, remesh this, invert the selection so the rest of the bag is selected, and then remesh that
5. Export the OBJ
   1. This will also (most likely) create a MTL file with the same name. However, the MTL is fairly useless at this point, because we haven't associated a texture with the OBJ yet. We can ignore this for now
6. Load the OBJ into Blender and add a texture to the mesh
   1. This is detailed in-depth [here](../docs/retexturing.md) for the Astrobee meshes, but this general process also applies for any mesh.

## Debugging

Does the mesh...
- Seem to be collapsing/shrinking in on itself?
  - This could be an issue with self-collision. Try turning this to `False`
- Look like it's vibrating all over the place?
  - Try increasing the physics frequency - 240 Hz might be too low, so try 350 or 500 Hz
- Explode outwards when you click+drag on it?
  - Make sure you've specified the elastic parameters of the softbody

## General tips

- Sparser meshes tend to be more computationally cheap than dense meshes (intuitively, this makes sense)
- You can have non-manifold elements for an OBJ-type softbody (e.g. a part of the mesh that is not a boundary of a closed volume)
  - If you want to do this, delete any faces to create the non-manifold areas in Meshmixer, (if needed) fill any unwanted holes in Blender, and (if needed) remesh in Meshmixer again
- If you are modeling fabric, use OBJ. If you are modeling a "squishy" object, use VTK. 

## Future TODOs

We still need to understand how varying mesh parameters (such as number of mesh elements, mesh density variations, ...) and CAD parameters (handle geometry/size, main compartment geometry/size, ...) affect the inertial properties of the object in Pybullet. This info will likely be coming soon
