# Working with tetrahedral meshes in Pybullet

Loading OBJ files for deformables is great for objects that are "floppy" or "thin" in nature, such as a tote bag, but not so good for objects that are more "squishy", like a duffel bag full of clothes. This is because files such as `.OBJ` or `.STL` are surface (triangular) meshes, so they have no volumetric information to them. 

So, instead we'll need to use a tetrahedral mesh in `.VTK` format

## Using GMSH

1. Open the GMSH GUI (`./gmsh` within the `bin/` directory of your GMSH install)
2. File -> Open -> Click on the STL file to load
3. Geometry -> Elementary entities -> Add -> Volume
4. When prompted with "Select volume boundary", click on the mesh
5. If a popup mentions "A scripting command is going to be appended to a non-geo file", click on "Create new .geo file"
6. When prompted with "Select hole boundaries", press 'e' since there are no holes in the cargo bag
7. Save the file
8. File -> Export -> Specify the filename with a .VTK extension. If a VTK options window shows up, keep it in ASCII format. The "save all elements" checkbox does not seem to make a difference

This `.VTK` file will look something like:
```
# vtk DataFile Version 2.0
bag_remesh, Created by Gmsh 4.11.1 
ASCII
DATASET UNSTRUCTURED_GRID
POINTS 518 double
0.125 -0.125 -0.0250000003725
0.107393302023 -0.139757603407 -0.0250000003725
...
0.06279948249067006 -0.07503616573439448 0.09226952087427626

CELLS 2577 12107
3 0 1 2
3 0 37 1
... 
3 384 385 382
4 308 427 121 442
4 188 405 186 422
...
4 424 468 357 481

CELL_TYPES 2577
5
5
...
5
10
10
...
10
```

## Modifying the VTK

Pybullet will not be able to load this file right now, because it is a combination of surface (triangle) meshes and volume (tetrahedral) meshes. But, it is not too difficult to modify this file to remove the surface mesh and just leave behind the tet mesh. 

Before this, we need to understand the file format:
- The line starting with `CELLS` has two values corresponding to
  - `NUM_CELLS`: The total number of (surface + volume) cells in the mesh
  - `LIST_SIZE`: The number of values in the `CELLS` list. This is equal to `4*NUM_TRIS + 5*NUM_TETS`
- Each line in the `CELLS` list starting with `3` corresponds to a triangular cell, and any line starting with `4` is a tetrahedral cell.
- The `CELL_TYPES` line contains the same value (`NUM_CELLS`) as was seen in `CELLS`
- Each line in `CELL_TYPES` is a `5` for a triangular cell and `10` for a tetrahedral cell.


To delete the surface (triangular) mesh, 
1. Count the number of triangular cells (`NUM_TRIS`)
2. Update the counts:
   1. `NUM_CELLS -= NUM_TRIS`
   2. `LIST_SIZE -= 4*NUM_TRIS`
3. Delete the lines in `CELLS` starting with `3`, and all `5`s in the `CELL_TYPES`
4. Update the `CELLS` and `CELL_TYPES` lines with the new counts

After these modifications, Pybullet should be able to successfully import the mesh.

## Future TODOs
Apparently, there is a way to export just the tetrahedral mesh elements via Physical groups -- see [this thread](http://onelab.info/pipermail/gmsh/2012/007253.html) for info. For now though, this manual modification process works.
