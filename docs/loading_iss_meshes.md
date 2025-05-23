# Loading ISS meshes into Pybullet

*Dealing with Pybullet's quirks*

Loading complex meshes with lots of textures into pybullet is tricky. Ideally, we'd be able to just load a URDF and call it a day, but that just doesn't seem to be possible if you want all of the textures to appear. So, we have to resort to loading multiple OBJs. 

If you load an OBJ file through `createVisualShape()`, this will look for an MTL file in the same directory and apply it to the OBJ. However, if the OBJ has multiple bodies each with different textures, pybullet will mess this up and apply only one of the textures to every body in the OBJ. To fix this problem, we have to separate the complex mesh into multiple OBJs, with each OBJ corresponding to a single texture. We can use Erwin's `obj2sdf` tool for this, even if we're not using an SDF. 

For the collision side of things, if we just import an ISS module into pybullet, the collision body will not be the mesh itself, but rather the convex hull of the entire body (which is not useful for us, since we need to go inside the ISS). Running VHACD is the solution here - this will give us a decomposition of the module into multiple convex hulls and allow for it to be hollow. VHACD should be run on each module, not each part within a module, because each part is complex and oddly shaped (for instance, all of the handles in a module are it's own OBJ, since all handles have the same texture)

Once we have the OBJs for each texture, and the VHACD OBJ for collision info, we can start loading these into pybullet. We now have some number n different obj files which represent the visual components of the ISS module, but we only have 1 VHACD file for the module. So, we'll load the first visual component paired with the VHACD collision information, and for the remainder of the bodies, we'll load them strictly as visual elements by setting the collision ID in the multibody as -1. Note: it does not appear to be possible to create an "invisible" collision body, because if you set the visual ID in the multibody as -1, it will still visualize the collision mesh but in a bunch of random colors. 

A full workflow of the steps required to go from the NASA-provided DAE meshes to a correct pybullet environment can be found below:

## 0. Folder structure

This is what the directory will look like at the end of the process:

```
meshes
├── dae
│   ├── cupola
│   │   ├── *.png # From Blender DAE export
│   │   └── cupola.dae # From Blender DAE export
│   ├── eu_lab
│   ├── iss
│   ├── jpm
│   ├── node_1
│   ├── node_2
│   ├── node_3
│   └── us_lab
└── obj
    ├── cupola
    │   ├── cupola.mtl # From Blender OBJ export
    │   ├── cupola.obj # From Blender OBJ export
    │   ├── decomp.mtl # From vhacd: Unused
    │   ├── decomp.obj # From vhacd: Collision body
    │   ├── decomp.stl # From vhacd: Unused
    │   ├── newsdf.sdf # From obj2sdf: Unused
    │   └── part*.obj # From obj2sdf: One obj per texture
    ├── eu_lab
    ├── iss
    ├── jpm
    ├── node_1
    ├── node_2
    ├── node_3
    ├── us_lab
    └── textures
        └── *.png # From astrobee_media textures folder
```

## 1. Fix problematic texture images

Certain texture PNG images are unable to be loaded into pybullet. After some debugging, this appears to be an issue with the PNG filetype - some were exported as PNG8 whereas others are PNG16/24. Pybullet seems to need PNGs to be 8-bit, so these need to be converted. 

To do so, import each problematic PNG into Gimp and then export/overwrite the image to the same location, with the `pixelformat` dropdown set to `8bpc RGBA`

Problematic images:

- `Generic_Intersection.png`
- `Node2_Interior_Racks.png`
- `Node2_Bulkheads.png`


## 2. Blender

The meshes for the ISS were loaded in crazy locations and orientations because NASA didn't do the best job of exporting them in the correct positions when they made the files. (This is why in the URDF, each module has its own position, orientation, and even scale). So, I just re-organized all of the components in Blender and then exported them with the correct properties. 

- Open blender
- Import each dae file from the `astrobee_media` repo
- Position each component to match the true ISS layout - confirm this against the ROS/Gazebo result
- Make sure everything is deselected, then export the full ISS DAE and OBJ
- Click to select each module, and for each one, export the DAE and the OBJ
  - If exporting DAE, select the following options:
    - Selection Only
    - Include Children
  - If exporting OBJ, select the following options:
    - Grouping -> Material Groups

This will add `(module).obj` and `(module).mtl` into the `obj/(module)` folder. The `dae` folder is a backup in case this is needed later.

## (2.5) A script for running both OBJ2SDF and VHACD

Prior to running steps 3 and 4, consider using the script at `pyastrobee/scripts/manage_objs.sh`, which will loop through the directory structure and automatically run these commands on the files. The only requirement is that the directory containing the OBJs is set up so each folder corresponds to an ISS module, containing that module's multi-texture OBJ inside it (with the same name). See the Folder Structure section above for reference. 

If you use this script, and it runs without issue, you can safely skip steps 3 and 4. 

## 3. OBJ2SDF

(Ensure that the C++ version of Bullet is locally built first)

- For each module OBJ in its respective folder (i.e. `obj/(module)/(module).obj`), run `obj2sdf` as follows:
- For example: 
    ```
    cd ~/pyastrobee/pyastrobee/assets/meshes/iss/obj/cupola
    ./../../../software/bullet3/build_cmake/Extras/obj2sdf/App_obj2sdf --fileName="cupola.obj"
    ```

This will add `newsdf.sdf` and multiple `part*.obj` files, one per texture, into the `obj/(module)` folder.

## 4. VHACD

- For each module OBJ in its respective folder (i.e. `obj/(module)/(module).obj`), run `VHACD` as follows:
- For example: 
    ```
    cd ~/pyastrobee/pyastrobee/assets/meshes/iss/obj/cupola
    ./../../../software/v-hacd-4.1.0/app/build/TestVHACD cupola.obj
    ```

This will add `decomp.obj`, `decomp.mtl`, and `decomp.stl` to the `obj/(module)` folder.

## 5. Final adjustments

Refine VHACD
- The VHACD output will most likely *not* be perfect. There seems to be no exact combination of parameters that leads to an exact decomposition of the ISS (see the [vhacd readme](https://github.com/kmammou/v-hacd) for more info on these). In particular, VHACD tends to poorly represent the corridors between the modules, leading to a very tight pathway. So, to fix this, open the `decomp.obj` VHACD result for each module in Blender and update the file manually (as in, replace any "bad" convex hulls with simple objects in Blender, like rectangular prisms).

Update the paths

- The paths at the top of the OBJ/MTL files from this output (`part0`, `part1`, ...) will be *absolute* paths, but pybullet works best with these as *relative* paths. These will need to be manually modified with a quick directory search/replace. For MTL files especially, ensure that these are pointing to files inside the `obj/textures` directory

## 6. Importing into Pybullet

For each module, 

- For the first OBJ file, load it as a visual shape. Pybullet will see the associated MTL file in the same directory and apply the texture. For the collision body, load the full VHACD result for the entire module.
- For each other OBJ file, load the visual shape as before, but set the collision body to `-1`, so that it doesn't load any collision info

The ISS should be fully loaded at this point! 😃
