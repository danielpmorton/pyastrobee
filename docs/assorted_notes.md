# Assorted notes

*Random things that I knew I should write down*

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
- Kevin Zakka has a nice OBJ -> MJCF converter that performs all of the steps I did, automatically, but for MJCF rather than URDF
  -  `obj2mjcf --obj-dir PATH_TO_OBJ_DIR --save-mjcf --save_mtl --vhacd-args.enable`
  -  This will split the OBJ into multiple components, make new mtl files, and then parse it into an XML file for mujoco
  -  This seems to work quite well when importing the files - tested on the cupola module. 
- createVisualShapeArray sounds like it would be useful for loading multiple textured objects, but it isn't - this is limited to a single texture applied to all of the visual objects. 
- If you load a URDF and it comes out looking entirely black, but you know it should have a texture applied, add the following block inside the `<visual>` block, after `<geometry>`: 
    ```
    <material name="default_material">
        <color rgba="1 1 1 1"/>
    </material>
    ```

## Notes 1/31 (TODO remove/debug)
- To load an obj with multiple visuals and one collision, we need to do the weird thing with the dummy objects
- ideally, we'd just be able to load a urdf for the iss
- loading the provided urdf doesn't work because of the textures issue, but at least all of the objects are in the correct places
- to get the textures working in a urdf, each of the decomposed parts (corresponding to a single texture each) needs to be specified as a visual geometry in the urdf
- Also, if you don't specify a material with an rgba value of 1 1 1 1, the module will show up totally black, so this needs to be added too
- If this is done for all of the parts in a module, once pybullet loads the module, it's no longer in the correct position anymore :(
- Additionally, when this was tested on multiple modules (cupola and eu_lab), the eu_lab for some reason got its textures all messed up, and was also in the wrong location
- If we load the objects via the hacky method with the anchors, this will be fine for the ISS, because the ISS is fixed and we don't really care where these dummies are. However, we can't do this for the astrobee, since if we stick a dummy object somewhere, this will interfere with the collision because it's moving
- We can likely get away with no textures on the astrobee because it doesn't really need to see itself - this would just be to make the simulation look a little prettier
- If we move forward with the hacky method, we'll probably need to specify positions of all of the modules (get this from the URDF), which will need some tuning, but shouldn't be too bad
- If we can figure out a better solution to the urdf problems, that would be ideal
- Mujoco seemed to be able to load single modules pretty well, because of that guy's tool online (obj2mjcf). It would be nice if there was some support for urdf here, and perhaps he knows some tricks that I don't, but this isn't currently the case. Note: converting from mjcf to urdf via pybullet didn't work because there is some new feature in mujoco's mjcf description that pybullet does not currently know about
- Other formats such as SDF did not seem to fix the issue either. There is a chance that I may have been formatting the SDF incorrectly, but it wasn't clear what might have been the issue. This loaded a totally black ISS just like the URDF did before I changed the material property

