# How to modify the textures on the Astrobee

Since Pybullet prefers to load a URDF with a single texture applied to all meshes, we have to do some manual modification to both the meshes and the texture to get things to look right. 

First, we have to understand how texture images are mapped to a mesh. Inside the mesh file (OBJ/DAE/...), there is an embedding of a UV map, which dictates where the faces of the mesh fall on a texture image. Since we have a complex mesh and a 2D image, the mapping from the mesh faces to the image can often look very complicated, because of the "unwrapping" of the mesh that needs to happen. 

So, when Pybullet loads a URDF and applies the same image to everything, one part will look correct (the one where the UV map lines up with the texture as expected), and the rest of them will be totally messed up, because their UV maps are pointing to arbitrary parts of this texture. 

To solve this problem, we need to modify the meshes so that the UV maps of all of the parts are designed around a single texture PNG. This is where Blender comes in. However, before we use Blender, we have to make our texture file.

## Making a texture

(Disclaimer: There are almost certainly better ways to make a texture than this, but this was the easiest solution I could figure out without needing to learn all of the intricacies of Blender and texture unpacking/baking)

The Astrobee has pretty much just one component with a real texture (the skin on the side of the PMC), with a few other solid-color parts (the arm is black, the body is light gray with a few colored elements, ...). So, the approach I went for was that I added all of the solid colors I needed to the skin texture, right in the middle of the image (which is an unused area). 

This way, we can keep the UV map for the skin the same, but take all of the solid-color mesh elements and map them all to the respective color blocks in the image.

## Blender workflow

A quick note: I ended up making two Blender files - one for the body of the Astrobee, and one for everything else. The reasoning for this is that the body is comprised of 31 different little components, all separate meshes, whereas all of the other links on the Astrobee are single-mesh parts. It was easier for exporting purposes to deal with things in separate files. The only difference between these is that in the Body file, all of the meshes need to be exported to a single OBJ file, whereas in the "Everything Else" file, each mesh needed to be clicked on and exported as its own separate OBJ (for example, one of the gripper finger meshes).

1. Open Blender and import the original DAE file(s) from the `astrobee_media` repo
    - It's helpful to rename the imported parts with their actual mesh names (for example, `base_link`), because the generic names make it difficult to distinguish between multiple loaded parts
2. Click on the `Viewport Shading` button in the top right of the 3D Viewport to view the texture applied to the object. This button looks like a slightly-shaded circle
3. Click on `UV editing` in the Workspaces bar at the top of the window
4. Click on `Image` -> `Open` at the top of the UV editing area, and select the new texture (`astrobee_texture.png`)
5. Click on the part in the 3D VIewport's Object mode
    - If you have multiple parts in the Blender file, the fastest way to go about this process is to select all of the parts you want to be the same color, and then continue with this process
6. Switch the 3D Viewport to Edit mode to make the mesh visible (press `Tab` to activate, or use the dropdown at the top left of the 3D Viewport)
7. Click on the `Material Properties` tab in the Properties panel in the bottom right side (it looks like a red 3D version of the centroid symbol)
8. Rename the material to something more useful, like `astrobee_texture`
9.  Click on the dropdown next to `Base Color`, and switch the image to the desired texture image you loaded earlier
    - If the image dropdown doesn't exist, you'll need to click the yellow dot next to `Base Color` and switch it to `Image Texture`
10. In Edit mode in the 3D Viewport, press `A` to select all faces (this should bring up the UV map in the UV Editor area)
11. Click inside the UV Editor area, then press `A` to select all of the mesh faces
12. (Optional) Click on `UV` -> `Unwrap` -> `Smart UV Project` to get the mesh elements into a slightly better layout (or `U` -> `S` -> `Enter`)
    - Don't do this if you want to retain the UV mapping pattern (for instance, for the Astrobee skin, I did not unwrap the UV because it already lined up well with my texture file)
13. With these UV mesh elements selected, move them to the correct place in the texture according to their desired color
    - Press `G` to grab
    - Press `R` to rotate
    - Press `S` to scale
14. Once you have at least one part done, we need to make sure that all of the parts in the file are associated with the same texture/material. To do this, 
    1. Click on the part that has the new texture applied, then select all of the other parts in the file (by pressing `A` or with `Shift` + Click)
    2. Press `Ctrl + L` together
    3. Click on `Link Materials`
    4. All parts should now have the same texture/material. The other parts may look totally wrong if you haven't modified their UV maps yet, but that's fine for now
15. Repeat for the remaining parts

## Exporting meshes

Pybullet seems to prefer OBJ to DAE when it comes to applying textures, so we'll use this as our preferred mesh format.

If this is a Blender file with multiple URDF links loaded, you'll need to select them in the 3D Viewport in Object mode, and then `File` -> `Export` -> `Wavefront (.obj)`, with the `Selected Only` checkbox marked. If this is a Blender file where multiple meshes correspond to a single URDF link (like the body file), there's no need to select the parts, you can just export with this checkbox deactivated.

*Important note*: The default export orientation for DAE and OBJ is different! So, for any OBJ being exported, ensure that `Forward Axis` is `Y` and `Up Axis` is `Z`. This will match the DAE defaults and the way that NASA specified the original URDF with the DAE meshes.

## Loading into Pybullet

If you just call `loadURDF()`, the Astrobee may show up totally black. It seems that this isn't actually a problem with the texture, it's moreso that there is a unset alpha channel. So, loop through all of the link indices and set the `rgbaColor` to be `[1, 1, 1, 1]` using `changeVisualShape()`, and this should fix the problem

(This might be an incorrect assumption, but it seems to work. Note that I don't think the RGB values matter much here, so I chose white arbitrarily)
