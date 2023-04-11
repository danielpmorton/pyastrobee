# Bag Dynamics

## Geometry

The coordinate frame below indicates the orientation of the bag when it is loaded into Pybullet as a rigid object. The origin of the bag is at the center of the rectangular prism. 

```
      .+------+        Z
    .' |    .'|        |  y
   +---+--+'  |        | / 
   |   |  |   |        |/----> x
H  |  .+--+---+        
   |.'    | .'  W
   +------+'
       L
```

The CAD model (and the mesh) has the following dimensions:
```
L = 50 cm (0.50 m)
W = 25 cm (0.25 m)
H = 42 cm (0.42 m)
```

## Inertia

If we assume any effects from the handle are negligible, that the bag is perfectly rectangular, and that the distribution of mass inside the bag is uniform, the inertia tensor is as follows:

$$
I =
\begin{bmatrix}
I_{xx} & 0 & 0\\
0 & I_{yy} & 0\\
0 & 0 & I_{zz}
\end{bmatrix}
$$

where (in the coordinate frame shown above)

$I_{xx} = \frac{1}{12}m(W^2 + H^2)$

$I_{yy} = \frac{1}{12}m(L^2 + H^2)$

$I_{zz} = \frac{1}{12}m(L^2 + W^2)$

If we want to add a nonuniform distribution of the mass in the future, we will most likely have to do something weird, because Pybullet's changeDynamics function seems to only handle the diagonal inertia entries. But, there also seems to be more fine-tuned control of inertia matrices in URDFs.

If we set $m = 10$ (as an example), this gives us the following inertia tensor:

$$
I =
\begin{bmatrix}
0.19990833 & 0 & 0\\
0 & 0.35533333 & 0\\
0 & 0 & 0.26041666
\end{bmatrix}
$$

## Pybullet notes

### Rigid body dynamics and origin location

I originally wondered if it would be easier to make the URDF if we had the bag mesh defined with the origin centered at the handle. However, after loading this mesh into Pybullet, I noticed it had very odd dynamics. It seemed that the center of mass was way off - near the handle. Originally, I was wondering if this was because the mesh was denser in this area, but the same mesh with a different origin didn't have this issue. 

So, it seems that Pybullet defines the center of mass for an imported mesh by its origin! This is quite important to remember - especially when we are loading a mesh outside of a URDF (which may specifically define the inertial frame and the moments of inertia). I suppose with the changeDynamics function we might be able to get around this, but for now, we can just remember to create all of the CAD with this in mind. 

### Softbody dynamics

Position and orientation of the softbody are relatively accurate with Pybullet's `getBasePositionAndOrientation` function, but `getBaseVelocity` does not seem to provide an accurate measurement. I've written a function `get_bag_frame` to determine a local reference frame for the bag based on the corners of the main compartment, which may be useful because `getBasePositionAndOrientation` is slightly unclear in how this is actually calculated. From preliminary tests, it looks like both frames (from my method and pybullet's) are roughly the same. However, this does not solve the velocity issue (a future TODO). We may be able to track the position/orientation across multiple timesteps and calculate the velocity that way, if it comes down to it and we need that info.
