"""Script to load Astrobee URDFs and view them in a Pybullet GUI window

TODO
- Get bullet to import the right colors for the astrobee? May need to specify in URDF
"""

from astrobee_pybullet.utils.bullet_utils import (
    initialize_pybullet,
    load_deformable_object,
    load_rigid_object,
    run_sim,
)


def main():
    astrobee_urdf = "urdf/astrobee.urdf"
    iss_urdf = "urdf/iss.urdf"
    bag_obj = "meshes/lumpy_remesh.obj"

    # Some starting positions for the astrobee within the ISS
    cupola = (-5.0, 0.0, 5.0)  # Don't use this one for now until the mesh is fixed
    us_lab = (3.0, 0.0, 5.0)
    outside = (0.0, -5.0, 5.0)  # A random place outside the ISS
    origin = (0.0, 0.0, 0.0)

    initialize_pybullet(use_gui=True)
    load_rigid_object(iss_urdf, fixed=True)
    load_rigid_object(astrobee_urdf, pos=outside)
    load_deformable_object(bag_obj, pos=origin)
    run_sim()


if __name__ == "__main__":
    main()
