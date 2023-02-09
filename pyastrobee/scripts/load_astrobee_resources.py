"""Script to load Astrobee URDFs and view them in a Pybullet GUI window

TODO
- Get bullet to import the right colors for the astrobee? May need to specify in URDF
"""

import pybullet
import pybullet_data

from pyastrobee.utils.bullet_utils import (
    initialize_pybullet,
    load_deformable_object,
    load_rigid_object,
    run_sim,
    load_floor,
)


def main():
    # idk why the relative paths aren't working anymore.... could be the change to the pybullet version?
    astrobee_urdf = "urdf/astrobee.urdf"
    # iss_urdf = "urdf/iss.urdf"
    # iss_urdf = "/home/dan/pyastrobee/pyastrobee/resources/urdf/iss.urdf"
    iss_urdf = "/home/dan/pyastrobee/astrobee_media/astrobee_iss/urdf/model.urdf"

    # bag_obj = "meshes/lumpy_remesh.obj"
    # bag_obj = "/home/dan/pyastrobee/pyastrobee/resources/meshes/lumpy_remesh.obj"
    # bag_obj = "meshes/bag_remesh_3.obj"
    # bag_obj = "/home/dan/temp/bag.obj"
    # bag_obj = "banana.obj"
    # bag_obj = "/home/dan/pyastrobee/pyastrobee/resources/meshes/bag_remesh_3.obj"
    # bag_obj = "/home/dan/dedo/dedo/data/bags/backpack_0.obj"
    # bag_obj = "/home/dan/pyastrobee/pyastrobee/resources/meshes/cargo.obj"

    bag_obj = "/home/dan/pyastrobee/pyastrobee/resources/meshes/test.obj"
    bag_mtl = "/home/dan/pyastrobee/pyastrobee/resources/meshes/test.mtl"

    # Some starting positions for the astrobee within the ISS
    cupola = (-5.0, 0.0, 5.0)  # Don't use this one for now until the mesh is fixed
    us_lab = (3.0, 0.0, 5.0)
    outside = (0.0, -5.0, 5.0)  # A random place outside the ISS
    origin = (0.0, 0.0, 0.0)

    initialize_pybullet(use_gui=True)
    # load_rigid_object(iss_urdf, fixed=True)
    # load_rigid_object(astrobee_urdf, pos=outside)
    # load_deformable_object(
    #     bag_obj, pos=origin, elastic_stiffness=100, bending_stiffness=100
    # )

    # load_rigid_object(iss_urdf, fixed=True)
    load_rigid_object(
        "/home/dan/pyastrobee/astrobee_media/astrobee_iss/meshes/obj/eu_lab.obj",
        "/home/dan/pyastrobee/astrobee_media/astrobee_iss/meshes/obj/eu_lab.mtl",
        fixed=True,
    )
    # load_rigid_object(astrobee_urdf, pos=outside)
    # load_deformable_object(
    #     bag_obj, pos=origin, elastic_stiffness=100, bending_stiffness=100
    # )

    # pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
    # load_rigid_object(iss_urdf, fixed=True)
    # load_rigid_object(astrobee_urdf, pos=outside)
    # pybullet.setGravity(0, 0, -9.81)
    # load_floor()
    # pybullet.setTimeStep(1 / 350)  # 1/500 worked
    # load_deformable_object(
    #     bag_obj,
    #     texture_filename=bag_mtl,
    #     pos=[0, 0, 2],
    #     elastic_stiffness=100,
    #     bending_stiffness=100,
    #     self_collision=False,
    # )

    run_sim()


if __name__ == "__main__":
    main()
