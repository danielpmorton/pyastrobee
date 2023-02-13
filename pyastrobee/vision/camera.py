"""Camera classes and hardware specifications

TODO figure out which astrobee cameras need to be modeled
TODO need to figure out what camera parameters are really needed
TODO determine how to best use the class inheritance


Things we need to know about the cameras:
Do they have a fixed focal distance? If so, what?
positions and orientations of each on the astrobee
intrinsics

Info on the camera specs and transformations can be found in:
astrobee/config/robots/sim.config
astrobee/config/robots/honey.config
astrobee/config/cameras.config
"""

from typing import Any

import numpy as np
import pybullet

# from pyastrobee.utils.rotations

# From astrobee/config/robots/sim.config
# Also check out astrobee/config/cameras.config
# And, astrobee/config/robots/honey.config

# TODO rename specs to something better
# And remove any parameters that pybullet can't use
NAV_CAM_SPECS = {
    "distortion_coeff": 0.9984679695413576,
    "intrinsics": [
        611.0529144295888,
        0.0,
        637.9586438046298,
        0.0,
        610.213041396481,
        558.0507290999258,
        0.0,
        0.0,
        1.0,
    ],
    "gain": 0,
    "exposure": 160,
}

PERCH_CAM_SPECS = {
    "distortion_coeff": 1.0,
    "intrinsics": [209.21199, 0.0, 94.688486, 0.0, 207.62067, 84.04047, 0.0, 0.0, 1.0],
    "gain": 100,
    "exposure": 150,
}

# _CAM_SPECS = {
#     "distortion_coeff": ,
#     "intrinsics": ,
#     "gain": ,
#     "exposure": ,
# }

# nav_cam = {
#     distortion_coeff=0.9984679695413576,
#     intrinsic_matrix={
#       611.0529144295888, 0.0, 637.9586438046298,
#       0.0, 610.213041396481, 558.0507290999258,
#       0.0, 0.0, 1.0
#     },
#     gain=0,
#     exposure=160
#   },
#   dock_cam = {
#     distortion_coeff=1.06251,
#     intrinsic_matrix={
#      830.0073133142722, 0.0, 566.9509633118676,
#       0.0, 829.3261610590715, 529.2929442341774,
#       0.0, 0.0, 1.0
#     },
#     gain=72,
#     exposure=127
#   },
# -- Placeholder for haz_cam, not accurate!
#   haz_cam = {
#     distortion_coeff = {-0.29405, -0.0597744, 0.00554234, 0.00463849},
#     intrinsic_matrix = {
#       217.12576, 0.0, 112.04125,
#       0.0, 216.12197, 82.598679,
#       0.0, 0.0, 1.0
#     },
#     gain=50,
#     exposure=150
#   },
#   perch_cam = {
#     distortion_coeff = 1.0,
#     intrinsic_matrix = {
#       209.21199, 0.0, 94.688486,
#       0.0, 207.62067, 84.04047,
#       0.0, 0.0, 1.0
#     },
#     gain=100,
#     exposure=150
#   },
# -- Placeholder for sci_cam, not accurate!
#   sci_cam = {
#     distortion_coeff = {0.128628, -0.167456, 0.00213421, -0.00262211},
#     intrinsic_matrix = {
#       859.44015, 0.0, 754.24485,
#       0.0, 849.35466, 487.7349,
#       0.0, 0.0, 1.0
#     },
#     gain=50,
#     exposure=150
#   },
#   nav_cam_to_haz_cam_timestamp_offset = -0.02,
#   nav_cam_to_sci_cam_timestamp_offset = 0.18
# }


# -- Engineering positions with idealized orientations
#   perch_cam_transform      = transform(vec3(-0.1331, 0.0509, -0.0166), quat4(0.000, -0.70710678118, 0.000, 0.70710678118)),-- placeholder, not valid!
#   haz_cam_transform        = transform(vec3(0.1328, 0.0362, -0.0826), quat4(-0.500, 0.500, -0.500, 0.500)), -- placeholder, not valid!
#   nav_cam_transform        = transform(vec3(0.1157+0.002, -0.0422, -0.0826), quat4(0.500, 0.500, 0.500, 0.500) ),
#   dock_cam_transform       = transform(vec3(-0.1032-0.0029, -0.0540, -0.0064), quat4(0.500, -0.500, -0.500, 0.500) ),
#   imu_transform            = transform(vec3(0.0247, 0.0183, 0.0094), quat4(0.000, 0.000, 0.70710678118, 0.70710678118) ),
#   -- Not accurate only for sim purposes
#   sci_cam_transform        = transform(vec3(0.118, 0.0, -0.096), quat4(0.500, 0.500, 0.500, 0.500) )
# };


def compute_view_matrix(robot_pose, camera_config) -> tuple[float, ...]:
    # Note: the output is NOT a numpy array, but rather a length-16 tuple of floats
    # representing the unpacked matrix. Pybullet needs it in this form for future functions

    # TODO remove any notimplemented

    # Eye position in cartesian world coordinates
    # Will probably involve a transformation matrix multiplication
    # e.g. (eye to world)  = (robot to world) @ (eye to robot)
    eye_position = NotImplemented
    # Focus point in world coordinates
    # something like (focal pt to world) = (eye to world) @ (focal pt to eye)
    # where (focal pt to eye) is just z translation or something
    target_position = NotImplemented
    # Up vector of the camera (points along the height direction of the image frame), in world coords
    # should just be a dot product of one axis (x, y, or z) with the (eye to world) matrix
    up_vector = NotImplemented
    return pybullet.computeViewMatrix(eye_position, target_position, up_vector)


# TODO need to update all of these
NAV_CAM_CONFIG = {
    "view_matrix": NotImplemented,
    "proj_matrix": NotImplemented,
    # "cam_forward": NotImplemented,
    # "cam_horiz": NotImplemented,
    # "cam_vert": NotImplemented,
    # "cam_dist": NotImplemented,
    # "cam_tgt": NotImplemented,
}

HAZ_CAM_CONFIG = {
    "view_matrix": NotImplemented,
    "proj_matrix": NotImplemented,
    # "cam_forward": NotImplemented,
    # "cam_horiz": NotImplemented,
    # "cam_vert": NotImplemented,
    # "cam_dist": NotImplemented,
    # "cam_tgt": NotImplemented,
}

PERCH_CAM_CONFIG = {
    "view_matrix": NotImplemented,
    "proj_matrix": NotImplemented,
    # "cam_forward": NotImplemented,
    # "cam_horiz": NotImplemented,
    # "cam_vert": NotImplemented,
    # "cam_dist": NotImplemented,
    # "cam_tgt": NotImplemented,
}


class Camera:
    def __init__(
        # TODO remove the default as none? See if there is a better way to load the config
        self,
        view_matrix=None,
        proj_matrix=None,
        width=None,
        height=None,
        # Edwin says that these parameters really aren't needed
        # (They encode the same info as the view and projection matrix)
        # So, can get rid of these, or find a better way of dynamically generating the matrices
        cam_forward=None,
        cam_horiz=None,
        cam_vert=None,
        cam_dist=None,
        cam_tgt=None,
    ):
        if not pybullet.isNumpyEnabled():
            raise Exception(
                "Camera functions assume that numpy is enabled. Check on your pybullet installation"
            )

        # Store inputs for the camera instance
        self.view_matrix = view_matrix
        self.proj_matrix = proj_matrix
        self.width = width
        self.height = height

    def load_config(self, config: dict[str, Any]):
        # TESTING THIS OUT. Pick one method to load this info
        self.view_matrix = config.get("view_matrix")
        self.proj_matrix = config.get("proj_matrix")
        self.width = config.get("width")
        self.height = config.get("height")

    def capture_image(self):
        # The output of this function depends on if numpy is enabled within pybullet
        # We will assume it is
        camera_output: tuple[
            int, int, np.ndarray, np.ndarray, np.ndarray
        ] = pybullet.getCameraImage(
            self.width,
            self.height,
            self.view_matrix,
            self.proj_matrix,
            shadow=True,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
        )
        width, height, rgba, depth, segmentation = camera_output  # Unpack

        # Explanation of pybullet getCameraImage outputs:
        # width (Same as the width input, it's the width of the image in pixels)
        # height (Same as the height input, it's the height of the image in pixels)
        # rgba: ndarray[int], size (height, width, 4)
        # depth: ndarray[float], size (height, width)
        # segmentation: ndarray[int], size(height, width) ints corresporing to an index of an object in the image.
        #    Seems to have -1 if no objects visible

        # Reshape these inputs into a more reasonable format with numpy:


class NavCam(Camera):
    def __init__(self):
        super().__init__(**NAV_CAM_CONFIG)


class HazCam(Camera):
    def __init__(self):
        super().__init__(**HAZ_CAM_CONFIG)


class PerchCam(Camera):
    def __init__(self):
        super().__init__(**PERCH_CAM_CONFIG)
