"""Pybullet-specific helper functions"""

import os
import time
from typing import Optional, Union
import struct

import numpy as np
import numpy.typing as npt
import pybullet
import pybullet_data
from pybullet_utils.bullet_client import BulletClient

from pyastrobee.utils.mesh_utils import get_mesh_data
from pyastrobee.utils.python_utils import print_red


def load_rigid_object(
    filename: str,
    texture_filename: Optional[str] = None,
    scale: float = 1.0,
    pos: npt.ArrayLike = (0.0, 0.0, 0.0),
    orn: npt.ArrayLike = (0.0, 0.0, 0.0, 1.0),
    mass: float = 1.0,
    fixed: bool = False,
    rgba: npt.ArrayLike = (1.0, 1.0, 1.0, 1.0),
    client: Optional[BulletClient] = None,
) -> int:
    """Loads a rigid object from an OBJ or URDF file

    Args:
        filename (str): Path to the OBJ/URDF file to load
        texture_filename (str, optional): Path to a texture file to apply. Defaults to None, in which case no
            texture will be applied
        scale (float, optional): Scaling factor for the loaded object. Defaults to 1.0.
        pos (npt.ArrayLike, optional): Initial position for the loaded object. Defaults to (0, 0, 0).
        orn (npt.ArrayLike, optional): Initial XYZW quaternion orientation. Defaults to (0, 0, 0, 1).
        mass (float, optional): Mass of the loaded object. Defaults to 1.0.
        fixed (bool, optional): Whether or not to fix the object in space. Defaults to False.
        rgba (npt.ArrayLike, optional): Color of the object, expressed as RGBA, each within range [0, 1].
            Defaults to (1.0, 1.0, 1.0, 1.0) (white).
        client (BulletClient, optional): If connecting to multiple physics servers, include the client
            (the class instance, not just the ID) here. Defaults to None (use default connected client)

    Raises:
        ValueError: If the filename is not a valid OBJ or URDF

    Returns:
        int: ID number for the object
    """
    client: pybullet = pybullet if client is None else client
    # Deal with pybullet's weird handling of mass = 0 being fixed
    if mass < 0:
        raise ValueError("Mass should not be a negative value")
    if mass == 0:
        print_red(
            f"Warning: the mass of {filename} is 0, which will make it fixed. Use the 'fixed' parameter instead"
        )
    if fixed:
        mass = 0.0
    if filename.endswith(".obj"):  # mesh info
        xyz_scale = [scale, scale, scale]
        visual_id = client.createVisualShape(
            shapeType=pybullet.GEOM_MESH,
            rgbaColor=rgba,
            fileName=filename,
            meshScale=xyz_scale,
        )
        collision_id = client.createCollisionShape(
            shapeType=pybullet.GEOM_MESH, fileName=filename, meshScale=xyz_scale
        )
        rigid_id = client.createMultiBody(
            baseMass=mass,  # mass==0 => fixed at position where it is loaded
            basePosition=pos,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id,
            baseOrientation=orn,
        )
    elif filename.endswith(".urdf"):  # URDF file
        rigid_id = client.loadURDF(
            filename,
            pos,
            orn,
            useFixedBase=fixed,
            globalScaling=scale,
        )
    else:
        raise ValueError(
            f"Invalid filename: {filename}. Import either an OBJ or URDF file"
        )

    # TODO: decide if these parameters should be included as inputs rather than hard-coded
    client.changeDynamics(
        rigid_id,
        -1,
        mass,
        lateralFriction=1.0,
        spinningFriction=1.0,
        rollingFriction=1.0,
        restitution=0.0,
    )

    if texture_filename is not None:
        add_texture_to_rigid(rigid_id, texture_filename, client)

    return rigid_id


def add_texture_to_rigid(
    object_id: int,
    texture_filename: str,
    client: Optional[BulletClient] = None,
) -> None:
    """Applies a texture to a rigid object

    Args:
        object_id (int): The ID of the rigid object in the pybullet simulation
        texture_filename (str): Path to the texture file
        client (BulletClient, optional): If connecting to multiple physics servers, include the client
            (the class instance, not just the ID) here. Defaults to None (use default connected client)
    """
    client: pybullet = pybullet if client is None else client
    texture_id = client.loadTexture(texture_filename)
    kwargs = {}
    if hasattr(pybullet, "VISUAL_SHAPE_DOUBLE_SIDED"):
        kwargs["flags"] = pybullet.VISUAL_SHAPE_DOUBLE_SIDED

    num_joints = client.getNumJoints(object_id)
    # TODO: check on the indexing here (for now, assuming dedo is correct)
    for i in range(-1, num_joints):
        client.changeVisualShape(
            object_id,
            i,
            rgbaColor=[1, 1, 1, 1],
            textureUniqueId=texture_id,
            **kwargs,
        )


def add_texture_to_deformable(
    object_id: int,
    texture_filename: str,
    client: Optional[BulletClient] = None,
) -> None:
    """Applies a texture to a deformable object

    Args:
        object_id (int): The ID of the deformable object in the pybullet simulation
        texture_filename (str): Path to the texture file
        client (BulletClient, optional): If connecting to multiple physics servers, include the client
            (the class instance, not just the ID) here. Defaults to None (use default connected client)
    """
    client: pybullet = pybullet if client is None else client
    texture_id = client.loadTexture(texture_filename)
    kwargs = {}
    if hasattr(pybullet, "VISUAL_SHAPE_DOUBLE_SIDED"):
        kwargs["flags"] = pybullet.VISUAL_SHAPE_DOUBLE_SIDED
    client.changeVisualShape(
        object_id, -1, rgbaColor=[1, 1, 1, 1], textureUniqueId=texture_id, **kwargs
    )


def load_deformable_object(
    filename: str,
    texture_filename: Optional[str] = None,
    scale: float = 1.0,
    pos: npt.ArrayLike = (0.0, 0.0, 0.0),
    orn: npt.ArrayLike = (0.0, 0.0, 0.0, 1.0),
    mass: float = 1.0,
    bending_stiffness: float = 50.0,
    damping_stiffness: float = 0.1,
    elastic_stiffness: float = 50.0,
    friction_coeff: float = 0.1,
    self_collision: bool = False,
    sim_file_name="",
    client: Optional[BulletClient] = None,
) -> int:
    """Loads a deformable object from an OBJ file

    Notes: bending and elastic stiffness >90 cause instabilities.

    Args:
        filename (str): Path to the deformable object to load
        texture_filename (str, optional): Path to a texture file to apply. Defaults to None, in which case no
            texture will be applied
        scale (float, optional): Scaling factor for the loaded object. Defaults to 1.0.
        pos (npt.ArrayLike, optional): Initial position for the loaded object. Defaults to (0.0, 0.0, 0.0).
        orn (npt.ArrayLike, optional): Initial XYZW quaternion orientation. Defaults to (0, 0, 0, 1).
        mass (float, optional): Mass of the loaded object. Defaults to 1.0 (Keeping at 1.0 is the most stable option).
        bending_stiffness (float, optional): Bending stiffness of the loaded object. Defaults to 1.0.
        damping_stiffness (float, optional): Damping stiffness of the loaded object. Defaults to 0.1.
        elastic_stiffness (float, optional): Elastic stiffness of the loaded object. Defaults to 1.0.
        friction_coeff (float, optional): Friction coefficient of the loaded object. Defaults to 0.1.
        self_collision (bool, optional): Whether or not to allow self-collisions for the object. Defaults to False.
            Note: setting this as True seemed to lead to mesh collapse
        sim_file_name (str, optional): This is an undocumented input to loadSoftBody, but it appears to allow you
            to load an OBJ in the filename (for texture purposes) and a corresponding VTK in the simFileName for
            volumetric softbody physics
        client (BulletClient, optional): If connecting to multiple physics servers, include the client
            (the class instance, not just the ID) here. Defaults to None (use default connected client)

    Returns:
        int: ID number for the object
    """
    client: pybullet = pybullet if client is None else client
    if mass < 1.0:
        print_red(
            "Warning: mass > 1 is the most stable for deformables. Small mass can cause instabilities"
        )

    # TODO: decide if some of these parameters should be included as inputs rather than hard-coded
    deform_id = client.loadSoftBody(
        mass=mass,
        fileName=filename,
        scale=scale,
        basePosition=list(pos),
        baseOrientation=list(orn),
        springElasticStiffness=elastic_stiffness,
        springDampingStiffness=damping_stiffness,
        springBendingStiffness=bending_stiffness,
        frictionCoeff=friction_coeff,
        # collisionMargin=0.003,  # how far apart do two objects begin interacting
        useSelfCollision=self_collision,
        springDampingAllDirections=1,
        useFaceContact=True,
        useNeoHookean=0,
        useMassSpring=True,
        useBendingSprings=True,
        # repulsionStiffness=10000000,
        simFileName=sim_file_name,
    )

    # TODO figure out what this sparseSdfVoxelSize parameter actually does (it's not documented)
    # See pybullet examples/deformable_anchor.py for its usage there
    client.setPhysicsEngineParameter(sparseSdfVoxelSize=0.25)
    if texture_filename is not None:
        add_texture_to_deformable(deform_id, texture_filename, client)

    # Validate the size of the mesh to assure stability
    num_mesh_vertices = get_mesh_data(deform_id, client)[0]
    if num_mesh_vertices > 2**13:
        print_red(
            f"Warning: high number of mesh vertices: {num_mesh_vertices}. Consider a lower-res mesh"
        )

    return deform_id


def initialize_pybullet(
    use_gui: bool = True,
    physics_freq: float = 350,
    gravity: float = 0.0,
    bg_color: npt.ArrayLike = (0.0, 0.0, 0.0),
) -> pybullet:
    """Starts a pybullet client with the required physics parameters we care about

    NOTE: the client object returned from this function should ALWAYS be assigned to a variable
    to keep the simulation in scope. i.e. don't just call initialize_pybullet(), use client = initialize_pybullet()
    even if you're not worrying about multiple physics simulations

    Args:
        use_gui (bool, optional): Whether or not to use the GUI as opposed to headless. Defaults to True
        physics_freq (float, optional): Physics simulation frequency, in Hz. Defaults to 350.
            Note: Pybullet defaults to 240 Hz, but this seemed to be unstable for soft bodies
        gravity (float, optional): Z component of gravitational acceleration vector. Defaults to 0.
        bg_color (npt.ArrayLike, optional): RGB values for the GUI background, each in range [0, 1].
            Defaults to (0.0, 0.0, 0.0) (black). Note: (1.0, 1.0, 1.0) is white

    Returns:
        BulletClient: Pybullet physics simulation client
    """
    # Make sure we're in the right directory so filepaths work well with pybullet
    cwd = os.getcwd()
    if not cwd.endswith("pyastrobee") or cwd.endswith("pyastrobee/pyastrobee"):
        raise ConnectionRefusedError(
            f"You are running scripts from {cwd}.\nEnsure you're at the top-level pyastrobee directory."
        )
    # Ensure that the background color values are within the proper range
    bg_color = np.array(bg_color)
    if len(bg_color) != 3 or not (all(bg_color >= 0) and all(bg_color <= 1)):
        raise ValueError(f"Invalid background color: {bg_color}")
    bg_args = (
        f"--background_color_red={bg_color[0]} "
        + f"--background_color_green={bg_color[1]} "
        + f"--background_color_blue={bg_color[2]}"
    )
    # Connect to pybullet
    connection_mode = pybullet.GUI if use_gui else pybullet.DIRECT
    client: pybullet = BulletClient(connection_mode, options=bg_args)
    # Configure physics
    client.resetSimulation(pybullet.RESET_USE_DEFORMABLE_WORLD)
    client.setTimeStep(1.0 / physics_freq)
    client.setGravity(0, 0, gravity)
    # Configure search paths
    client.setAdditionalSearchPath(pybullet_data.getDataPath())
    # client.setAdditionalSearchPath(os.path.join(os.getcwd(), "pyastrobee/assets"))
    client.setAdditionalSearchPath(cwd)
    # Remove the extra windows in PyBullet GUI (until we use them for cameras).
    client.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, False)
    return client


def configure_visualization(
    camera_params: Optional[list[float]] = None,
    flags_to_enable: Optional[list[float]] = None,
    flags_to_disable: Optional[list[float]] = None,
    client: Optional[BulletClient] = None,
    **kwargs,
) -> None:
    """Configures the pybullet debug visualizer

    Args:
        camera_params (list[float], optional): Used to reset camera position. [dist, pitch, yaw, pos_x, pos_y, pos_z]
            where dist is the distance from eye to camera target, yaw is left/right angle, pitch is up/down angle, and
            the xyz positions are for the focus point. Defaults to None.
        flags_to_enable (list[float], optional): A list of pybullet flags (for example, COV_ENABLE_WIREFRAME).
            Defaults to None.
        flags_to_disable (list[float], optional): A list of pybullet flags (for example, COV_ENABLE_WIREFRAME).
            Defaults to None.
        client (BulletClient, optional): If connecting to multiple physics servers, include the client
            (the class instance, not just the ID) here. Defaults to None (use default connected client)
        **kwargs: Any additional kwargs to set. See the pybullet configureDebugVisualizer documentation for more info
    """
    client: pybullet = pybullet if client is None else client
    if camera_params:
        dist, pitch, yaw, pos_x, pos_y, pos_z = camera_params
        client.resetDebugVisualizerCamera(
            cameraDistance=dist,
            cameraPitch=pitch,
            cameraYaw=yaw,
            cameraTargetPosition=[pos_x, pos_y, pos_z],
        )
    if flags_to_enable:
        for flag in flags_to_enable:
            client.configureDebugVisualizer(flag, True)
    if flags_to_disable:
        for flag in flags_to_disable:
            client.configureDebugVisualizer(flag, False)
    if kwargs:
        client.configureDebugVisualizer(**kwargs)


def load_floor(
    texture_filename: Optional[str] = None,
    z_pos: float = 0.0,
    client: Optional[BulletClient] = None,
) -> int:
    """Loads a floor into the pybullet simulation

    Args:
        texture_filename (str, optional): If adding a texture to the floor plane, pass in the filename.
            Defaults to None.
        z_pos (float, optional): Height (z-coordinate) of the floor in the world. Defaults to 0.0
        client (BulletClient, optional): If connecting to multiple physics servers, include the client
            (the class instance, not just the ID) here. Defaults to None (use default connected client)

    Returns:
        int: Pybullet ID corresponding to the floor
    """
    client: pybullet = pybullet if client is None else client
    client.setAdditionalSearchPath(pybullet_data.getDataPath())
    floor_id = client.loadURDF("plane.urdf", basePosition=[0, 0, z_pos])
    if texture_filename is not None:
        texture_id = client.loadTexture(texture_filename)
        client.changeVisualShape(
            floor_id,
            -1,
            rgbaColor=[1, 1, 1, 0],
            textureUniqueId=texture_id,
        )
    return floor_id


def run_sim(
    viz_freq: float = 240,
    timeout: Optional[float] = None,
    client: Optional[BulletClient] = None,
):
    """Runs the pybullet simulation

    Args:
        viz_freq (float, optional): Frequency (Hz) to run the visualization (if connected via GUI). Defaults to 240.
        timeout (float, optional): Amount of time to run the simulation. Defaults to None, in which case the simulation
            will remain open until it is killed manually.
        client (BulletClient, optional): If connecting to multiple physics servers, include the client
            (the class instance, not just the ID) here. Defaults to None (use default connected client)

    Raises:
        ConnectionError: If a pybullet client is not currently running
        ValueError: If the visualization frequency is greater than the physics frequency
    """
    client: pybullet = pybullet if client is None else client
    connect_info: dict[str, int] = client.getConnectionInfo()
    if not connect_info["isConnected"]:
        raise ConnectionError("Connect to a pybullet client before running the sim")
    connect_mode = "GUI" if connect_info["connectionMethod"] == 1 else "DIRECT"
    phys_info = client.getPhysicsEngineParameters()
    phys_freq = 1.0 / phys_info["fixedTimeStep"]
    if viz_freq > phys_freq:
        raise ValueError(
            f"Cannot visualize ({viz_freq} Hz) faster than the physics ({phys_freq} Hz)"
        )

    if timeout is None:
        timeout = float("inf")
    start_time = time.time()
    try:
        while (time.time() - start_time < timeout) and client.isConnected():
            client.stepSimulation()
            if connect_mode == "GUI":
                time.sleep(1.0 / viz_freq)
    finally:
        client.disconnect()


def create_sphere(
    pos: npt.ArrayLike,
    mass: float,
    radius: float,
    use_collision: bool,
    rgba: npt.ArrayLike = (1, 1, 1, 1),
    client: Optional[BulletClient] = None,
) -> int:
    """Creates a rigid sphere in the Pybullet simulation

    Args:
        pos (npt.ArrayLike): Position of the sphere in world frame, shape (3)
        mass (float): Mass of the sphere. If set to 0, the sphere is fixed in space
        radius (float): Radius of the sphere
        use_collision (bool): Whether or not collision is enabled for the sphere
        rgba (npt.ArrayLike, optional): Color of the sphere, with each RGBA value being in [0, 1].
            Defaults to (1, 1, 1, 1) (white)
        client (BulletClient, optional): If connecting to multiple physics servers, include the client
            (the class instance, not just the ID) here. Defaults to None (use default connected client)

    Returns:
        int: ID of the sphere in Pybullet
    """
    client: pybullet = pybullet if client is None else client
    visual_id = client.createVisualShape(
        pybullet.GEOM_SPHERE, radius=radius, rgbaColor=rgba
    )
    if use_collision:
        collision_id = client.createCollisionShape(pybullet.GEOM_SPHERE, radius=radius)
    else:
        collision_id = -1
    sphere_id = client.createMultiBody(
        baseMass=mass,
        basePosition=pos,
        baseCollisionShapeIndex=collision_id,
        baseVisualShapeIndex=visual_id,
    )
    return sphere_id


def create_anchor(
    soft_body_id: int,
    vertex_id: int,
    parent_id: int,
    link_id: int,
    parent_frame_pos: Optional[list[float]] = None,
    add_geom: bool = False,
    geom_pos: Optional[npt.ArrayLike] = None,
    client: Optional[BulletClient] = None,
) -> tuple[int, Optional[int]]:
    """Creates an anchor between a softbody and another object (or the world)

    Args:
        soft_body_id (int): ID of the softbody in Pybullet
        vertex_id (int): Index of the mesh vertex on the softbody we are anchoring to
        parent_id (int): ID of the parent object in Pybullet. If anchoring to world, this will be -1
        link_id (int): Index of the link on the parent object. If the parent does not have links, use -1
        parent_frame_pos (Optional[list[float]]): If the anchor is being affixed to a specific point on the parent
            object, pass in the location here. Defaults to None.
        add_geom (bool, optional): Whether or not to add a small sphere to visualize the positioning on the anchor.
            Defaults to False.
        geom_pos (Optional[npt.ArrayLike]): Position of the sphere in world frame, shape (3,).
            Defaults to None, (in which case add_geom must also be None)
        client (BulletClient, optional): If connecting to multiple physics servers, include the client
            (the class instance, not just the ID) here. Defaults to None (use default connected client)

    Returns:
        tuple[int, Optional[int]]:
            int: The Pybullet ID for the anchor
            Optional[int]: The Pybullet ID for the geometry (if add_geom is False, this is None)
    """
    client: pybullet = pybullet if client is None else client
    anchor_id = client.createSoftBodyAnchor(
        soft_body_id, vertex_id, parent_id, link_id, parent_frame_pos
    )
    if add_geom:
        if geom_pos is None:
            raise ValueError(
                "If visualizing the anchor, the world-position of the anchor must be included"
            )
        # Create a collision-less sphere to visualize the anchor position
        geom_id = create_sphere(geom_pos, 0.01, 0.01, False, [0, 1, 0, 0.5], client)
        # Then create a secondary anchor to make sure this sphere stays in the right place
        geom_anchor_id = client.createSoftBodyAnchor(
            soft_body_id, vertex_id, geom_id, -1
        )
    else:
        geom_id = None
    return anchor_id, geom_id


def create_box(
    pos: npt.ArrayLike,
    orn: npt.ArrayLike,
    mass: float,
    sidelengths: npt.ArrayLike,
    use_collision: bool,
    rgba: npt.ArrayLike = (1, 1, 1, 1),
    client: Optional[BulletClient] = None,
) -> int:
    """Creates a rigid box in the Pybullet simulation

    Args:
        pos (npt.ArrayLike): Position of the box in world frame, shape (3)
        orn (npt.ArrayLike): Orientation (XYZW quaternion) of the box in world frame, shape (4,)
        mass (float): Mass of the box. If set to 0, the box is fixed in space
        sidelengths (npt.ArrayLike): Sidelengths of the box along the local XYZ axes, shape (3,)
        use_collision (bool): Whether or not collision is enabled for the box
        rgba (npt.ArrayLike, optional): Color of the box, with each RGBA value being in [0, 1].
            Defaults to (1, 1, 1, 1) (white)
        client (BulletClient, optional): If connecting to multiple physics servers, include the client
            (the class instance, not just the ID) here. Defaults to None (use default connected client)

    Returns:
        int: ID of the box in Pybullet
    """
    client: pybullet = pybullet if client is None else client
    if len(sidelengths) != 3:
        raise ValueError("Must provide the dimensions of the three sides of the box")
    half_extents = np.asarray(sidelengths) / 2
    visual_id = client.createVisualShape(
        pybullet.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=rgba,
    )
    if use_collision:
        collision_id = client.createCollisionShape(
            pybullet.GEOM_BOX,
            halfExtents=half_extents,
        )
    else:
        collision_id = -1
    box_id = client.createMultiBody(
        baseMass=mass,
        basePosition=pos,
        baseOrientation=orn,
        baseCollisionShapeIndex=collision_id,
        baseVisualShapeIndex=visual_id,
    )
    return box_id


def read_log_file(
    filename: str, verbose: bool = False
) -> list[list[Union[int, float]]]:
    """Reads a Pybullet state log for generic robots and objects

    We assume that when generating the log:
    - The loggingType was set to STATE_LOGGING_GENERIC_ROBOT
    - No softbodies were included in the logged items
    - The maxLogDof is left at the default value (12)

    This code is modified from bullet3/examples/pybullet/examples/kuka_with_cube_playback.py

    Args:
        filename (str): Filename for the saved log
        verbose (bool, optional): Whether to print info about the log when reading. Defaults to False.

    Returns:
        list[list[Union[int, float]]]: The Pybullet log.
            Length is (number of timesteps) * (number of objects logged)
            If multiple objects are logged, there will be multiple consecutive log entries for the same timestep
            Each record in the log contains the following info:
                step_count = record[0]
                timestamp = record[1]
                unique_id = record[2]
                position = record[3:6]
                orientation = record[6:10]
                velocity = record[10:13]
                angular_velocity = record[13:16]
                num_joints = record[16]
                joint_positions = record[17 : 17 + max_log_dof]
                joint_torques = record[17 + max_log_dof :]
    """
    with open(filename, "rb") as f:
        keys = f.readline().decode("utf8").rstrip("\n").split(",")
        fmt = f.readline().decode("utf8").rstrip("\n")
        # The byte number of one record
        sz = struct.calcsize(fmt)
        # The type number of one record
        ncols = len(fmt)
        info = {
            "filename": filename,
            "keys": keys,
            "format": fmt,
            "size": sz,
            "columns": ncols,
        }
        if verbose:
            print(info)
        # Read data
        whole_file = f.read()
    # split by alignment word
    chunks = whole_file.split(b"\xaa\xbb")
    log = []
    for chunk in chunks:
        if len(chunk) == sz:
            values = struct.unpack(fmt, chunk)
            record = []
            for i in range(ncols):
                record.append(values[i])
            log.append(record)
    return log


# TODO determine if this really is real time or if it's sim time during the original run...
def playback_from_log(
    log_file: str,
    real_time: bool = True,
    client: Optional[BulletClient] = None,
) -> None:
    """Reads a log file and then replays that logged simulation

    We assume that the environment has been set up the same way as the original environment
    when the logging occurred. If this is not the case, it can lead to odd results

    The logs also contain a lot more info like velocities and torques, but we don't really need
    this information to play things back (simply resetting pose will appear correct)

    This code is modified from bullet3/examples/pybullet/examples/kuka_with_cube_playback.py

    Args:
        log_file (str): Filename for the saved log
        real_time (bool, optional): Whether to play back the log in real time (according to the timestamps saved in
            the log). Defaults to True. If False, things will be very speedy
        client (BulletClient, optional): If connecting to multiple physics servers, include the client
            (the class instance, not just the ID) here. Defaults to None (use default connected client)
    """
    client: pybullet = pybullet if client is None else client
    log = read_log_file(log_file)
    start_time = time.time()
    for record in log:
        timestamp = record[1]
        uid = record[2]
        pos = record[3:6]
        orn = record[6:10]
        client.resetBasePositionAndOrientation(uid, pos, orn)
        num_joints = client.getNumJoints(uid)
        for i in range(num_joints):
            joint_info = client.getJointInfo(uid, i)
            q_index = joint_info[3]
            if q_index > -1:
                client.resetJointState(uid, i, record[q_index - 7 + 17])
        if real_time:
            time.sleep(max(0, timestamp - time.time() + start_time))
