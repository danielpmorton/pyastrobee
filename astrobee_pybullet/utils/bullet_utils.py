"""Pybullet-specific helper functions

TODO:
- Clear up URDF/OBJ method confusion
"""

import os
import time
from typing import Optional

import pybullet
import pybullet_data

from astrobee_pybullet.utils.mesh_utils import get_mesh_data
from astrobee_pybullet.utils.python_utils import print_red


def load_rigid_object(
    filename: str,
    texture_filename: Optional[str] = None,
    scale: float = 1.0,
    pos: list[float] = [0.0, 0.0, 0.0],
    orn: list[float] = [0.0, 0.0, 0.0],
    mass: float = 1.0,
    fixed: bool = False,
    rgba: list[float] = [1.0, 1.0, 1.0, 1.0],
) -> int:
    """Loads a rigid object from an OBJ or URDF file

    TODO consider clearing up some of the confusion between which inputs apply to which import methods
    (e.g. mass/fixed differs between these, and rgba only applies to obj)

    Args:
        filename (str): Path to the OBJ/URDF file to load
        texture_filename (str, optional): Path to a texture file to apply. Defaults to None, in which case no
            texture will be applied
        scale (float, optional): Scaling factor for the loaded object. Defaults to 1.0.
        pos (list[float], optional): Initial position for the loaded object. Defaults to [0.0, 0.0, 0.0].
        orn (list[float], optional): Initial (euler) orientation for the loaded object. Defaults to [0.0, 0.0, 0.0].
        mass (float, optional): Mass of the loaded object. Defaults to 1.0.
        fixed (bool, optional): Whether or not to fix the object in space. Defaults to False.
        rgba (list[float], optional): Color of the object, expressed as RGBA, each within range [0, 1].
            Defaults to [1.0, 1.0, 1.0, 1.0] (white).

    Raises:
        ValueError: If the filename is not a valid OBJ or URDF

    Returns:
        int: ID number for the object
    """
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
        visual_id = pybullet.createVisualShape(
            shapeType=pybullet.GEOM_MESH,
            rgbaColor=rgba,
            fileName=filename,
            meshScale=xyz_scale,
        )
        collision_id = pybullet.createCollisionShape(
            shapeType=pybullet.GEOM_MESH, fileName=filename, meshScale=xyz_scale
        )
        rigid_id = pybullet.createMultiBody(
            baseMass=mass,  # mass==0 => fixed at position where it is loaded
            basePosition=pos,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id,
            baseOrientation=pybullet.getQuaternionFromEuler(orn),
        )
    elif filename.endswith(".urdf"):  # URDF file
        rigid_id = pybullet.loadURDF(
            filename,
            pos,
            pybullet.getQuaternionFromEuler(orn),
            useFixedBase=fixed,
            globalScaling=scale,
        )
    else:
        raise ValueError(
            f"Invalid filename: {filename}. Import either an OBJ or URDF file"
        )

    # TODO: decide if these parameters should be included as inputs rather than hard-coded
    pybullet.changeDynamics(
        rigid_id,
        -1,
        mass,
        lateralFriction=1.0,
        spinningFriction=1.0,
        rollingFriction=1.0,
        restitution=0.0,
    )

    if texture_filename is not None:
        add_texture_to_rigid(rigid_id, texture_filename)

    return rigid_id


def add_texture_to_rigid(object_id: int, texture_filename: str) -> None:
    """Applies a texture to a rigid object

    Args:
        object_id (int): The ID of the rigid object in the pybullet simulation
        texture_filename (str): Path to the texture file
    """
    texture_id = pybullet.loadTexture(texture_filename)
    kwargs = {}
    if hasattr(pybullet, "VISUAL_SHAPE_DOUBLE_SIDED"):
        kwargs["flags"] = pybullet.VISUAL_SHAPE_DOUBLE_SIDED

    num_joints = pybullet.getNumJoints(object_id)
    # TODO: check on the indexing here (for now, assuming dedo is correct)
    for i in range(-1, num_joints):
        pybullet.changeVisualShape(
            object_id,
            i,
            rgbaColor=[1, 1, 1, 1],
            textureUniqueId=texture_id,
            **kwargs,
        )


def add_texture_to_deformable(object_id: int, texture_filename: str) -> None:
    """Applies a texture to a deformable object

    Args:
        object_id (int): The ID of the deformable object in the pybullet simulation
        texture_filename (str): Path to the texture file
    """
    texture_id = pybullet.loadTexture(texture_filename)
    kwargs = {}
    if hasattr(pybullet, "VISUAL_SHAPE_DOUBLE_SIDED"):
        kwargs["flags"] = pybullet.VISUAL_SHAPE_DOUBLE_SIDED
    pybullet.changeVisualShape(
        object_id, -1, rgbaColor=[1, 1, 1, 1], textureUniqueId=texture_id, **kwargs
    )


def load_deformable_object(
    filename: str,
    texture_filename: Optional[str] = None,
    scale: float = 1.0,
    pos: list[float] = [0.0, 0.0, 0.0],
    orn: list[float] = [0.0, 0.0, 0.0],
    mass: float = 1.0,
    bending_stiffness: float = 1.0,
    damping_stiffness: float = 0.1,
    elastic_stiffness: float = 1.0,
    friction_coeff: float = 0.1,
    self_collision: bool = False,
) -> int:
    """Loads a deformable object from an OBJ file

    TODO add support for deformable URDF files!
    TODO check if it makes any sense (or is even possible) to have a fixed deformable?

    Args:
        filename (str): Path to the deformable object to load
        texture_filename (str, optional): Path to a texture file to apply. Defaults to None, in which case no
            texture will be applied
        scale (float, optional): Scaling factor for the loaded object. Defaults to 1.0.
        pos (list[float], optional): Initial position for the loaded object. Defaults to [0.0, 0.0, 0.0].
        orn (list[float], optional): Initial (euler) orientation for the loaded object. Defaults to [0.0, 0.0, 0.0].
        mass (float, optional): Mass of the loaded object. Defaults to 1.0 (Keeping at 1.0 is the most stable option).
        bending_stiffness (float, optional): Bending stiffness of the loaded object. Defaults to 1.0.
        damping_stiffness (float, optional): Damping stiffness of the loaded object. Defaults to 0.1.
        elastic_stiffness (float, optional): Elastic stiffness of the loaded object. Defaults to 1.0.
        friction_coeff (float, optional): Friction coefficient of the loaded object. Defaults to 0.1.
        self_collision (bool, optional): Whether or not to allow self-collisions for the object. Defaults to False.
            Note: setting this as True seemed to lead to mesh collapse

    Returns:
        int: ID number for the object
    """
    if mass != 1.0:
        print_red(
            "Warning: mass = 1 is the most stable for deformables. Small mass can cause instabilities"
        )

    # TODO: decide if some of these parameters should be included as inputs rather than hard-coded
    deform_id = pybullet.loadSoftBody(
        mass=mass,
        fileName=filename,
        scale=scale,
        basePosition=pos,
        baseOrientation=pybullet.getQuaternionFromEuler(orn),
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
    )

    # TODO figure out what this sparseSdfVoxelSize parameter actually does (it's not documented)
    # See pybullet examples/deformable_anchor.py for its usage there
    pybullet.setPhysicsEngineParameter(sparseSdfVoxelSize=0.25)
    if texture_filename is not None:
        add_texture_to_deformable(deform_id, texture_filename)

    # Validate the size of the mesh to assure stability
    num_mesh_vertices = get_mesh_data(pybullet, deform_id)[0]
    if num_mesh_vertices > 2**13:
        print_red(
            f"Warning: high number of mesh vertices: {num_mesh_vertices}. Consider a lower-res mesh"
        )

    return deform_id


def initialize_pybullet(
    use_gui: bool = True, physics_freq: float = 350, gravity: float = 0.0
) -> int:
    """Starts a pybullet client with the required physics parameters we care about

    Args:
        use_gui (bool, optional): Whether or not to use the GUI as opposed to headless. Defaults to True
        physics_freq (float, optional): Physics simulation frequency, in Hz. Defaults to 350.
            Note: Pybullet defaults to 240 Hz, but this seemed to be unstable for soft bodies
        gravity (float, optional): Z component of gravitational acceleration vector. Defaults to 0.

    Returns:
        int: A Physics Client ID
    """

    if use_gui:
        client_id = pybullet.connect(pybullet.GUI)
    else:
        client_id = pybullet.connect(pybullet.DIRECT)
    # Configure physics
    pybullet.resetSimulation(pybullet.RESET_USE_DEFORMABLE_WORLD)
    pybullet.setTimeStep(1.0 / physics_freq)
    pybullet.setGravity(0, 0, gravity)
    # Configure search paths
    pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
    pybullet.setAdditionalSearchPath(
        os.path.join(os.getcwd(), "astrobee_pybullet/resources")
    )
    return client_id


def configure_visualization(
    camera_params: Optional[list[float]] = None,
    flags_to_enable: Optional[list[float]] = None,
    flags_to_disable: Optional[list[float]] = None,
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
        **kwargs: Any additional kwargs to set. See the pybullet configureDebugVisualizer documentation for more info
    """
    if camera_params:
        dist, pitch, yaw, pos_x, pos_y, pos_z = camera_params
        pybullet.resetDebugVisualizerCamera(
            cameraDistance=dist,
            cameraPitch=pitch,
            cameraYaw=yaw,
            cameraTargetPosition=[pos_x, pos_y, pos_z],
        )
    if flags_to_enable:
        for flag in flags_to_enable:
            pybullet.configureDebugVisualizer(flag, True)
    if flags_to_disable:
        for flag in flags_to_disable:
            pybullet.configureDebugVisualizer(flag, False)
    if kwargs:
        pybullet.configureDebugVisualizer(**kwargs)


def load_floor(texture_filename: Optional[str] = None, z_pos: float = 0.0) -> None:
    """Loads a floor into the pybullet simulation

    Args:
        texture_filename (str, optional): If adding a texture to the floor plane, pass in the filename.
            Defaults to None.
        z_pos (float, optional): Height (z-coordinate) of the floor in the world. Defaults to 0.0
    """
    pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
    floor_id = pybullet.loadURDF("plane.urdf", basePosition=[0, 0, z_pos])
    if texture_filename is not None:
        texture_id = pybullet.loadTexture(texture_filename)
        pybullet.changeVisualShape(
            floor_id,
            -1,
            rgbaColor=[1, 1, 1, 1],
            textureUniqueId=texture_id,
        )


def run_sim(viz_freq: float = 120, timeout: Optional[float] = None):
    """Runs the pybullet simulation

    Args:
        viz_freq (float, optional): Frequency (Hz) to run the visualization (if connected via GUI). Defaults to 120.
        timeout (float, optional): Amount of time to run the simulation. Defaults to None, in which case the simulation
            will remain open until it is killed manually.

    Raises:
        ConnectionError: If a pybullet client is not currently running
        ValueError: If the visualization frequency is greater than the physics frequency
    """
    connect_info: dict[str, int] = pybullet.getConnectionInfo()
    if not connect_info["isConnected"]:
        raise ConnectionError("Connect to a pybullet client before running the sim")
    connect_mode = "GUI" if connect_info["connectionMethod"] == 1 else "DIRECT"
    phys_info = pybullet.getPhysicsEngineParameters()
    phys_freq = 1.0 / phys_info["fixedTimeStep"]
    if viz_freq > phys_freq:
        raise ValueError(
            f"Cannot visualize ({viz_freq} Hz) faster than the physics ({phys_freq} Hz)"
        )

    if timeout is None:
        timeout = float("inf")
    start_time = time.time()
    while (time.time() - start_time < timeout) and pybullet.isConnected():
        pybullet.stepSimulation()
        if connect_mode == "GUI":
            time.sleep(1.0 / viz_freq)
