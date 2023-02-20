"""Helper functions associated with rotations

NOTE: This code has been replaced by the wrapper around pytransform3d.
However, it is still useful for test cases, to ensure pytransform3d's conventions
and math match what we expect
"""

import numpy as np
import numpy.typing as npt


def Rx(theta: float) -> np.ndarray:
    """Rotation matrix for a rotation by theta radians about the X axis

    As an operator: Rx @ P will rotate point P by theta about the X axis
    As a mapping: Gives R_B_in_A ({A/B}R), where frame B is rotated theta radians about A's X axis
    """
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    )


def Ry(theta: float) -> np.ndarray:
    """Rotation matrix for a rotation by theta radians about the Y axis

    As an operator: Ry @ P will rotate point P by theta about the Y axis
    As a mapping: Gives R_B_in_A ({A/B}R), where frame B is rotated theta radians about A's Y axis
    """
    return np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )


def Rz(theta: float) -> np.ndarray:
    """Rotation matrix for a rotation by theta radians about the Z axis

    As an operator: Rz @ P will rotate point P by theta about the Z axis
    As a mapping: Gives R_B_in_A ({A/B}R), where frame B is rotated theta radians about A's Z axis
    """
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )


def euler_zyx_to_rmat(th_z: float, th_y: float, th_x: float) -> np.ndarray:
    """ZYX 3-angle Euler rotation matrix

    Args:
        th_z (float): Starting with frame A, angle to rotate about Z_A to obtain intermediate frame B'
        th_y (float): Angle to rotate about Y_B' to obtain intermediate frame B''
        th_x (float): Angle to rotate about X_B'' to obtain B

    Returns:
        np.ndarray: R_B_in_A ({A/B}R) where frame B is composed by three rotations in ZYX order starting from frame A
    """
    return Rz(th_z) @ Ry(th_y) @ Rx(th_x)


def euler_xyz_to_rmat(th_x: float, th_y: float, th_z: float) -> np.ndarray:
    """XYZ 3-angle Euler rotation matrix

    Args:
        th_x (float): Starting with frame A, angle to rotate about X_A to obtain intermediate frame B'
        th_y (float): Angle to rotate about Y_B' to obtain intermediate frame B''
        th_z (float): Angle to rotate about Z_B'' to obtain B

    Returns:
        np.ndarray: R_B_in_A ({A/B}R) where frame B is composed by three rotations in XYZ order starting from frame A
    """
    return Rx(th_x) @ Ry(th_y) @ Rz(th_z)


def rmat_to_axis_angle(rmat: np.ndarray) -> tuple[np.ndarray, float]:
    """Converts a rotation matrix into an axis-angle representation

    Args:
        rmat (np.ndarray): (3, 3) rotation matrix

    Raises:
        ZeroDivisionError: If the rotation matrix leads to a singularity
            when represented in axis-angle form

    Returns:
        Tuple of:
            np.ndarray: Axis of rotation. Shape (3,)
            float: Rotation angle
    """
    angle = np.arccos((rmat[0, 0] + rmat[1, 1] + rmat[2, 2] - 1) / 2)
    if np.abs(np.sin(angle)) < 1e-14:
        raise ZeroDivisionError("Cannot convert to axis-angle: Near singularity")
    axis = (1 / (2 * np.sin(angle))) * np.array(
        [rmat[2, 1] - rmat[1, 2], rmat[0, 2] - rmat[2, 0], rmat[1, 0] - rmat[0, 1]]
    )
    return axis, angle


def axis_angle_to_rmat(axis: list[float], angle: float) -> np.ndarray:
    """Converts an axis-angle representation to a rotation matrix

    Args:
        axis (list[float]): Axis to rotate around: [x, y, z]
        angle (float): Angle to rotate (radians)

    Returns:
        np.ndarray: Rotation matrix equivalent for the axis-angle representation
    """
    # Converting to a slightly more compact notation
    axis = normalize(axis)
    kx, ky, kz = axis
    ct = np.cos(angle)
    st = np.sin(angle)
    vt = 1 - ct
    return np.array(
        [
            [kx * kx * vt + ct, kx * ky * vt - kz * st, kx * kz * vt + ky * st],
            [kx * ky * vt + kz * st, ky * ky * vt + ct, ky * kz * vt - kx * st],
            [kx * kz * vt - ky * st, ky * kz * vt + kx * st, kz * kz * vt + ct],
        ]
    )


def euler_angles_to_rmat(convention: str, *angles: float) -> np.ndarray:
    """Converts euler angles of a specified convention (like 'xyz') to a rotation matrix

    Args:
        convention (str): Rotation axis order for the specified angles: Must be some
            permutation of 'xyz' - e.g. 'zyx', 'yxy', 'xyxyxyxyxy', ... and the length
            of the string must match the number of angles in the inputs
        *angles (float): Angle arguments. Length must match the convention string length

    Returns:
        np.ndarray: (3,3) rotation matrix
    """
    if not len(angles) == len(convention):
        raise ValueError("Number of angles must match the specified convention")
    if not all(c in {"x", "y", "z"} for c in convention.lower()):
        raise ValueError("Convention must only include x, y, and z")
    funcs = {"x": Rx, "y": Ry, "z": Rz}
    R = np.eye(3)
    for angle, axis in zip(angles, convention):
        R = R @ funcs[axis](angle)
    return R


def fixed_angles_to_rmat(convention: str, *angles: float) -> np.ndarray:
    """Converts fixed angles of a specified convention (like 'xyz') to a rotation matrix

    Args:
        convention (str): Rotation axis order for the specified angles: Must be some
            permutation of 'xyz' - e.g. 'zyx', 'yxy', 'xyxyxyxyxy', ... and the length
            of the string must match the number of angles in the inputs
        *angles (float): Angle arguments. Length must match the convention string length

    Returns:
        np.ndarray: (3,3) rotation matrix
    """
    return euler_angles_to_rmat(convention[::-1], *angles[::-1])


def rmat_to_euler_zyx(rmat: np.ndarray) -> tuple[float, float, float]:
    """Converts a rotation matrix to Euler ZYX angles

    Args:
        rmat (np.ndarray): (3,3) rotation matrix

    Raises:
        ZeroDivisionError: If there is a singularity in the representation

    Returns:
        tuple[float, float, float]: ZYX Euler angles
    """
    [r11, _, _], [r21, _, _], [r31, r32, r33] = rmat
    beta = np.arctan2(-r31, np.sqrt(r11**2 + r21**2))
    cb = np.cos(beta)
    if abs(cb) < 1e-14:
        raise ZeroDivisionError("At a singularity")
    alpha = np.arctan2(r21 / cb, r11 / cb)
    gamma = np.arctan2(r32 / cb, r33 / cb)
    return alpha, beta, gamma


def rmat_to_euler_xyz(rmat: np.ndarray) -> tuple[float, float, float]:
    """Converts a rotation matrix to Euler XYZ angles

    Args:
        rmat (np.ndarray): (3,3) rotation matrix

    Raises:
        ZeroDivisionError: If there is a singularity in the representation

    Returns:
        tuple[float, float, float]: XYZ Euler angles
    """
    [r11, r12, r13], [_, _, r23], [_, _, r33] = rmat
    beta = np.arctan2(r13, np.sqrt(r23**2 + r33**2))
    cb = np.cos(beta)
    if abs(cb) < 1e-14:
        raise ZeroDivisionError("At a singularity")
    alpha = np.arctan2(-r23 / cb, r33 / cb)
    gamma = np.arctan2(-r12 / cb, r11 / cb)
    return alpha, beta, gamma


def rmat_to_fixed_xyz(rmat: np.ndarray) -> tuple[float, float, float]:
    """Converts a rotation matrix to Fixed XYZ angles

    Args:
        rmat (np.ndarray): (3,3) rotation matrix

    Returns:
        tuple[float, float, float]: XYZ Fixed angles
    """
    return rmat_to_euler_zyx(rmat)[::-1]


def rmat_to_fixed_zyx(rmat: np.ndarray) -> tuple[float, float, float]:
    """Converts a rotation matrix to Fixed ZYX angles

    Args:
        rmat (np.ndarray): (3,3) rotation matrix

    Returns:
        tuple[float, float, float]: ZYX Fixed angles
    """
    return rmat_to_euler_xyz(rmat)[::-1]


def rmat_to_euler_params(rmat: np.ndarray, eta: int = 1) -> list[float]:
    """Converts a rotation matrix into Euler parameters

    Args:
        rmat (np.ndarray): (3,3) rotation matrix
        eta (int, optional): Parameter for sign determination, must be either 1 or -1. Defaults to 1.

    Returns:
        list[float]: The four Euler parameters
    """
    if eta not in {1, -1}:
        raise ValueError(f"Invalid eta: {eta}. Must be 1 or -1")
    [r11, r12, r13], [r21, r22, r23], [r31, r32, r33] = rmat
    # Perform an initial solve for the parameters
    l3 = (eta / 2) * np.sqrt(r11 + r22 + r33 + 1)
    l2 = (eta / 2) * np.sign(r21 - r12) * np.sqrt(-r11 - r22 + r33 + 1)
    l1 = (eta / 2) * np.sign(r13 - r31) * np.sqrt(-r11 + r22 - r33 + 1)
    l0 = (eta / 2) * np.sign(r32 - r23) * np.sqrt(r11 - r22 - r33 + 1)
    params = [l0, l1, l2, l3]
    # Update the solution based on which is largest
    d0 = 2 * np.sign(l0) * np.sqrt(r11 + r22 + r33 + 1)
    d1 = 2 * np.sign(l1) * np.sqrt(r11 - r22 - r33 + 1)
    d2 = 2 * np.sign(l2) * np.sqrt(-r11 + r22 - r33 + 1)
    d3 = 2 * np.sign(l3) * np.sqrt(-r11 - r22 + r33 + 1)
    max_ind = np.argmax(np.abs(params))
    if max_ind == 0:
        l0 = d0 / 4
        l1 = (r32 - r23) / d0
        l2 = (r13 - r31) / d0
        l3 = (r21 - r12) / d0
    elif max_ind == 1:
        l0 = (r32 - r23) / d1
        l1 = d1 / 4
        l2 = (r21 + r12) / d1
        l3 = (r13 + r31) / d1
    elif max_ind == 2:
        l0 = (r13 - r31) / d2
        l1 = (r21 + r12) / d2
        l2 = d2 / 4
        l3 = (r32 + r23) / d2
    else:  # max_ind == 3
        l0 = (r21 - r12) / d3
        l1 = (r13 + r31) / d3
        l2 = (r32 + r23) / d3
        l3 = d3 / 4
    return [l0, l1, l2, l3]


def euler_params_to_rmat(l0: float, l1: float, l2: float, l3: float) -> np.ndarray:
    """Converts Euler parameters into a rotation matrix

    Args:
        l0 (float): 1st Euler parameter
        l1 (float): 2nd Euler parameter
        l2 (float): 3rd Euler parameter
        l3 (float): 4th Euler parameter

    Returns:
        np.ndarray: Rotation matrix equivalent for the Euler parameter representation
    """
    return np.array(
        [
            [
                2 * (l0**2 + l1**2) - 1,
                2 * (l1 * l2 - l0 * l3),
                2 * (l1 * l3 + l0 * l2),
            ],
            [
                2 * (l1 * l2 + l0 * l3),
                2 * (l0**2 + l2**2) - 1,
                2 * (l2 * l3 - l0 * l1),
            ],
            [
                2 * (l1 * l3 - l0 * l2),
                2 * (l2 * l3 + l0 * l1),
                2 * (l0**2 + l3**2) - 1,
            ],
        ]
    )


def check_rotation_mat(R: np.ndarray, atol: float = 1e-14) -> bool:
    """Determines if a rotation matrix is valid by checking orthogonality and determinant

    Args:
        R (np.ndarray): A rotation matrix
        atol (float): Absolute tolerance on the closeness checks. Defaults to 1e-14.

    Returns:
        bool: Whether or not R is a valid rotation matrix
    """
    if not np.shape(R) == (3, 3):
        return False  # Incorrect size
    if not np.allclose(R @ np.transpose(R), np.eye(R.shape[0]), atol=atol):
        return False  # Its transpose is not its inverse
    if not np.isclose(np.linalg.det(R), 1):
        return False  # Determinant is not 1
    return True


def check_euler_params(l0: float, l1: float, l2: float, l3: float) -> bool:
    # TODO!
    raise NotImplementedError


def rotate_point(rmat: np.ndarray, point: npt.ArrayLike):
    # Use rotation matrix as operator within a single frame
    # TODO
    raise NotImplementedError


# TODO figure out a better place for this
def normalize(vec):
    return np.array(vec) / np.linalg.norm(vec)


def quat_to_rmat(quat: npt.ArrayLike) -> np.ndarray:
    """Converts XYZW quaternions to a rotation matrix

    Args:
        quat (npt.ArrayLike): XYZW quaternions

    Returns:
        np.ndarray: (3,3) rotation matrix
    """
    quat = normalize(quat)
    x, y, z, w = quat
    x2 = x * x
    y2 = y * y
    z2 = z * z
    # w2 = w * w
    return np.array(
        [
            [1 - 2 * y2 - 2 * z2, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x2 - 2 * z2, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x2 - 2 * y2],
        ]
    )


def rmat_to_quat(rmat: np.ndarray) -> np.ndarray:
    """Converts a rotation matrix into XYZW quaternions

    (computer graphics solution by Shoemake 1994, same as NASA's code)

    Args:
        rmat (np.ndarray): (3,3) rotation matrix

    Returns:
        np.ndarray: XYZW quaternions
    """

    tr = rmat[0, 0] + rmat[1, 1] + rmat[2, 2]
    if tr >= 0:
        s4 = 2.0 * np.sqrt(tr + 1.0)
        x = (rmat[2, 1] - rmat[1, 2]) / s4
        y = (rmat[0, 2] - rmat[2, 0]) / s4
        z = (rmat[1, 0] - rmat[0, 1]) / s4
        w = s4 / 4.0
    elif rmat[0, 0] > rmat[1, 1] and rmat[0, 0] > rmat[2, 2]:
        s4 = 2.0 * np.sqrt(1.0 + rmat[0, 0] - rmat[1, 1] - rmat[2, 2])
        x = s4 / 4.0
        y = (rmat[0, 1] + rmat[1, 0]) / s4
        z = (rmat[2, 0] + rmat[0, 2]) / s4
        w = (rmat[2, 1] - rmat[1, 2]) / s4
    elif rmat[1, 1] > rmat[2, 2]:
        s4 = 2.0 * np.sqrt(1.0 + rmat[1, 1] - rmat[0, 0] - rmat[2, 2])
        x = (rmat[0, 1] + rmat[1, 0]) / s4
        y = s4 / 4.0
        z = (rmat[1, 2] + rmat[2, 1]) / s4
        w = (rmat[0, 2] - rmat[2, 0]) / s4
    else:
        s4 = 2.0 * np.sqrt(1.0 + rmat[2, 2] - rmat[0, 0] - rmat[1, 1])
        x = (rmat[2, 0] + rmat[0, 2]) / s4
        y = (rmat[1, 2] + rmat[2, 1]) / s4
        z = s4 / 4.0
        w = (rmat[1, 0] - rmat[0, 1]) / s4

    return np.array([x, y, z, w])
