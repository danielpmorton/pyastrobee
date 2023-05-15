# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8324769
# They use WXYZ quaternions with inertial definitions of angular velocity

# Implementing all of these functions with WXYZ for now
# We can figure out XYZW later

# SLERP will find the shortest path along the unit circle in 4D
# Here the curve cannot be an arc along the great circle because that
# implies a constant direction of the angular velocity vector
# So, we need to do some sort of a bezier-style curve on the surface of the
# 4d unit sphere to define the boundary conditions

import numpy as np


def dot(q1, q2):
    # Sum of element-wise multiplied values
    # (standard dot product definition)
    return np.dot(q1, q2)


def conj(q):
    return np.array([1, -1, -1, -1]) * q


def inv(q):
    N = np.linalg.norm(q)
    return (1 / (N**2)) * conj(q)


def pure(v):
    # Pure XZYZ quaternion (vector part only with 0 scalar)
    return np.concatenate([[0], v])


def multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def squared_norm(v):
    return np.dot(v, v)


def get_p(qi, qf, dqi, dqf, ddqi, ddqf, T):
    # Equation 13
    return np.row_stack(
        [
            qi,
            3 * qi + dqi * T,
            (ddqi * T**2 + 6 * dqi * T + 12 * qi) / 2,
            qf,
            3 * qf - dqf * T,
            (ddqf * T**2 - 6 * dqf * T + 12 * qf) / 2,
        ]
    )


def get_q_from_curve(p, tau):
    # Equation 12
    return (1 - tau) ** 3 * (p[0] + p[1] * tau + p[2] * tau**2) + (tau**3) * (
        p[3] + p[4] * (1 - tau) + p[5] * (1 - tau) ** 2
    )


def get_dq_from_curve(p, tau):
    # Derivative of equation 12 wrt time
    return (
        -3 * (1 - tau) ** 2 * (p[0] + p[1] * tau + p[2] * tau**2)
        + (1 - tau) ** 3 * (p[1] + 2 * p[2] * tau)
        + 3 * tau**2 * (p[3] + p[4] * (1 - tau) + p[5] * (1 - tau) ** 2)
        + tau**3 * (-p[4] - 2 * p[5] * (1 - tau))
    )


def get_ddq_from_curve(p, tau):
    # Second derivative of equation 12 wrt time
    return (
        6 * (1 - tau) * (p[0] + p[1] * tau + p[2] * tau**2)
        - 6 * (1 - tau) ** 2 * (p[1] + 2 * p[2] * tau)
        + (1 - tau) ** 3 * 2 * p[2]
        + 6 * tau * (p[3] + p[4] * (1 - tau) + p[5] * (1 - tau) ** 2)
        + 6 * tau**2 * (-p[4] - 2 * p[5] * (1 - tau))
        + tau**3 * 2 * p[5]
    )


def get_N(q):
    # Norm of quaternion
    return np.linalg.norm(q)


def get_dN(p, tau):
    pass


def get_w(q, dq):
    # Equation 6, part 2
    return (2 * dq * inv(q))[1:]  # Index the vector part of pure quat


def get_dw(q, dq, ddq):
    # Equation 7, part 2
    q_inv = inv(q)
    return (2 * ddq * q_inv - 2 * (dq * q_inv) ** 2)[
        1:
    ]  # Index the vector part of pure quat


def get_dqn(wn, dNn, qn):
    # Equation 21
    return multiply(pure((1 / 2) * wn + dNn), qn)


def get_ddqn(wn, dwn, dNn, ddNn, qn):
    # Equation 22
    return multiply(
        pure((1 / 2) * dwn + dNn * wn - (1 / 4) * squared_norm(wn) + ddNn), qn
    )


def poly_coeffs(tk, tf, qbark, wk, dwk, qf, dqf, ddqf, T):
    # Algorithm 2
    Tk = tf - tk
    Nk = np.linalg.norm(qbark)  # Should be 1

    # dNk = (1 / Nk) * dot(q, dq)
    ddNk = 0  # IGNORE THIS
    dqk = multiply(pure((1 / 2) * wk), qbark)
    ddqk = multiply(pure((1 / 2) * dwk - (1 / 4) * squared_norm(wk) + ddNk), qbark)
    return get_p(qbark, qf, dqk, dqf, ddqk, ddqf, T)


def quaternion_interpolation(ti, tf, qi, wi, dwi, qf, wf, dwf):
    # Algorithm 1
    if dot(qi, qf) < 0:
        qf = -qf  # Ensure shortest path interpolation
    dNf, ddNf = (0, 0)
    dqf = get_dqn(wf, dNf, qf)
    ddqf = get_ddqn(wf, dwf, dNf, ddNf, qf)
    T = tf - ti
    pfixed = poly_coeffs(ti, tf, qi, wi, dwi, qf, dqf, ddqf, T)
    t = ti
    # TODO This loop should be handled differently
    dt = 1 / 350  # NEW
    m = int(T / dt)  # NEW
    qs = np.zeros((m, 4))
    ws = np.zeros((m, 3))
    dws = np.zeros((m, 3))
    for k in range(m):
        t = ti + (k / m) * (tf - ti)  # NEW
        tau = (t - ti) / (tf - ti)
        q = get_q_from_curve(pfixed, tau)
        dq = get_dq_from_curve(pfixed, tau)
        ddq = get_ddq_from_curve(pfixed, tau)
        w = get_w(q, dq)
        dw = get_dw(q, dq, ddq)
        qs[k] = q
        ws[k] = w
        dws[k] = dw
    return qs, ws, dws


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ti = 0
    tf = 5
    qi = np.array([1, 0, 0, 0])
    # wi = np.zeros(3)
    wi = np.array([0.1, 0.2, 0.3])
    dwi = np.zeros(3)
    qf = np.random.rand(4)
    qf /= np.linalg.norm(qf)
    wf = np.zeros(3)
    dwf = np.zeros(3)
    qs, ws, dws = quaternion_interpolation(ti, tf, qi, wi, dwi, qf, wf, dwf)
    fig = plt.figure()
    subfigs = fig.subfigures(1, 3)
    left = subfigs[0].subplots(1, 4)
    middle = subfigs[1].subplots(1, 3)
    right = subfigs[2].subplots(1, 3)
    x_axis = range(qs.shape[0])
    q_labels = ["qw", "qx", "qy", "qz"]
    w_labels = ["wx", "wy", "wz"]
    dw_labels = ["ax", "ay", "az"]
    x_label = "Time"
    for i, ax in enumerate(left):
        ax.plot(x_axis, qs[:, i])
        ax.set_title(q_labels[i])
        ax.set_xlabel(x_label)
    for i, ax in enumerate(middle):
        ax.plot(x_axis, ws[:, i])
        ax.set_title(w_labels[i])
        ax.set_xlabel(x_label)
    for i, ax in enumerate(right):
        ax.plot(x_axis, dws[:, i])
        ax.set_title(dw_labels[i])
        ax.set_xlabel(x_label)
    plt.show()

    from pyastrobee.utils.quaternions import quats_to_angular_velocities, wxyz_to_xyzw

    new_ws = quats_to_angular_velocities(wxyz_to_xyzw(qs), 1 / 350)
    plt.figure()
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.plot(x_axis, new_ws)
        plt.title(w_labels[i])
    plt.show()
