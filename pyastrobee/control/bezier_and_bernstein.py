"""Generating trajectories using Bezier curves and Bernstein polynomials

When compared with polynomial trajectories, these are:
- More numerically stable during optimization
- Easier to formulate constraints on maximum position, acceleration, ...
- Easier to formulate cost functions (such as minimizing jerk)
- Easier to specify motions within a safe convex set (the curve will be in the convex hull of the control points, so
  we can just impose that the control points are within a safe set)

Stephen Boyd and Tobia Marcucci recommended using these 
"""


from typing import Union

import numpy as np
import numpy.typing as npt
import cvxpy as cp

# from scipy.interpolate import BPoly
from scipy.special import binom
import matplotlib.pyplot as plt


def bernstein(
    M: int, n: int, a: float, b: float, t: Union[float, npt.ArrayLike]
) -> npt.ArrayLike:
    """Evaluate the nth Bernstein polynomial of degree M at a point (points) t

    Notation is from "Fast Path Planning Through Large Collections of Safe Boxes"

    Args:
        M (int): Degree of the Bernstein basis
        n (int): Index of the Bernstein polynomial
        a (float): Interval minimum value (e.g. starting time of trajectory)
        b (float): Interval maximum value (e.g. ending time of trajectory)
        t (Union[float, npt.ArrayLike]): Evaluation point(s) (e.g. trajectory value at time t)

    Returns:
        npt.ArrayLike: Evaluation(s) of the bernstein polynomial at point(s) t. Returns a float if t is a float,
            otherwise will return an array of evaluations
    """
    assert n <= M
    assert b > a
    if np.ndim(t) >= 0:
        t = np.asarray(t)
    assert np.all(a <= t) and np.all(t <= b)
    return binom(M, n) * ((t - a) / (b - a)) ** n * ((b - t) / (b - a)) ** (M - n)


# DIRECTLY FROM TOBIA'S REPO (TODO cite)
class BezierCurve:
    def __init__(self, points, a=0, b=1):
        assert b > a

        self.points = points
        self.h = points.shape[0] - 1
        self.d = points.shape[1]
        self.a = a
        self.b = b
        self.duration = b - a

    def __call__(self, t):
        c = np.array([self.berstein(t, n) for n in range(self.h + 1)])
        return c.T.dot(self.points)

    def berstein(self, t, n):
        c1 = binom(self.h, n)
        c2 = (t - self.a) / self.duration
        c3 = (self.b - t) / self.duration
        value = c1 * c2**n * c3 ** (self.h - n)

        return value

    def start_point(self):
        return self.points[0]

    def end_point(self):
        return self.points[-1]

    def derivative(self):
        points = (self.points[1:] - self.points[:-1]) * (self.h / self.duration)

        return BezierCurve(points, self.a, self.b)

    def l2_squared(self):
        A = np.zeros((self.h + 1, self.h + 1))
        for m in range(self.h + 1):
            for n in range(self.h + 1):
                A[m, n] = binom(self.h, m) * binom(self.h, n) / binom(2 * self.h, m + n)
        A *= self.duration / (2 * self.h + 1)
        A = np.kron(A, np.eye(self.d))

        p = self.points.flatten()

        return p.dot(A.dot(p))

    def plot2d(self, samples=51, **kwargs):
        import matplotlib.pyplot as plt

        options = {"c": "b"}
        options.update(kwargs)
        t = np.linspace(self.a, self.b, samples)
        plt.plot(*self(t).T, **options)

    def scatter2d(self, **kwargs):
        import matplotlib.pyplot as plt

        options = {"fc": "orange", "ec": "k", "zorder": 3}
        options.update(kwargs)
        plt.scatter(*self.points.T, **options)

    def plot_2dpolygon(self, **kwargs):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        from scipy.spatial import ConvexHull

        options = {"fc": "lightcoral"}
        options.update(kwargs)
        hull = ConvexHull(self.points)
        ordered_points = hull.points[hull.vertices]
        poly = Polygon(ordered_points, **options)
        plt.gca().add_patch(poly)

    # MY NEW FUNCTION - needs some cleaning up
    def plot3d(self, samples=51, **kwargs):
        t = np.linspace(self.a, self.b, samples)
        # Evaluate the positions, velocities, and accelerations on the curve at specified times
        pos_evals = self(t)
        # vel_evals = np.gradient(pos_evals, self.duration / len(pos_evals), axis=0)
        # accel_evals = np.gradient(vel_evals, self.duration / len(vel_evals), axis=0)
        vel_curve = self.derivative()
        vel_evals = vel_curve(t)
        accel_curve = vel_curve.derivative()
        accel_evals = accel_curve(t)
        # Plot the position, velocity, acceleration components on separate axes
        fig = plt.figure()
        subfigs = fig.subfigures(1, 3)
        left = subfigs[0].subplots(1, 3)
        middle = subfigs[1].subplots(1, 3)
        right = subfigs[2].subplots(1, 3)
        pos_labels = ["x", "y", "z"]
        vel_labels = ["vx", "vy", "vz"]
        accel_labels = ["ax", "ay", "az"]
        for i, ax in enumerate(left):
            ax.plot(pos_evals[:, i])
            ax.set_title(pos_labels[i])
        for i, ax in enumerate(middle):
            ax.plot(vel_evals[:, i])
            ax.set_title(vel_labels[i])
        for i, ax in enumerate(right):
            ax.plot(accel_evals[:, i])
            ax.set_title(accel_labels[i])
        plt.show()


# Based on the derivative function in Tobia's code
def bezier_deriv(pts, a, b):
    h = pts.shape[0] - 1
    return (pts[1:] - pts[:-1]) * (h / (b - a))


def test_bezier_traj():
    n_pts = 6
    M = n_pts - 1
    a = 0
    b = 5
    dim = 3
    # Arbitrary boundary conditions
    p0 = [0, 0, 0]
    v0 = [0.1, 0.2, 0.3]
    a0 = [0, 0, 0]
    pf = [1, 2, 3]
    vf = [0, 0, 0]
    af = [-0.3, -0.2, -0.1]
    pts = cp.Variable((n_pts, dim))
    d_pts = bezier_deriv(pts, a, b)
    d2_pts = bezier_deriv(d_pts, a, b)
    objective = cp.Minimize(0)  # TODO update this... just using feasibility for now
    constraints = [
        pts[0] == p0,
        pts[-1] == pf,
        d_pts[0] == v0,
        d_pts[-1] == vf,
        d2_pts[0] == a0,
        d2_pts[-1] == af,
    ]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    curve = BezierCurve(pts.value, a, b)
    curve.plot3d()


def test_plot_bernstein_polys():
    a = 0
    b = 10
    n = 50
    t = np.linspace(a, b, n, endpoint=True)
    M = 4
    fig = plt.figure()
    for n in range(M + 1):
        evals = bernstein(M, n, a, b, t)
        plt.plot(t, evals, label=str(n))
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # test_plot_bernstein_polys()
    test_bezier_traj()
