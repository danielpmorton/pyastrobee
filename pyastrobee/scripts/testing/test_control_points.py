"""Test script to view how the number of control points influences the "tightness" of the convex hull

Key takeaway:
- If we increase the number of control points, the hull will be a tighter bound on the curve
- This is critical for when we define limits on maximum velocity and acceleration, for instance
- If we set a limit on velocity but we're working with a very small number of control points, the hull may be
  not tight, and there might be no solution, since a control point might need to be far away from the actual curve
- So in general, we should find a balance between number of control points, and tightness of velocity/accel bounds
"""


import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

from pyastrobee.trajectories.bezier import BezierCurve, plot_1d_bezier_curve

# Some parameters to tune and see how it affects the curve / hull / solution validity
dim = 1
n_control_pts = 10
t0 = 0
tf = 1
p0 = 0
pf = 1
v0 = 0
vf = 0
a0 = 0
af = 0
# Set the max vel/accel to None if we want to remove those constraints
v_max = 3
a_max = 10
# Not worrying about safe sets for now
box = None


# Same optimization process from the Bezier trajectory function
pos_pts = cp.Variable((n_control_pts, dim))
pos_curve = BezierCurve(pos_pts, t0, tf)
vel_curve = pos_curve.derivative
vel_pts = vel_curve.points
accel_curve = vel_curve.derivative
accel_pts = accel_curve.points
jerk_curve = accel_curve.derivative
constraints = [pos_pts[0] == p0, pos_pts[-1] == pf]
if v0 is not None:
    constraints.append(vel_pts[0] == v0)
if vf is not None:
    constraints.append(vel_pts[-1] == vf)
if a0 is not None:
    constraints.append(accel_pts[0] == a0)
if af is not None:
    constraints.append(accel_pts[-1] == af)
if box is not None:
    lower, upper = box
    constraints.append(pos_pts >= lower)
    constraints.append(pos_pts <= upper)
if v_max is not None:
    constraints.append(cp.norm2(vel_pts, axis=1) <= v_max)
if a_max is not None:
    constraints.append(cp.norm2(accel_pts, axis=1) <= a_max)
jerk = jerk_curve.l2_squared
objective = cp.Minimize(jerk)
prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.CLARABEL)
assert prob.status == cp.OPTIMAL
solved_pos_curve = BezierCurve(pos_pts.value, t0, tf)
solved_vel_curve = solved_pos_curve.derivative
solved_accel_curve = solved_vel_curve.derivative

# Plotting
n = 50
plt.figure()
plot_1d_bezier_curve(solved_pos_curve, n, show=False)
plt.title("Position vs time")
plt.figure()
plot_1d_bezier_curve(solved_vel_curve, n, show=False)
plt.title("Velocity vs time")
plt.figure()
plot_1d_bezier_curve(solved_accel_curve, n, show=False)
plt.title("Acceleration vs time")
plt.show()
