"""It looks like there is a difference in "effective mass" between rigid and deformable bags of the same mass

So this script will just apply a constant 1 Newton force to the astrobee/bag system and see what the mass is


TODO figure this out... might need to be a "fudge factor" multiple of the deformable mass when constructing the rigid
analog for MPC
"""

# TODO decide how to handle this mass difference... "fudge factor" multiple when constructing the bags?

import time

import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial


from pyastrobee.utils.bullet_utils import initialize_pybullet
from pyastrobee.core.deformable_bag import DeformableCargoBag
from pyastrobee.core.constraint_bag import ConstraintCargoBag
from pyastrobee.core.astrobee import Astrobee

client = initialize_pybullet()
robot = Astrobee()
bag_mass = 10
bag = ConstraintCargoBag("top_handle", bag_mass)
bag.attach_to(robot, "bag")
force = (0, 0, 1)
dt = client.getPhysicsEngineParameters()["fixedTimeStep"]


def collect_data():
    duration = 5
    n_timesteps = int(duration / dt)
    pos_hist = np.zeros((n_timesteps, 3))
    for i in range(n_timesteps):
        client.applyExternalForce(robot.id, -1, force, (0, 0, 0), client.WORLD_FRAME)
        pos_hist[i] = client.getBasePositionAndOrientation(robot.id)[0]
        client.stepSimulation()
        time.sleep(1 / 120)
    return pos_hist


hist_rigid = collect_data()
bag.detach()
bag.unload()
client.resetBasePositionAndOrientation(robot.id, (0, 0, 0), (0, 0, 0, 1))
bag = DeformableCargoBag("top_handle", bag_mass)
bag.attach_to(robot, "bag")
hist_deformable = collect_data()
client.disconnect()

# Throw out a bit of data at the start since that's not as good
z_pos_rigid = hist_rigid[:, 2][int(1 / dt) :]
z_pos_deformable = hist_deformable[:, 2][int(1 / dt) :]
z_vel_rigid = np.gradient(z_pos_rigid, dt)
z_accel_rigid = np.gradient(z_vel_rigid, dt)
z_vel_deformable = np.gradient(z_pos_deformable, dt)
z_accel_deformable = np.gradient(z_vel_deformable, dt)

times = np.arange(len(z_pos_rigid)) * dt

pos_poly_rigid = Polynomial.fit(times, z_pos_rigid, 2)
pos_poly_deformable = Polynomial.fit(times, z_pos_deformable, 2)
vel_poly_rigid = pos_poly_rigid.deriv()
vel_poly_deformable = pos_poly_deformable.deriv()
accel_poly_rigid = vel_poly_rigid.deriv()
accel_poly_deformable = vel_poly_deformable.deriv()

accel_rigid = accel_poly_rigid.coef[0]
accel_deformable = accel_poly_deformable.coef[0]

# Use the second half of the simulation to determine the effective mass
# since the beginning can have weird effects
f_mag = np.linalg.norm(force)
rigid_system_mass = f_mag / accel_rigid
deformable_system_mass = f_mag / accel_deformable

print("Rigid mass: ", rigid_system_mass - robot.mass)
print("Deformable mass: ", deformable_system_mass - robot.mass)

plt.figure()
plt.plot(times, z_pos_rigid, label="Rigid")
plt.plot(times, z_pos_deformable, label="Deformable")
plt.plot(times, pos_poly_rigid(times), label="Rigid fit")
plt.plot(times, pos_poly_deformable(times), label="Deformable fit")
plt.legend()
plt.title("Z position vs time")
plt.figure()
plt.plot(times, z_vel_rigid, label="Rigid")
plt.plot(times, z_vel_deformable, label="Deformable")
plt.plot(times, vel_poly_rigid(times), label="Rigid fit")
plt.plot(times, vel_poly_deformable(times), label="Deformable fit")
plt.legend()
plt.title("Z velocity vs time")
plt.figure()
plt.plot(times, z_accel_rigid, label="Rigid")
plt.plot(times, z_accel_deformable, label="Deformable")
plt.plot(times, accel_poly_rigid(times), label="Rigid fit")
plt.plot(times, accel_poly_deformable(times), label="Deformable fit")
plt.legend()
plt.title("Z acceleration vs time")
plt.show()
