import pybullet
import numpy as np

from pyastrobee.core.astrobee import Astrobee
from pyastrobee.utils.transformations import invert_transform_mat
from pyastrobee.utils.dynamics import inertial_transformation


def main():
    pybullet.connect(pybullet.GUI)
    robot = Astrobee((1, 1, 1, 0, 0, 0, 1))
    T_B2W = robot.tmat  # Base to world
    inertia = np.zeros((3, 3))
    total_mass = 0
    for link in Astrobee.Links:
        info = pybullet.getDynamicsInfo(robot.id, link.value)
        mass = info[0]
        inertia_diagonal = info[2]
        total_mass += mass
        if link.value == -1:
            inertia += np.diag(inertia_diagonal)
        else:  # not base
            T_L2W = robot.get_link_transform(link.value)  # Link to world
            T_L2B = invert_transform_mat(T_B2W) @ T_L2W  # Link to base
            inertia += inertial_transformation(mass, np.diag(inertia_diagonal), T_L2B)
    print("Inertia tensor:\n", inertia)
    print("Total robot mass:\n", total_mass)


if __name__ == "__main__":
    main()
