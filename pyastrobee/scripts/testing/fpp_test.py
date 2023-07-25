"""Script to directly test out Tobia's code (which includes knot retiming)"""

import time

import numpy as np

# Git clone https://github.com/danielpmorton/fastpathplanning and pip install -e
# I haven't added this to the requirements yet since I don't know if I'll fully use it
import fastpathplanning as fpp

from pyastrobee.core.iss import ISS
from pyastrobee.utils.bullet_utils import initialize_pybullet
from pyastrobee.config.iss_safe_boxes import ALL_BOXES, compute_iss_graph
from pyastrobee.utils.boxes import find_containing_box
from pyastrobee.utils.debug_visualizer import visualize_path
from pyastrobee.utils.algos import dfs
from pyastrobee.trajectories.splines import spline_trajectory


def fpp_method(p0, pf):
    """Create an example trajectory using exactly the structure from fastpathplanning"""

    # Reorganize the safe set structure into the same L, U matrices from the paper
    lowers = []
    uppers = []
    for name, box in ALL_BOXES.items():
        lowers.append(box.lower)
        uppers.append(box.upper)

    L = np.array(lowers)
    U = np.array(uppers)
    S = fpp.SafeSet(L, U, verbose=False)

    T = 1
    alpha = [1, 1, 5]
    p = fpp.plan(S, p0, pf, T, alpha, verbose=False)
    t = np.linspace(0, 1, 50, endpoint=True)
    p_t = p(t)

    return p_t  # Positions in the trajectory


def my_method(p0, pf):
    names = []
    boxes = []
    for name, box in ALL_BOXES.items():
        names.append(name)
        boxes.append(box)
    start_id = find_containing_box(p0, boxes)
    end_id = find_containing_box(pf, boxes)
    graph = compute_iss_graph()
    path = dfs(graph, names[start_id], names[end_id])
    box_path = [ALL_BOXES[p] for p in path]  # TODO improve this
    pos, vel, accel, times = spline_trajectory(
        p0,
        pf,
        0,
        1,
        6,
        50,
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        box_path,
    )
    return pos


def main():
    """Generate trajectories between the same two points using Tobia's method and mine"""
    jpm_midpt = (
        np.array([3.542, -0.623, -0.739])
        + (np.array([10.242, 0.760, 0.749]) - np.array([3.542, -0.623, -0.739])) / 2
    )
    cupola_midpt = (
        np.array([6.140, -15.028, 1.648])
        + (np.array([6.371, -14.437, 2.877]) - np.array([6.140, -15.028, 1.648])) / 2
    )

    time_a = time.time()
    pos_fpp = fpp_method(jpm_midpt, cupola_midpt)
    time_b = time.time()
    pos_mine = my_method(jpm_midpt, cupola_midpt)
    time_c = time.time()
    print("Time for Tobia's method: ", time_b - time_a)
    print("Time for my method (with graph computation): ", time_c - time_b)

    client = initialize_pybullet()
    iss = ISS()
    iss.show_safe_set()
    visualize_path(pos_fpp, color=(0, 0, 1))
    visualize_path(pos_mine, color=(0, 1, 0))
    input()


if __name__ == "__main__":
    main()
