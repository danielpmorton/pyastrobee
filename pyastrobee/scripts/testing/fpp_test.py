"""Script to directly test out Tobia's Fast Path Planning code and see how it compares to ours"""

import time

import numpy as np

# Git clone https://github.com/danielpmorton/fastpathplanning and pip install -e
# I haven't added this to the requirements yet since I don't know if I'll fully use it
import fastpathplanning as fpp

from pyastrobee.core.iss import ISS
from pyastrobee.utils.bullet_utils import initialize_pybullet
from pyastrobee.config.iss_safe_boxes import ALL_BOXES, compute_iss_graph
from pyastrobee.utils.boxes import find_containing_box_name
from pyastrobee.utils.debug_visualizer import visualize_path, animate_path
from pyastrobee.utils.algos import dfs
from pyastrobee.trajectories.splines import spline_trajectory_with_retiming


def fpp_method(p0, pf, T):
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

    T = 10
    alpha = [0, 0, 1]  # [1, 1, 5]
    der_init = {1: [0, 0, 0], 2: [0, 0, 0]}
    der_term = {1: [0, 0, 0], 2: [0, 0, 0]}
    p = fpp.plan(S, p0, pf, T, alpha, der_init, der_term, verbose=False)
    t = np.linspace(0, T, 50, endpoint=True)
    p_t = p(t)

    # print the total jerk
    total_jerk = 0
    for bez in p.beziers:
        total_jerk += bez.derivative().derivative().derivative().l2_squared()
    print("COST (FPP): ", total_jerk)
    return p_t  # Positions in the trajectory


def my_method(p0, pf, T, max_retiming_iters):
    """Create a trajectory using my own trajectory generation method

    We can set the maximum retiming iterations to 0 if we want to see how the retiming
    afects the path shape
    """
    start = find_containing_box_name(p0, ALL_BOXES)
    end = find_containing_box_name(pf, ALL_BOXES)
    graph = compute_iss_graph()
    path = dfs(graph, start, end)
    box_path = [ALL_BOXES[p] for p in path]
    pos, *_ = spline_trajectory_with_retiming(
        p0,
        pf,
        0,
        T,
        50,
        8,
        box_path,
        np.ones(len(box_path)) * (T / len(box_path)),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        max_iters=max_retiming_iters,
    )
    return pos


def main():
    """Generate trajectories between the same two points using Tobia's method and mine"""

    start_pt = ALL_BOXES["jpm"].center
    end_pt = ALL_BOXES["cupola"].center
    T = 10

    time_a = time.time()
    pos_fpp = fpp_method(start_pt, end_pt, T)
    time_b = time.time()
    print("Time for Tobia's method: ", time_b - time_a)
    pos_mine_no_retiming = my_method(start_pt, end_pt, T, 0)
    time_c = time.time()
    print("Time for my method (with graph computation): ", time_c - time_b)
    pos_mine_retimed = my_method(start_pt, end_pt, T, 10)
    time_d = time.time()
    print("Time for my sequential method: ", time_d - time_c)

    client = initialize_pybullet()
    iss = ISS()
    iss.show_safe_set()
    visualize_path(pos_fpp, color=(0, 0, 1))
    visualize_path(pos_mine_no_retiming, color=(0, 1, 0))
    visualize_path(pos_mine_retimed, color=(1, 0, 0))
    input("Press Enter to animate path")
    animate_path(pos_fpp)
    animate_path(pos_mine_no_retiming)
    animate_path(pos_mine_retimed)
    input("Press Enter to finish")
    client.disconnect()


if __name__ == "__main__":
    main()
