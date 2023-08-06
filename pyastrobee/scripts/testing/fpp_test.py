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
from pyastrobee.trajectories.splines import optimal_spline_trajectory


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


def my_method(p0, pf, T):
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
    pos, *_ = optimal_spline_trajectory(
        p0,
        pf,
        0,
        T,
        8,
        50,
        box_path,
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        max_iters=0,
    )
    return pos


def my_sequential_method(p0, pf, T):
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
    pos, *_ = optimal_spline_trajectory(
        p0,
        pf,
        0,
        T,
        8,
        50,
        box_path,
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        # max_iters=0,
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
    pos_mine = my_method(start_pt, end_pt, T)
    time_c = time.time()
    print("Time for my method (with graph computation): ", time_c - time_b)
    pos_my_seq = my_sequential_method(start_pt, end_pt, T)
    time_d = time.time()
    print("Time for my sequential method: ", time_d - time_c)

    client = initialize_pybullet()
    iss = ISS()
    iss.show_safe_set()
    visualize_path(pos_fpp, color=(0, 0, 1))
    visualize_path(pos_mine, color=(0, 1, 0))
    visualize_path(pos_my_seq, color=(1, 0, 0))
    input("Press Enter to animate path")
    animate_path(pos_fpp)
    animate_path(pos_mine)
    animate_path(pos_my_seq)


def animate_path(positions):
    import pybullet
    from pyastrobee.utils.bullet_utils import create_sphere
    from pyastrobee.utils.debug_visualizer import visualize_points

    # sphere = create_sphere(positions[0], 1, 0.2, False, (1, 1, 1, 1))
    # cid = pybullet.createConstraint(
    #     sphere, -1, -1, -1, pybullet.JOINT_FIXED, None, (0, 0, 0), (0, 0, 0)
    # )
    dt = 1 / 30
    for position in positions:
        visualize_points([position], (1, 1, 1), lifetime=2 * dt)
        # pybullet.changeConstraint(cid, position, (0, 0, 0, 1))
        pybullet.stepSimulation()
        time.sleep(dt)
    # pybullet.removeConstraint(cid)
    # pybullet.removeBody(sphere)


if __name__ == "__main__":
    main()
