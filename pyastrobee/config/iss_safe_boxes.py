"""Description of the free space within the ISS as a collection of safe boxes"""

from collections import defaultdict

from pyastrobee.utils.boxes import Box, visualize_3D_box, check_box_intersection
from pyastrobee.utils.algos import dfs

# Names of the safe boxes in the ISS
# Originally these were an enum but this made indexing in the graph search weird
# Strings are easier so we'll assign them to constants here just for reduced chance of error
JPM = "jpm"
NODE_2 = "node_2"
EU_LAB = "eu_lab"
US_LAB = "us_lab"
NODE_1 = "node_1"
# Node 3 has two boxes since there is a toilet blocking an area
NODE_3_A = "node_3_a"
NODE_3_B = "node_3_b"
CUPOLA = "cupola"
# Corridors between modules
JPM_N2_CORRIDOR = "jpm_n2_corridor"
N2_EU_CORRIDOR = "n2_eu_corridor"
N2_US_CORRIDOR = "n2_us_corridor"
US_N1_CORRIDOR = "us_n1_corridor"
N1_N3_CORRIDOR = "n1_n3_corridor"
N3_CUPOLA_CORRIDOR = "n3_cupola_corridor"

# Calibrated values from simulation
ALL_BOXES = {
    JPM: Box([3.542, -0.623, -0.739], [10.242, 0.760, 0.749]),
    NODE_2: Box([-0.761, -2.306, -0.860], [0.831, 0.939, 0.643]),
    EU_LAB: Box([-7.412, -0.714, -0.775], [-3.770, 0.773, 0.827]),
    US_LAB: Box([-0.745, -11.113, -0.745], [0.583, -5.958, 0.756]),
    NODE_1: Box([-0.661, -17.013, -0.832], [0.829, -13.601, 0.747]),
    NODE_3_A: Box([3.025, -15.437, -0.772], [6.031, -14.932, 0.307]),
    NODE_3_B: Box([5.249, -15.378, -0.774], [7.355, -14.259, 0.658]),
    CUPOLA: Box([6.140, -15.028, 1.648], [6.371, -14.437, 2.877]),
    JPM_N2_CORRIDOR: Box([0.25, -0.338, -0.354], [4.5, 0.325, 0.413]),
    N2_EU_CORRIDOR: Box([-4.5, -0.338, -0.354], [-0.25, 0.325, 0.413]),
    N2_US_CORRIDOR: Box([-0.410, -6.75, -0.402], [0.368, -1.5, 0.440]),
    US_N1_CORRIDOR: Box([-0.410, -14.2, -0.402], [0.368, -10.5, 0.440]),
    N1_N3_CORRIDOR: Box([0, -15.179, -0.396], [3.889, -14.515, 0.267]),
    N3_CUPOLA_CORRIDOR: Box([6.021, -15.269, 0.4], [6.771, -14.337, 1.898]),
}


def compute_iss_graph() -> dict[str, list[str]]:
    """Computes the graph between the safe sets in the ISS based on intersecting safe boxes

    Returns:
        dict[str, list[str]]: Adjacency list / graph dictating safe paths in the ISS
    """
    names = list(ALL_BOXES.keys())
    n = len(names)
    adj = defaultdict(list)
    for i in range(n):
        for j in range(i + 1, n):
            if check_box_intersection(ALL_BOXES[names[i]], ALL_BOXES[names[j]]):
                adj[names[i]].append(names[j])
                adj[names[j]].append(names[i])
    return adj


def _show_iss_boxes():
    """Visualize the safe set in Pybullet"""
    import pybullet  # pylint: disable=import-outside-toplevel

    pybullet.connect(pybullet.GUI)
    padding = None  # np.array([0.290513, 0.151942, 0.281129]) / 2
    for box in ALL_BOXES.values():
        visualize_3D_box(box, padding)
    input("Press Enter to exit")
    pybullet.disconnect()


def _test_graph_search():
    """Example of computing the path between boxes in the ISS"""
    graph = compute_iss_graph()
    start = JPM
    end = CUPOLA
    path = dfs(graph, start, end)
    print(f"Box sequence between {start} and {end}: ")
    print(path)


if __name__ == "__main__":
    _test_graph_search()
    _show_iss_boxes()
