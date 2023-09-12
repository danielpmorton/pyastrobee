"""Description of the free space within the ISS as a collection of safe boxes"""

from pyastrobee.config.astrobee_geom import COLLISION_RADIUS
from pyastrobee.utils.boxes import Box, visualize_3D_box, compute_graph, contract_box
from pyastrobee.utils.algos import dfs

# TODO:
# Test cases:
# - Avoid 3-way intersections
# - Ensure that the graph is connected
# (Run these tests when accounting for the collision radius of the Astrobee)

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

# Full description of free space within the ISS (Manually calibrated from simulation)
FULL_SAFE_SET = {
    JPM: Box([3.200, -0.775, -1.000], [10.500, 0.912, 1.000]),
    NODE_2: Box([-1.000, -2.458, -1.100], [1.100, 1.091, 0.924]),
    EU_LAB: Box([-7.703, -0.866, -1.000], [-3.400, 0.925, 1.000]),
    US_LAB: Box([-1.000, -11.265, -1.000], [0.874, -5.806, 1.000]),
    NODE_1: Box([-0.952, -17.165, -1.000], [1.000, -13.449, 1.028]),
    NODE_3_A: Box([2.900, -15.589, -1.000], [6.322, -14.700, 0.588]),
    NODE_3_B: Box([4.958, -15.53, -1.000], [7.646, -14.107, 0.939]),
    CUPOLA: Box([5.900, -15.18, 1.367], [6.662, -14.285, 3.158]),
    JPM_N2_CORRIDOR: Box([0.100, -0.475, -0.600], [4.800, 0.475, 0.600]),
    N2_EU_CORRIDOR: Box([-4.800, -0.475, -0.600], [-0.100, 0.475, 0.600]),
    N2_US_CORRIDOR: Box([-0.600, -6.902, -0.600], [0.600, -1.348, 0.600]),
    US_N1_CORRIDOR: Box([-0.600, -14.350, -0.600], [0.600, -10.200, 0.600]),
    N1_N3_CORRIDOR: Box([-0.100, -15.400, -0.600], [4.18, -14.400, 0.600]),
    N3_CUPOLA_CORRIDOR: Box([5.800, -15.421, 0.119], [7.000, -14.100, 2.179]),
}

# Locations where the robot base can travel (the full safe set but shrunk by the collision radius)
ROBOT_SAFE_SET = {
    module: contract_box(box, COLLISION_RADIUS)
    for (module, box) in FULL_SAFE_SET.items()
}


def compute_iss_graph() -> dict[str, list[str]]:
    """Computes the graph between the safe sets in the ISS based on intersecting safe boxes

    Returns:
        dict[str, list[str]]: Adjacency list / graph dictating safe paths in the ISS. Key/value pair is:
            (name of the module) -> (list of names of all neighbors of that module)
    """
    # Use the robot safe set here because it gives a better description of motion
    # If we use the full safe set there could be some locations with 3-way intersections that don't present themselves
    # when planning for working with the robot
    return compute_graph(ROBOT_SAFE_SET)


def _show_iss_boxes():
    """Visualize the safe set in Pybullet"""
    import pybullet  # pylint: disable=import-outside-toplevel

    pybullet.connect(pybullet.GUI)
    for box in FULL_SAFE_SET.values():
        visualize_3D_box(box, rgba=(1, 0, 0, 0.5))
    for box in ROBOT_SAFE_SET.values():
        visualize_3D_box(box, rgba=(0, 0, 1, 0.5))
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
