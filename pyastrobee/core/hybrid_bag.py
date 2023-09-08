"""Testing to see if we can attach a deformable handle to a rigid bag

WORK IN PROGRESS!!!!!!
"""

import time

import numpy as np

from pyastrobee.utils.bullet_utils import (
    initialize_pybullet,
    load_deformable_object,
    create_box,
    create_anchor,
)
from pyastrobee.utils.mesh_utils import get_closest_mesh_vertex, get_mesh_data
from pyastrobee.config.bag_properties import BOX_LENGTH, BOX_WIDTH, BOX_HEIGHT


handle_corner_pos_local = [
    # Outer corners
    (0.075, 0.025, 0),  # ID 32
    (0.075, -0.025, 0),  # ID 1
    (-0.075, 0.025, 0),  # ID 21
    (-0.075, -0.025, 0),  # ID 19
    # Inner corners
    (0.06138287, 0.025, 0),  # ID 31
    (0.06138287, -0.025, 0),  # ID 0
    (-0.06138287, 0.025, 0),  # ID 22
    (-0.06138287, -0.025, 0),  # ID 20
]


client = initialize_pybullet()
box = create_box(
    (0, 0, -BOX_HEIGHT / 2),
    (0, 0, 0, 1),
    5,
    (BOX_LENGTH, BOX_WIDTH, BOX_HEIGHT),
    True,
    client=client,
)
handle = load_deformable_object(
    "pyastrobee/assets/meshes/handle_only.vtk", pos=(0, 0, 0), mass=0.2, client=client
)
n_verts, vert_positions = get_mesh_data(handle, client=client)

ids = []
for i, pos in enumerate(handle_corner_pos_local):
    actual_pos, vert_id = get_closest_mesh_vertex(pos, vert_positions)
    print(f"Corner #{i}: ID = {vert_id}. Distance: {np.linalg.norm(pos - actual_pos)}")
    ids.append(vert_id)

anchor_ids = []
for id in ids:
    aid, _ = create_anchor(handle, id, box, -1)
    anchor_ids.append(aid)

while True:
    client.stepSimulation()
    time.sleep(1 / 120)
