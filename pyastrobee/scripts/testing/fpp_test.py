"""Script to directly test out Tobia's code (which includes knot retiming)"""

import numpy as np
import pybullet

# Git clone https://github.com/danielpmorton/fastpathplanning and pip install -e
# I haven't added this to the requirements yet since I don't know if I'll fully use it
import fastpathplanning as fpp

from pyastrobee.config.iss_safe_boxes import ALL_BOXES
from pyastrobee.utils.boxes import visualize_3D_box
from pyastrobee.utils.debug_visualizer import visualize_path


lowers = []
uppers = []
for name, box in ALL_BOXES.items():
    lowers.append(box.lower)
    uppers.append(box.upper)

L = np.array(lowers)
U = np.array(uppers)
S = fpp.SafeSet(L, U)

jpm_midpt = (
    np.array([3.542, -0.623, -0.739])
    + (np.array([10.242, 0.760, 0.749]) - np.array([3.542, -0.623, -0.739])) / 2
)
cupola_midpt = (
    np.array([6.140, -15.028, 1.648])
    + (np.array([6.371, -14.437, 2.877]) - np.array([6.140, -15.028, 1.648])) / 2
)
p_init = jpm_midpt
p_term = cupola_midpt
T = 1
alpha = [1, 1, 5]
p = fpp.plan(S, p_init, p_term, T, alpha)
t = np.linspace(0, 1, 50, endpoint=True)
p_t = p(t)

pybullet.connect(pybullet.GUI)
for box in ALL_BOXES.values():
    visualize_3D_box(box)
visualize_path(p_t, color=(0, 0, 1))
input()
