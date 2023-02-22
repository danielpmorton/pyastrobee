"""Inertial parameters for the astrobee

Sourced from:
- A Brief Guide to Astrobee

TODO
- Decide if these are even useful? (might be redundant with the URDF values)
- Add more parameters
"""

import numpy as np

MASS = 9.58  # kg
INERTIA = np.diag([0.153, 0.143, 0.162])  # kg-m^2
