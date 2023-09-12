"""Constants associated with the geometry of the Astrobee"""

import numpy as np

# This is useful for modeling drag forces
CROSS_SECTION_AREA = 0.092903

# Defines the bounding sphere for collision modeling. Based on URDF collision geometry
COLLISION_RADIUS = np.linalg.norm([0.319199 / 2, 0.117546 + 0.083962 / 2, 0.319588 / 2])
