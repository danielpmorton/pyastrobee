"""Quick script to confirm the relationship between Euler/Rodrigues parameters and quaternions

It appears that Euler/Rodrigues parameters are the same thing as WXYZ quaternions, which 
is excellent because I have a ton of useful equations for Euler/Rodrigues and can use these
with a simple permutation to convert to XYZW
"""

import numpy as np

from pyastrobee.archive.my_rotations import rmat_to_euler_params
from pyastrobee.utils.rotations import quat_to_rmat
from pyastrobee.utils.quaternions import xyzw_to_wxyz

# Create a random array of quaternions
m = 500
quats = np.random.rand(m, 4)
quats = quats / np.linalg.norm(quats, axis=1).reshape(-1, 1)

# Convert them into rotation matrices and then into euler/rodrigues params
rods = np.zeros_like(quats)
for i, quat in enumerate(quats):
    R = quat_to_rmat(quat)
    rods[i, :] = rmat_to_euler_params(R)

# Compare the quaternions to the rodrigues params
if np.allclose(xyzw_to_wxyz(quats), rods):
    print("WXYZ quaternions are the same thing as rodrigues params!")
