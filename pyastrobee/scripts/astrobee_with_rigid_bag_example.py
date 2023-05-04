"""Quick example of loading the Astrobee URDF with the rigid bag attached"""

import time
import pybullet

pybullet.connect(pybullet.GUI)
pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, False)
# Zoom in on Astrobee
pybullet.resetDebugVisualizerCamera(1, -60, -12.2, [-0.05, 0.03, -0.17])
# Load the URDF with the bag included
robot = pybullet.loadURDF("pyastrobee/assets/urdf/astrobee_with_rigid_bag.urdf")
# Loop sim
while True:
    pybullet.stepSimulation()
    time.sleep(1 / 240)
