"""Quick example of loading the Astrobee URDF with the rigid bag attached"""

import time
import pybullet

pybullet.connect(pybullet.GUI)
pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, False)
# Zoom in on Astrobee
pybullet.resetDebugVisualizerCamera(1, -60, -12.2, [-0.05, 0.03, -0.17])
robot = pybullet.loadURDF("pyastrobee/assets/urdf/astrobee_with_rigid_bag.urdf")
# Load textures
for link_id in range(-1, 8):
    pybullet.changeVisualShape(
        objectUniqueId=robot, linkIndex=link_id, rgbaColor=[1, 1, 1, 1]
    )
# Loop sim
while True:
    pybullet.stepSimulation()
    time.sleep(1 / 120)
