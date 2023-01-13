"""https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/deformable_anchor.py"""

from time import sleep

import pybullet
import pybullet_data

physicsClient = pybullet.connect(pybullet.GUI)

pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
pybullet.resetSimulation(pybullet.RESET_USE_DEFORMABLE_WORLD)

gravZ = -10
pybullet.setGravity(0, 0, gravZ)

planeOrn = [0, 0, 0, 1]  # p.getQuaternionFromEuler([0.3,0,0])
# planeId = p.loadURDF("plane.urdf", [0,0,-2],planeOrn)

boxId = pybullet.loadURDF("cube.urdf", [0, 1, 2], useMaximalCoordinates=True)

clothId = pybullet.loadSoftBody(
    "cloth_z_up.obj",
    basePosition=[0, 0, 2],
    scale=0.5,
    mass=1.0,
    useNeoHookean=0,
    useBendingSprings=1,
    useMassSpring=1,
    springElasticStiffness=40,
    springDampingStiffness=0.1,
    springDampingAllDirections=1,
    useSelfCollision=0,
    frictionCoeff=0.5,
    useFaceContact=1,
)

pybullet.changeVisualShape(clothId, -1, flags=pybullet.VISUAL_SHAPE_DOUBLE_SIDED)

pybullet.createSoftBodyAnchor(clothId, 24, -1, -1)
pybullet.createSoftBodyAnchor(clothId, 20, -1, -1)
pybullet.createSoftBodyAnchor(clothId, 15, boxId, -1, [0.5, -0.5, 0])
pybullet.createSoftBodyAnchor(clothId, 19, boxId, -1, [-0.5, -0.5, 0])
pybullet.setPhysicsEngineParameter(sparseSdfVoxelSize=0.25)

debug = True
if debug:
    data = pybullet.getMeshData(clothId, -1, flags=pybullet.MESH_DATA_SIMULATION_MESH)
    print("--------------")
    print("data=", data)
    print(data[0])
    print(data[1])
    text_uid = []
    for i in range(data[0]):
        pos = data[1][i]
        uid = pybullet.addUserDebugText(str(i), pos, textColorRGB=[1, 1, 1])
        text_uid.append(uid)

while pybullet.isConnected():
    pybullet.getCameraImage(320, 200)

    if debug:
        data = pybullet.getMeshData(
            clothId, -1, flags=pybullet.MESH_DATA_SIMULATION_MESH
        )
        for i in range(data[0]):
            pos = data[1][i]
            uid = pybullet.addUserDebugText(
                str(i), pos, textColorRGB=[1, 1, 1], replaceItemUniqueId=text_uid[i]
            )

    pybullet.setGravity(0, 0, gravZ)
    pybullet.stepSimulation()
    # p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)
    # sleep(1./240.)
