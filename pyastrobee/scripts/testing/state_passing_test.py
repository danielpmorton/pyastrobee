"""Testing to see if we can use pybullet's shared memory to pass saved states between sims

TODO add test between vectorized environments
"""

import time

import pybullet
import pybullet_data
from pybullet_utils.bullet_client import BulletClient

from pyastrobee.core.astrobee import Astrobee


def test_shared_memory():
    """Test to see if we can use pybullet's shared memory to pass saved states between sims

    Results:
    - This seems to not work the way we want it to...
    - It seems like loading objects in separate sims doesn't really work the way we'd want (it seems to get loaded
      all into the same sim?)
    - saveState seems to do something, but it isn't clear what
    - Also, this thread seems to indicate that the shared memory can only allow for one additional connection
      https://pybullet.org/Bullet/phpBB3/viewtopic.php?t=13353
    """
    # Start 1 process with GUI but use the SERVER variety to allow for shared memory
    client_1: pybullet = BulletClient(pybullet.GUI_SERVER)
    # Start a second process using shared memory
    client_2: pybullet = BulletClient(pybullet.SHARED_MEMORY)
    # Check to make sure that both are valid clients
    # The _client parameter is a part of the BulletClient class
    assert client_1._client >= 0
    assert client_2._client >= 0
    # Set up the sims identically
    robot_1 = Astrobee(client=client_1)
    robot_2 = Astrobee(client=client_2)
    # Interact with and then save the state from client 1
    duration = 5
    print(f"{duration} seconds to interact with the sim")
    start_time = time.time()
    while time.time() - start_time < duration:
        client_1.stepSimulation()
        time.sleep(1 / 120)
    input("Press Enter to save the state")
    state_id = pybullet.saveState()
    # These values should be different because we messed with the first simulation in the GUI
    print(
        f"Robot position/orientation in sim 1: {client_1.getBasePositionAndOrientation(robot_1.id)}"
    )
    print(
        f"Robot position/orientation in sim 2: {client_2.getBasePositionAndOrientation(robot_2.id)}"
    )
    input("Press Enter to restore the state in sim 2")
    # Use that state id to reset the state in sim 2
    client_2.restoreState(stateId=state_id)
    # These values should be the same since we've reset sim 2 to be the same as sim 1
    print(
        f"Robot position/orientation in sim 1: {client_1.getBasePositionAndOrientation(robot_1.id)}"
    )
    print(
        f"Robot position/orientation in sim 2: {client_2.getBasePositionAndOrientation(robot_2.id)}"
    )
    input("Done, press Enter to exit")


def test_gui_and_direct():
    """See if we can pass states between the standard GUI and DIRECT methods, without vec envs or shared memory

    Results:
    - This doesn't work either
    """
    client_1: pybullet = BulletClient(pybullet.GUI)
    client_2: pybullet = BulletClient(pybullet.DIRECT)
    client_1.setAdditionalSearchPath(pybullet_data.getDataPath())
    client_2.setAdditionalSearchPath(pybullet_data.getDataPath())
    # Save the state in the first client
    state_id_1 = client_1.saveState()
    # Try to reset the state in the second client
    try:
        # This restoreState call will likely throw an error if the state is not in memory for sim 2
        client_2.restoreState(state_id_1)
        while True:
            client_1.stepSimulation()
            client_2.stepSimulation()
            time.sleep(1 / 120)
    except pybullet.error as exc:
        raise Exception("Test failed") from exc


if __name__ == "__main__":
    test_shared_memory()
    # test_gui_and_direct()
