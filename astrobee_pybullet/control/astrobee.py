import pybullet


class Astrobee:
    def __init__(
        self,
        pos,
        orn,
    ):
        # TODO need to finish this
        self.id = pybullet.loadURDF(
            "home/dan/astrobee_pybullet/astrobee_pybullet/resources/urdf/astrobee.urdf"
        )
        self.num_joints = pybullet.getNumJoints(self.id)
        pass

    def set_pose(self, pose):
        pass

    def get_pose(self):
        link_index = 0  # Should be the main body (TODO check)
        compute_velocity = False  # ??? Figure out a better way to use this
        compute_fwd_kin = False  # Use in another function?
        link_state = pybullet.getLinkState(
            self.id, link_index, compute_velocity, compute_fwd_kin
        )
        # Unpack link state (This unpacking assumes that we are NOT computing velocity)
        # If computing velocity, there are two additional outputs
        # Note that we probably want the link frame pos/orn rather than the center of mass (COM)
        (
            COM_world_pos,
            COM_world_orn,
            local_pos,
            local_orn,
            world_pos,
            world_orn,
        ) = link_state

        # NOT DONE
        # Should return a transformation matrix at the end of this
        raise NotImplementedError
        pass

    def open_gripper(self):
        pass

    def close_gripper(self):
        pass

    def is_near(self, pose, tol):
        pass

    @property
    def state(self):
        pass

    @state.setter
    def state(self, state):
        pass
