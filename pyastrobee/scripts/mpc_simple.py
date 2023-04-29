#
# A simple MPC example with minimal dependencies.
#
# All units are SI: meters, seconds, etc.
# Quaternions are in [x,y,z,w] format (pybullet default).
#
# @contactrika, @imitsioni
#
"""

A demo with a trivial piece of cloth (no hole):
python -m pyastrobee.scripts.mpc_simple --cloth_obj cloth_z_up.obj \
  --cloth_scale 0.1 --cloth_init_ori 1.57 0 0 --do_perfect_model_rollouts


OLD INFO:

A default demo with a small T-Shirt:
python -m pyastrobee.scripts.mpc_simple

A demo with a generated piece of cloth:
python -m rl_top_euc.mpc_simple \
    --cloth_obj generated_cloth/generated_cloth.obj \
    --cloth_scale 1.5 --cloth_init_ori 0 0 1.57 \
    --cloth_init_hole_pos_offset 0.03 0.0 -0.10

A demo with the generated piece of cloth and an MPC using the perfect rollouts.
The MPC uses the perfect rollout to calculate the forward dynamics and plan for actions
STPS_PER_WPT ahead in the future. The controller executes the first MPC_STPS steps of
the optimal action and then re-plans.

python -m rl_top_euc.mpc_simple \
    --cloth_obj generated_cloth/generated_cloth.obj \
    --cloth_scale 1.5 \
    --cloth_init_ori 0 0 1.57 \
    --cloth_init_hole_pos_offset 0.03 0.0 -0.10 \
    --do_perfect_model_rollouts

~~~~~ Configs that kinda work for different items
- Purse bag

python -m rl_top_euc.mpc_simple \
    --cloth_obj bags/ts_purse_bag_resampled.obj \
    --cloth_init_ori  0  0  1.57 \
    --cloth_init_hole_pos_offset 0.0  0.0 -0.065 \
    --cloth_scale 1.5 \
    --do_perfect_model_rollouts \
    --reset_angular_vel

- Small bag

 python -m rl_top_euc.mpc_simple \
    --cloth_obj bags/ts_small_bag_resampled.obj \
    --cloth_init_ori 0 0 1.57 \
    --do_perfect_model_rollouts \
    --cloth_init_pos_offset 0 0.0 -0.1 \
    --anchor_init_pos -0.05 0.5 0.65 \
    --other_anchor_init_pos 0.05 0.5 0.65 \
    --cloth_init_hole_pos_offset -0.0  0.0 -0.05\
    --reset_angular_vel

- Backpack (it technically needs to have the hook a bit higher up)

 python -m rl_top_euc.mpc_simple \
    --cloth_obj bags/ts_backpack_resampled.obj \
    --do_perfect_model_rollouts \
    --cloth_init_pos_offset -0.12 0.08 -0.22 \
    --anchor_init_pos 0.05 0.5 0.8 \
    --other_anchor_init_pos -0.05 0.5 0.8 \
    --cloth_init_hole_pos_offset -0.0  0.0 -0.05\
    --reset_angular_vel

- Apron

 python -m rl_top_euc.mpc_simple \
    --cloth_obj cloth/ts_apron_oneloop.obj \
    --cloth_init_ori 0 0 1.57 \
    --cloth_scale 0.75 \
    --do_perfect_model_rollouts \
    --cloth_init_pos_offset 0 0.0 -0.15 \
    --cloth_init_hole_pos_offset 0.0 0.0 -0 \
    --anchor_init_pos -0.05 0.5 0.7 \
    --other_anchor_init_pos 0.05 0.5 0.7 \
    --reset_angular_vel

Use --do_perfect_model_rollouts to try a way simply way to fake a perfect model
by running forward simulation, then resetting to the last state when done.
WARNING: Pybullet does not support saving/reloading the state of deformables.
So this option is only here for very initial debugging. It will not actually
let us use simulation as a perfect model, but can be useful for initial tuning.

"""

import argparse
import os

import numpy as np
import pybullet
import pybullet_data
import pybullet_utils.bullet_client as bclient

# NOTE: NO IMPORTS FROM OUR CODE
# ONLY COPY PASTE THE MINIMAL AMOUNT OF CODE SO THAT IT IS TRIVIAL TO SEE
# WHERE AND HOW EVERYTHING IS INITIALIZED.

# Gains and limits for a simple PD controller for the anchors.
CTRL_MAX_FORCE = 10
CTRL_PD_KP = 10.0
CTRL_PD_KD = 5.0
STPS_PER_WPT = 50
MPC_STPS = 25
MAX_STPS = 300


def get_args():
    parser = argparse.ArgumentParser(description="MPC")
    parser.add_argument('--sim_frequency', type=int, default=500,
                        help='Number of simulation steps per second')  # 250-1K
    parser.add_argument('--cloth_obj', type=str,
                        default='cloth/tshirt_small.obj',
                        help='Obj file for cloth item')
    parser.add_argument('--cloth_init_pos_offset', type=float, nargs=3,
                        default=[0, 0, -0.10],
                        help='Offset for the the center of the cloth object'
                             'relative to the midpoint between the anchors')
    parser.add_argument('--cloth_init_ori', type=float, nargs=3,
                        default=[0, 0, 0],
                        help='Initial orientation for cloth (in Euler angles)')
    parser.add_argument('--cloth_scale', type=float, default=1.0,
                        help='Scaling for the cloth object')
    parser.add_argument('--cloth_init_hole_pos_offset', type=float, nargs=3,
                        default=[0.0, 0.0, 0.0],
                        help='Initial offset for the center of the cloth hole'
                             'relative to the midpoint between the anchors')
    parser.add_argument('--cloth_bending_stiffness', type=float, default=30.0,
                        help='Cloth spring elastic stiffness (k)')  # 1.0-300.0
    parser.add_argument('--cloth_damping_stiffness', type=float, default=1.0,
                        help='Cloth spring damping stiffness (c)')
    parser.add_argument('--cloth_elastic_stiffness', type=float, default=30.0,
                        help='Cloth spring elastic stiffness (k)')  # 1.0-300.0
    parser.add_argument('--cloth_friction_coeff', type=float, default=0.4,
                        help='Cloth friction coefficient')
    parser.add_argument('--cloth_texture', type=str,
                        default='textures/blue_bright.png',
                        help='Texture png for cloth item')
    parser.add_argument('--anchor_init_pos', type=float, nargs=3,
                        default=[-0.04, 0.45, 0.70],
                        help='Initial position for an anchor')
    parser.add_argument('--other_anchor_init_pos', type=float, nargs=3,
                        default=[0.04, 0.45, 0.70],
                        help='Initial position for another anchors')
    parser.add_argument('--do_perfect_model_rollouts', action='store_true',
                        help='Enable for forward rollouts'
                             'WARNING: no way to reload the state well')
    parser.add_argument('--reset_angular_vel', action='store_true',
                        help='Enable to for zeroing anchor angular vels')
    args = parser.parse_args()
    return args


def get_closest(pt, vertices, max_dist=None, num_pins_per_pt=None):
    # Returns a list of vertex ids for mesh vertices
    # that are closest the given 3D points specified by pt.
    pt = np.array(pt).reshape(1, 3)
    vertices = np.array(vertices)
    if num_pins_per_pt is None:
        num_pins_per_pt = max(1, vertices.shape[0] // 50)
    num_to_pin = min(vertices.shape[0], num_pins_per_pt)
    dists = np.linalg.norm(vertices - pt, axis=1)
    to_pin_ids = np.argpartition(dists, num_to_pin)[0:num_to_pin]
    if max_dist is not None:
        to_pin_ids = to_pin_ids[dists[to_pin_ids] <= max_dist]
    return to_pin_ids


def create_trajectory(waypoints, steps_per_waypoint, frequency):
    # Creates a trajectory through the given 3D waypoints.
    assert (len(waypoints) == len(steps_per_waypoint))
    num_wpts = len(waypoints)
    tot_steps = sum(steps_per_waypoint[:-1])
    dt = 1.0 / frequency
    traj = np.zeros([tot_steps, 3 + 3])  # 3D pos , 3D vel
    prev_pos = waypoints[0]  # start at the 0th waypoint
    t = 0
    for wpt in range(1, num_wpts):
        tgt_pos = waypoints[wpt]
        dur = steps_per_waypoint[wpt - 1]
        Y, Yd, Ydd = plan_min_jerk_trajectory(prev_pos, tgt_pos, dur * dt, dt)
        traj[t:t + dur, 0:3] = Y[:]
        traj[t:t + dur, 3:6] = Yd[:]  # vel
        # traj[t:t+dur,6:9] = Ydd[:]  # acc
        t += dur
        prev_pos = tgt_pos
    if t < tot_steps: traj[t:, :] = traj[t - 1, :]  # set rest to last entry
    return traj


def create_anchor(sim, pos, mass, radius, rgba=(1, 0, 1, 1.0)):
    # Create a small visual object at the provided pos in world coordinates.
    # If mass==0: this object does not collide with any other objects
    # and only serves to show grip location.
    # input: sim (pybullet sim), pos (list of 3 coords for anchor in world frame)
    # output: anchorId (long) --> unique bullet ID to refer to the anchor object
    anchorVisualShape = sim.createVisualShape(
        pybullet.GEOM_SPHERE, radius=radius * 1.5, rgbaColor=rgba)
    if mass > 0:
        anchorCollisionShape = sim.createCollisionShape(
            pybullet.GEOM_SPHERE, radius=radius)
    else:
        anchorCollisionShape = -1
    kwargs = {}
    if hasattr(pybullet, 'MESH_DATA_SIMULATION_MESH'):  # 3.+ pybullet version
        kwargs['useMaximalCoordinates'] = True
    anchorId = sim.createMultiBody(baseMass=mass, basePosition=pos,
                                   baseCollisionShapeIndex=anchorCollisionShape,
                                   baseVisualShapeIndex=anchorVisualShape,
                                   **kwargs)
    return anchorId


def calculate_min_jerk_step(y_curr, yd_curr, ydd_curr, goal, rem_dur, dt):
    # Computes y, yd, ydd for a step on min jerk trajectory segment.
    # Code from https://github.com/contactrika/bo-svae-dc/blob/master/
    # gym-bullet-extensions/gym_bullet_extensions/control/control_util.py#L77
    if rem_dur < 0:
        return

    if dt > rem_dur:
        dt = rem_dur

    t1 = dt
    t2 = t1 * dt
    t3 = t2 * dt
    t4 = t3 * dt
    t5 = t4 * dt

    dur1 = rem_dur
    dur2 = dur1 * rem_dur
    dur3 = dur2 * rem_dur
    dur4 = dur3 * rem_dur
    dur5 = dur4 * rem_dur

    dist = goal - y_curr
    a1t2 = 0.0  # goaldd
    a0t2 = ydd_curr * dur2
    v1t1 = 0.0  # goald
    v0t1 = yd_curr * dur1

    c1 = (6.0 * dist + (a1t2 - a0t2) / 2.0 - 3.0 * (v0t1 + v1t1)) / dur5
    c2 = (-15.0 * dist + (3.0 * a0t2 - 2.0 * a1t2) /
          2.0 + 8.0 * v0t1 + 7.0 * v1t1) / dur4
    c3 = (10.0 * dist + (a1t2 - 3.0 * a0t2) /
          2.0 - 6.0 * v0t1 - 4.0 * v1t1) / dur3
    c4 = ydd_curr / 2.0
    c5 = yd_curr
    c6 = y_curr

    y = c1 * t5 + c2 * t4 + c3 * t3 + c4 * t2 + c5 * t1 + c6
    yd = 5 * c1 * t4 + 4 * c2 * t3 + 3 * c3 * t2 + 2 * c4 * t1 + c5
    ydd = 20 * c1 * t3 + 12 * c2 * t2 + 6 * c3 * t1 + 2 * c4

    return y, yd, ydd


def plan_min_jerk_trajectory(y0, goal, dur, dt):
    # Creates a trajectory that starts at y0 and approaches the given goal.
    # Code from https://github.com/contactrika/bo-svae-dc/blob/master/
    # gym-bullet-extensions/gym_bullet_extensions/control/control_util.py#L59
    N = int(dur / dt)
    nDim = np.shape(y0)[0]
    Y = np.zeros((N, nDim))
    Yd = np.zeros((N, nDim))
    Ydd = np.zeros((N, nDim))
    Y[0, :] = y0
    rem_dur = dur
    for n in range(1, N):
        y_curr = Y[n - 1, :]
        yd_curr = Yd[n - 1, :]
        ydd_curr = Ydd[n - 1, :]
        for d in range(nDim):
            Y[n, d], Yd[n, d], Ydd[n, d] = calculate_min_jerk_step(
                y_curr[d], yd_curr[d], ydd_curr[d], goal[d], rem_dur, dt)
        rem_dur = rem_dur - dt
    return Y, Yd, Ydd


def pd_control_force(anc_pos, anc_linvel, tgt_pos_vel):
    pos_diff = tgt_pos_vel[0:3] - np.array(anc_pos)
    vel_diff = tgt_pos_vel[3:6] - np.array(anc_linvel)
    force = CTRL_PD_KP * pos_diff + CTRL_PD_KD * vel_diff
    force = np.clip(force, -1.0 * CTRL_MAX_FORCE, CTRL_MAX_FORCE)
    return force


def viz_cross(sim, pos, rgb=(0, 1, 0), delta=0.02):
    sim.addUserDebugLine(pos - np.array([delta, 0, 0]),
                         pos + np.array([delta, 0, 0]),
                         lineColorRGB=rgb, lifeTime=10)
    sim.addUserDebugLine(pos - np.array([0, delta, 0]),
                         pos + np.array([0, delta, 0]),
                         lineColorRGB=rgb, lifeTime=10)


def get_hole_center(sim, cloth_id, cloth_hole_vertex_ids, rgb=(0, 1, 0)):
    # Returns the approximate center of the cloth_hole_vertex_ids positions.
    # Draws a green line through the vertices that are close to the hole center.
    kwargs = {}
    if hasattr(pybullet, 'MESH_DATA_SIMULATION_MESH'):
        kwargs['flags'] = pybullet.MESH_DATA_SIMULATION_MESH
    num_mesh_vertices, mesh_vetex_positions = sim.getMeshData(cloth_id, **kwargs)
    centroid_sum = np.array([0.0, 0.0, 0.0])
    for v in cloth_hole_vertex_ids:
        v_pos = mesh_vetex_positions[v]
        centroid_sum += np.array(v_pos)
        # viz_cross(sim, v_pos, rgb=rgb)
    centroid_pos = centroid_sum / len(cloth_hole_vertex_ids)
    viz_cross(sim, centroid_pos, rgb)
    return centroid_pos


def viz_traj_deviation(sim, anchor_ids, trajs, step, rgb=(1, 0, 0)):
    for anc_i in range(len(anchor_ids)):
        anc_pos, anc_quat = sim.getBasePositionAndOrientation(anchor_ids[anc_i])
        tgt_pos = trajs[anc_i][step][0:3]
        sim.addUserDebugLine(anc_pos, tgt_pos, lineColorRGB=rgb, lifeTime=0)


def perfect_model_rollout(sim, anchor_ids, cloth_id, cloth_hole_vertex_ids, reset_angular_vel,
                          trajs, outer_step, max_steps=200):
    # A simple rollout with a fixed action.
    # Freezes and records full current simulation state; then rolls out a
    # trajectory for max_steps while applying force_action to the anchors.
    # Records the hole_center at the end of the rollout,
    # then restores to the original state.
    # This function should be useful for faking the output of a perfect model.
    print('Start perfect model rollout')
    # input('Press Enter to continue')
    max_traj_step = trajs[0].shape[0]
    for local_step in range(max_steps):
        step = np.clip(outer_step + local_step, 0, max_traj_step - 1)
        # print(local_step, step)
        for anc_i in range(len(anchor_ids)):
            tgt_pos_vel = trajs[anc_i][step]
            anc_pos, anc_quat = sim.getBasePositionAndOrientation(anchor_ids[anc_i])
            anc_linvel, anc_angvel = sim.getBaseVelocity(anchor_ids[anc_i])
            for anc_i in range(len(anchor_ids)):
                # if reset_angular_vel:
                    # sim.resetBaseVelocity(anchor_ids[anc_i], linearVelocity=anc_linvel, angularVelocity=[0, 0, 0])
                sim.resetBaseVelocity(
                   anchor_ids[anc_i], linearVelocity=tgt_pos_vel[3:6].tolist(),
                   angularVelocity=[0, 0, 0])
                # sim.applyExternalForce(  # apply desired force to the anchors
                #     anchor_ids[anc_i], -1,
                #     pd_control_force(anc_pos, anc_linvel, tgt_pos_vel).tolist(),
                #     [0, 0, 0], pybullet.LINK_FRAME)
        sim.stepSimulation()  # this advances physics sim (must have this)
        viz_traj_deviation(sim, anchor_ids, trajs, step, rgb=(1, 1, 0))
    hole_center = get_hole_center(sim, cloth_id, cloth_hole_vertex_ids,
                                  rgb=(0.5, 0.5, 0.5))
    print('Rollout done')
    # input('Press Enter to continue')
    return hole_center


def simple_init(sim, args, anchor_init_pos, other_anchor_init_pos):
    # Create pybullet client. This is the main simulator object.
    if sim is None:
        sim = bclient.BulletClient(connection_mode=pybullet.GUI)
    else:
        sim.resetSimulation()

    # Set up the data path that will automatically point to
    # gym-bullet-deform/gym_bullet_deform/data regardless of runtime directory.
    args.data_path = os.path.join(os.path.split(__file__)[0], '..', 'assets', 'simple')

    # Set up  basic simulation parameters.
    sim.resetDebugVisualizerCamera(
        cameraDistance=1.2, cameraYaw=80.0, cameraPitch=-60,
        cameraTargetPosition=[0, 0, 0])
    sim.resetSimulation() # pybullet.RESET_USE_DEFORMABLE_WORLD) # don't want FEM
    sim.setGravity(0, 0, -9.8)
    sim.setTimeStep(1.0 / args.sim_frequency)

    # Load floor plane and rigid objects.
    sim.setAdditionalSearchPath(pybullet_data.getDataPath())
    floor_id = sim.loadURDF('plane.urdf')
    yofst = -0.15
    sim.setAdditionalSearchPath(os.path.join(args.data_path))
    print('path', args.data_path)
    cuboid_id = sim.loadURDF(
        'cuboid.urdf', [0.0, yofst, 0.2], useFixedBase=1)
    hook_quat = pybullet.getQuaternionFromEuler([0, 0, np.pi / 2])
    goal_pos = [0.00, (0.3 + 0.1) / 2 + yofst, 0.30]
    hook_id = sim.loadURDF('hook.urdf', goal_pos, hook_quat, useFixedBase=1)

    # Load cloth object.
    # print('Loading cloth from', os.path.join(args.data_path, args.cloth_obj))
    anc_mid = (np.array(anchor_init_pos) + np.array(other_anchor_init_pos)) / 2.0
    cloth_init_pos = np.array(args.cloth_init_pos_offset) + anc_mid
    cloth_id = sim.loadSoftBody(
        mass=2.0,  # 1kg is default; bad sim with lower mass
        fileName=os.path.join(args.data_path, args.cloth_obj),
        basePosition=cloth_init_pos,
        baseOrientation=pybullet.getQuaternionFromEuler(args.cloth_init_ori),
        scale=args.cloth_scale,
        springElasticStiffness=args.cloth_elastic_stiffness,
        springDampingStiffness=args.cloth_damping_stiffness,
        springBendingStiffness=args.cloth_bending_stiffness,
        frictionCoeff=args.cloth_friction_coeff,
        collisionMargin=0.05, useSelfCollision=1,
        springDampingAllDirections=1, useFaceContact=1,
        useNeoHookean=0, useMassSpring=1, useBendingSprings=1)
    #texture_file_name = os.path.join(args.data_path, args.cloth_texture)
    #texture_id = sim.loadTexture(texture_file_name)
    #kwargs = {}
    #if hasattr(pybullet, 'VISUAL_SHAPE_DOUBLE_SIDED'):
    #    kwargs['flags'] = pybullet.VISUAL_SHAPE_DOUBLE_SIDED
    #sim.changeVisualShape(
    #    cloth_id, -1, textureUniqueId=texture_id, **kwargs)

    # Loading done, turn on visualizer
    sim.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)
    sim.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)  # turn off extras
    sim.configureDebugVisualizer(pybullet.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
    sim.configureDebugVisualizer(pybullet.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
    sim.configureDebugVisualizer(pybullet.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

    # Create anchors that can pull the cloth along trajectories.
    anchor_pos_list = [anchor_init_pos, other_anchor_init_pos]
    kwargs = {}
    if hasattr(pybullet, 'MESH_DATA_SIMULATION_MESH'):
        kwargs['flags'] = pybullet.MESH_DATA_SIMULATION_MESH
    num_mesh_vertices, mesh_vetex_positions = sim.getMeshData(cloth_id, **kwargs)
    anchor_ids = []
    for anchor_pos in anchor_pos_list:
        # This is where the anchor object is created. It is a simple rigid body
        # that is a sphere; can change its mass and radius below.
        anchor_id = create_anchor(sim, anchor_pos, mass=0.1, radius=0.007)
        # Bind the rigid body with anchor_id to the closest cloth vertices.
        cloth_closest_vertex_ids = get_closest(anchor_pos, mesh_vetex_positions)
        # print('Anchor at cloth vertex ids', cloth_closest_vertex_ids)
        assert (len(cloth_closest_vertex_ids) > 0)
        for v in cloth_closest_vertex_ids:
            sim.createSoftBodyAnchor(cloth_id, v, anchor_id, -1)
        anchor_ids.append(anchor_id)

    # Get the IDs of the vertices around the hole or goal region of the cloth.
    cloth_hole_vertex_ids = get_closest(
        anc_mid + args.cloth_init_hole_pos_offset,
        mesh_vetex_positions, max_dist=0.2, num_pins_per_pt=20)
    # print('cloth_hole_vertex_ids', cloth_hole_vertex_ids)
    assert (len(cloth_hole_vertex_ids) > 0)

    # Return only the immediately useful info.
    return sim, anchor_ids, goal_pos, cloth_id, cloth_hole_vertex_ids


def make_mpc_actions(sim, args, anchor_ids, debug=False):
    side_shift = np.array([0.0, -0.10, 0.0])  # shift by -10cm in y direction
    down_shift = np.array([0.0, 0.0, -0.10])  # shift by -10cm in z direction
    wiggle_shift = np.array([-0.05, 0.0, 0.0])  # shift by 5cm in x direction
    trajs = []

    feasible_actions = [side_shift,
                        down_shift,
                        side_shift + down_shift,
                        # side_shift / 2,
                        # down_shift / 2,
                        # (side_shift + down_shift) / 2,
                        # wiggle_shift,
                        # -wiggle_shift
                        ]

    for f_a in feasible_actions:
        for anchor_id in anchor_ids:
            anc_pos, anc_quat = sim.getBasePositionAndOrientation(anchor_id)
            anc_pos = np.array(anc_pos)
            end_pos = anc_pos + f_a
            if debug: print("Anchor", anchor_id, " I'll start at ", anc_pos, ", end up at ", end_pos)
            # Define 3D positions of waypoints.
            waypoints = [
                anc_pos, end_pos,
            ]
            traj = create_trajectory(
                waypoints, [STPS_PER_WPT] * len(waypoints), args.sim_frequency)
            trajs.append(traj)
    return trajs


def mpc_main(args):
    sim, anchor_ids, goal_pos, cloth_id, cloth_hole_vertex_ids = simple_init(
        None, args, args.anchor_init_pos, args.other_anchor_init_pos)
    assert args.do_perfect_model_rollouts is True  # todo change if using model
    # Get the initial hole center
    hole_center = get_hole_center(sim, cloth_id, cloth_hole_vertex_ids)
    dist_to_goal = np.linalg.norm(hole_center - goal_pos)
    for step in range(MAX_STPS):
        if dist_to_goal < 0.1: break
        # ok_to_reload = dist_to_goal > 0.3  # a hack to be far from other objects
        if args.do_perfect_model_rollouts and (step % MPC_STPS == 0):
            # input("New planning round; press Enter to continue.")
            sim.removeAllUserDebugItems()
            costs = []
            anc_pos, _ = sim.getBasePositionAndOrientation(anchor_ids[0])
            other_anc_pos, _ = sim.getBasePositionAndOrientation(anchor_ids[1])
            # 1. generate the actions for this round
            trajs = make_mpc_actions(sim, args, anchor_ids)
            # 2. try the pair of actions
            # todo: This acts as the model prediction/cost calculation, it's what you need to replace if you add a model
            for act_id in range(len(trajs) // 2):
                # Do the perfect rollout for the specified amount of steps per waypoint
                pred_hole_center = perfect_model_rollout(
                    sim, anchor_ids, cloth_id, cloth_hole_vertex_ids, args.reset_angular_vel,
                    trajs[2 * act_id:2 * (act_id + 1)], step, STPS_PER_WPT)
                # Reset the state
                _, anchor_ids, goal_pos, cloth_id, cloth_hole_vertex_ids = simple_init(
                    sim, args, anc_pos, other_anc_pos)
                # input('Reset done; press Enter to continue')
                costs.append(np.linalg.norm(pred_hole_center - goal_pos))
            bst_id = costs.index(min(costs))
            print("Best move index is", bst_id)
            # input("Done with cost calculation, press enter to continue to execution")
            best_move = trajs[2 * bst_id:2 * (bst_id + 1)]

        # Execute the action
        for anc_i in range(len(anchor_ids)):
            anc_pos, anc_quat = sim.getBasePositionAndOrientation(anchor_ids[anc_i])
            anc_linvel, anc_angvel = sim.getBaseVelocity(anchor_ids[anc_i])
            tgt_pos_vel = best_move[anc_i][step % STPS_PER_WPT]
            # Reset angular velocity of the anchors to avoid spinning too much.
            # Can comment this out if you would like to test full physics,
            # but be warned that it might make things less not more realistic.
            # if args.reset_angular_vel:
            sim.resetBaseVelocity(
               anchor_ids[anc_i], linearVelocity=tgt_pos_vel[3:6].tolist(),
               angularVelocity=[0, 0, 0])
            # Apply the desired force to the anchors.
            # sim.applyExternalForce(
            #     anchor_ids[anc_i], -1,
            #     pd_control_force(anc_pos, anc_linvel, tgt_pos_vel).tolist(),
            #     [0, 0, 0], pybullet.LINK_FRAME)
        sim.stepSimulation()  # this advances physics sim (must have this)
        viz_traj_deviation(sim, anchor_ids, best_move, step % STPS_PER_WPT)
        # Show distance to goal.
        hole_center = get_hole_center(sim, cloth_id, cloth_hole_vertex_ids)
        dist_to_goal = np.linalg.norm(hole_center - goal_pos) # todo Do I wanna move this inside the loop so it reacts faster to being at the goal?
        sim.addUserDebugText(f'dist {dist_to_goal:0.4f}',
                             textPosition=[0, 0, 0.8], textColorRGB=[0, 0, 1],
                             lifeTime=0, textSize=2)

    input("All done, press Enter to exit. ")


def main(args):
    steps_per_wpt = 100
    side_shift = np.array([0.0, -0.20, 0.0])  # shift by -20cm in y direction
    down_shift = np.array([0.0, 0.0, -0.30])  # shift by -30cm in z direction
    sim, anchor_ids, goal_pos, cloth_id, cloth_hole_vertex_ids = simple_init(
        None, args, args.anchor_init_pos, args.other_anchor_init_pos)
    #
    # Create trajectories for anchors.
    #
    trajs = []
    for anchor_id in anchor_ids:
        anc_pos, anc_quat = sim.getBasePositionAndOrientation(anchor_id)
        anc_pos = np.array(anc_pos)
        end_pos = anc_pos + 2 * side_shift + down_shift
        # Define 3D positions of waypoints.
        waypoints = [
            anc_pos, anc_pos,  # first stay in place for a bit
            anc_pos + side_shift,  # then shift along y
            anc_pos + 2 * side_shift,  # then shift more along y
            end_pos, end_pos  # then shift down and hold
        ]
        traj = create_trajectory(
            waypoints, [steps_per_wpt] * len(waypoints), args.sim_frequency)
        trajs.append(traj)
    #
    # Pull the anchors along the specified trajectories.
    #
    max_steps = trajs[0].shape[0]
    hole_center = get_hole_center(sim, cloth_id, cloth_hole_vertex_ids)
    dist_to_goal = np.linalg.norm(hole_center - goal_pos)
    for step in range(max_steps):
        ok_to_reload = dist_to_goal > 0.3  # a hack to be far from other objects
        if args.do_perfect_model_rollouts and ok_to_reload and (step % 50 == 0):
            anc_pos, _ = sim.getBasePositionAndOrientation(anchor_ids[0])
            other_anc_pos, _ = sim.getBasePositionAndOrientation(anchor_ids[1])
            pred_hole_center = perfect_model_rollout(
                sim, anchor_ids, cloth_id, cloth_hole_vertex_ids, args.reset_angular_vel, trajs, step)
            print('pred_hole_center', pred_hole_center)
            # Unfortunately pybullet cat save and restore state of deformables.
            # Hence we can only do a crude reset, and unfortunately can only
            # reload the cloth in its initial unfolded state.
            _, anchor_ids, goal_pos, cloth_id, cloth_hole_vertex_ids = simple_init(
                sim, args, anc_pos, other_anc_pos)
            input('Reset done; press Enter to continue')
        # Start real MPC action.
        for anc_i in range(len(anchor_ids)):
            anc_pos, anc_quat = sim.getBasePositionAndOrientation(anchor_ids[anc_i])
            anc_linvel, anc_angvel = sim.getBaseVelocity(anchor_ids[anc_i])
            tgt_pos_vel = trajs[anc_i][step]
            # Reset angular velocity of the anchors to avoid spinning too much.
            # Can comment this out if you would like to test full physics,
            # but be warned that it might make things less not more realistic.
            # sim.resetBaseVelocity(
            #    anchor_ids[anc_i], linearVelocity=tgt_pos_vel[3:6].tolist(),
            #    angularVelocity=[0, 0, 0])
            # Apply the desired force to the anchors.
            sim.applyExternalForce(
                anchor_ids[anc_i], -1,
                pd_control_force(anc_pos, anc_linvel, tgt_pos_vel).tolist(),
                [0, 0, 0], pybullet.LINK_FRAME)
        sim.stepSimulation()  # this advances physics sim (must have this)
        viz_traj_deviation(sim, anchor_ids, trajs, step)
        # Show distance to goal.
        hole_center = get_hole_center(sim, cloth_id, cloth_hole_vertex_ids)
        dist_to_goal = np.linalg.norm(hole_center - goal_pos)
        sim.addUserDebugText(f'dist {dist_to_goal:0.4f}',
                             textPosition=[0, 0, 0.8], textColorRGB=[0, 0, 1],
                             lifeTime=1, textSize=2)
    input('Done; press Enter to exit')


if __name__ == '__main__':
    # main(get_args())
    mpc_main(get_args())
