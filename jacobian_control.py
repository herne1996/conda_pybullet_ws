import numpy as np
import pybullet as p
import pybullet_data
import time
import math
import os

# =========================
# Utility / helpers
# =========================
def smooth_step(current, target, alpha):
    """Exponential smoothing: current <- (1-alpha)*current + alpha*target"""
    return [(1.0 - alpha) * c + alpha * t for c, t in zip(current, target)]

def wrap_to_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def get_base_xy_yaw(robot_id):
    pos, orn = p.getBasePositionAndOrientation(robot_id)
    _, _, yaw = p.getEulerFromQuaternion(orn)
    return np.array([pos[0], pos[1]]), yaw

def set_diffdrive_wheels(robot_id, left_wheels, right_wheels,
                         v, omega,
                         wheel_radius, half_track,
                         max_wheel_speed, max_force):
    """Send (v,omega) to wheel VELOCITY_CONTROL."""
    v_left  = v - omega * half_track
    v_right = v + omega * half_track
    w_left  = v_left / wheel_radius
    w_right = v_right / wheel_radius

    w_left  = max(min(w_left,  max_wheel_speed), -max_wheel_speed)
    w_right = max(min(w_right, max_wheel_speed), -max_wheel_speed)

    for j in left_wheels:
        p.setJointMotorControl2(robot_id, j, p.VELOCITY_CONTROL,
                                targetVelocity=w_left, force=max_force)
    for j in right_wheels:
        p.setJointMotorControl2(robot_id, j, p.VELOCITY_CONTROL,
                                targetVelocity=w_right, force=max_force)

def goto_controller_diffdrive(robot_id, goal_xy, goal_yaw=None,
                              k_rho=0.8, k_alpha=2.0, k_yaw=2.0,
                              max_v=0.7, max_omega=1.2,
                              pos_tol=0.3, yaw_tol=0.12):
    """
    Simple go-to-goal for diff-drive.
    Returns: v, omega, reached_pos, reached_yaw
    """
    xy, yaw = get_base_xy_yaw(robot_id)
    d = goal_xy - xy
    rho = float(np.linalg.norm(d))

    # close enough in position
    if rho < pos_tol:
        if goal_yaw is None:
            return 0.0, 0.0, True, True
        yaw_err = wrap_to_pi(goal_yaw - yaw)
        if abs(yaw_err) < yaw_tol:
            return 0.0, 0.0, True, True
        omega = float(np.clip(k_yaw * yaw_err, -max_omega, max_omega))
        return 0.0, omega, True, False

    target_heading = math.atan2(d[1], d[0])
    alpha = wrap_to_pi(target_heading - yaw)

    v = float(np.clip(k_rho * rho, 0.0, max_v))
    omega = float(np.clip(k_alpha * alpha, -max_omega, max_omega))
    return v, omega, False, (goal_yaw is None)

# =========================
# Robot joint discovery
# =========================
def find_wheel_joints(robot_id):
    """Find Husky wheel joints by name."""
    wheel_names = {
        b'husky_front_left_wheel': 'fl',
        b'husky_front_right_wheel': 'fr',
        b'husky_rear_left_wheel': 'rl',
        b'husky_rear_right_wheel': 'rr',
    }
    left_wheels, right_wheels = [], []
    for j in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, j)
        name = info[1]
        if name in wheel_names:
            tag = wheel_names[name]
            if tag in ['fl', 'rl']:
                left_wheels.append(j)
            else:
                right_wheels.append(j)
    return left_wheels, right_wheels

def find_arm_joints(robot_id):
    """Find UR3 arm joints by name."""
    joint_names = {
        b'ur3_joint': 'j1',
        b'ur3_joint_2': 'j2',
        b'ur3_joint_3': 'j3',
        b'ur3_joint_4': 'j4',
        b'ur3_joint_5': 'j5',
        b'ur3_joint_6': 'j6',
    }
    arm_joints = []
    for j in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, j)
        name = info[1]
        if name in joint_names:
            arm_joints.append(j)
    return arm_joints

# =========================
# Arm / gripper motions
# =========================
def move_ee_to_payload_ik(robot_id, arm_joints, ee_link_index, payload_id,
                          max_force=200.0, steps=240, residual_thresh=1e-3,
                          keep_current_yaw=True):
    """
    Move end-effector to payload position using IK, with EE pointing vertically downward (world frame).
    Downward: EE tool local +Z points to world -Z (approx via roll=pi).
    """
    # Target position (world)
    target_pos, _ = p.getBasePositionAndOrientation(payload_id)
    target_pos = np.array(target_pos) + np.array([0.0, 0.0, 0.2])
    print("target_pos:", target_pos)

    ee_state = p.getLinkState(robot_id, ee_link_index, computeForwardKinematics=1)
    ee_orn_cur = ee_state[5]

    if keep_current_yaw:
        _, _, yaw_cur = p.getEulerFromQuaternion(ee_orn_cur)
        target_orn = p.getQuaternionFromEuler([math.pi, 0.0, yaw_cur])
    else:
        target_orn = p.getQuaternionFromEuler([math.pi, 0.0, 0.0])

    print("target_orn (quat):", target_orn)

    # Movable joints list (matches IK output ordering)
    movable_joints = []
    for j in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, j)
        joint_type = info[2]
        if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            movable_joints.append(j)

    # Joint limits / ranges / rest poses in movable joint order
    lowers, uppers, ranges, rest = [], [], [], []
    for j in movable_joints:
        info = p.getJointInfo(robot_id, j)
        lo = float(info[8])
        hi = float(info[9])
        if hi <= lo:  # continuous or unspecified
            lo, hi = -2.0 * math.pi, 2.0 * math.pi
        lowers.append(lo)
        uppers.append(hi)
        ranges.append(hi - lo)
        rest.append(p.getJointState(robot_id, j)[0])

    ik_all = p.calculateInverseKinematics(
        bodyUniqueId=robot_id,
        endEffectorLinkIndex=ee_link_index,
        targetPosition=target_pos,
        targetOrientation=target_orn,
        lowerLimits=lowers,
        upperLimits=uppers,
        jointRanges=ranges,
        restPoses=rest,
        maxNumIterations=200,
        residualThreshold=residual_thresh
    )

    # Map IK output -> actual joint index
    ik_map = {j_idx: ik_all[k] for k, j_idx in enumerate(movable_joints)}

    # Desired arm joints in arm_joints order
    q_des = [ik_map[j] for j in arm_joints]
    q_cur = [p.getJointState(robot_id, j)[0] for j in arm_joints]

    sim_dt = 1.0 / 240.0
    for t in range(steps):
        alpha = (t + 1) / float(steps)
        q_cmd = [(1 - alpha) * qc + alpha * qd for qc, qd in zip(q_cur, q_des)]
        for j, q in zip(arm_joints, q_cmd):
            p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL,
                                    targetPosition=q, force=max_force)
        p.stepSimulation()
        time.sleep(sim_dt)

def move_gripper_pos(robot_id, gripper_joints, target_pos, alpha, force=100):
    q_cmd = [p.getJointState(robot_id, j)[0] for j in gripper_joints]
    q_cmd = smooth_step(q_cmd, [float(x) for x in target_pos], alpha)
    for idx, j in enumerate(gripper_joints):
        p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL,
                                targetPosition=q_cmd[idx], force=force)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    if p.isConnected():
        p.disconnect()

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")

    # --- Tables ---
    table1_pos = [3,  3, -0.25]
    table2_pos = [3, -3, -0.25]
    p.loadURDF("table_square/table_square.urdf", basePosition=table1_pos, useFixedBase=True)
    p.loadURDF("table_square/table_square.urdf", basePosition=table2_pos, useFixedBase=True)

    # --- Payload cube on table 1 ---
    half_extents_payload = [0.025, 0.025, 0.025]
    col_id_pl = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents_payload)
    vis_id_pl = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents_payload, rgbaColor=[1, 0, 0, 1])
    pl_id = p.createMultiBody(
        baseMass=0.1,
        baseCollisionShapeIndex=col_id_pl,
        baseVisualShapeIndex=vis_id_pl,
        basePosition=[3, 2.8, 0.7]
    )

    # --- Robot ---
    robot_id = p.loadURDF(
        "Customized Robots/UR3 n Husky/husky_ur3_mobile_manipulator_with_wsg50.urdf",
        basePosition=[0, 0, 0.1],
        useFixedBase=False
    )

    print("Number of joints:", p.getNumJoints(robot_id))
    for j in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, j)
        print(j, info[1].decode("utf-8"))

    # --- Gripper ---
    finger_joints = [15, 17]
    finger_open  = np.array([0.05, -0.05])
    finger_close = np.zeros(2)
    p.resetJointState(robot_id, finger_joints[0], finger_open[0])
    p.resetJointState(robot_id, finger_joints[1], finger_open[1])

    # --- Arm ---
    arm_joints = find_arm_joints(robot_id)
    q_cmd_arm = [p.getJointState(robot_id, j)[0] for j in arm_joints]
    q_zero = np.zeros(6)

    # Your EE link index (as you set)
    EE_LINK_INDEX = 14

    # --- Base wheels (Husky diff-drive) ---
    left_wheels, right_wheels = find_wheel_joints(robot_id)
    wheel_radius = 0.165
    half_track = 0.3
    max_wheel_speed = 15.0
    base_max_force = 100.0

    # --- Teleop params ---
    lin_vel = 0.0
    ang_vel = 0.0
    lin_step = 0.3
    ang_step = 0.8
    max_lin = 1.0
    max_ang = 2.0
    lin_damp = 0.9
    ang_damp = 0.9

    # --- Camera ---
    base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
    _, _, yaw = p.getEulerFromQuaternion(base_orn)
    p.resetDebugVisualizerCamera(
        cameraDistance=2.0,
        cameraYaw=math.degrees(yaw) + 180.0,
        cameraPitch=-30.0,
        cameraTargetPosition=base_pos
    )

    dt = 1.0 / 240.0
    return_alpha = 0.05

    # =========================
    # AUTO mission setup
    # =========================
    # Approach points (stand-off so base doesn't collide with the table)
    # Tune these if needed.
    APPROACH1_XY = np.array([1.5, 2.8])  # near table 1 / payload
    APPROACH2_XY = np.array([2.8, -2.8])   # near table 2 / destination
    FACE_TABLE_YAW = 0.0                   # face +x direction (tables around x=3)

    AUTO_IDLE      = 0
    AUTO_GOTO_T1   = 1
    AUTO_ALIGN_T1  = 2
    AUTO_IK_PICK   = 3
    AUTO_CLOSE     = 4
    AUTO_GOTO_T2   = 5
    AUTO_ALIGN_T2  = 6
    AUTO_OPEN      = 7
    AUTO_DONE      = 8

    auto_active = False
    auto_state = AUTO_IDLE
    auto_key_was_down = False

    # base controller limits
    ctrl_max_v = 0.7
    ctrl_max_w = 1.2

    # Key edge detectors
    one_was_down = False
    two_was_down = False
    nine_was_down = False
    zero_was_down = False

    while p.isConnected():
        keys = p.getKeyboardEvents()

        # =========================
        # Toggle AUTO mission: press 'a'
        # =========================
        if ord('a') in keys and (keys[ord('a')] & p.KEY_IS_DOWN):
            if not auto_key_was_down:
                auto_active = not auto_active
                if auto_active:
                    auto_state = AUTO_GOTO_T1
                    print("[AUTO] START: go to payload table -> pick -> go to other table -> place.")
                else:
                    auto_state = AUTO_IDLE
                    print("[AUTO] STOP: back to teleop.")
                auto_key_was_down = True
        else:
            auto_key_was_down = False

        # =========================
        # BASE COMMAND: AUTO overrides TELEOP
        # =========================
        if auto_active and auto_state != AUTO_IDLE:
            # default: stop base
            v_cmd, w_cmd = 0.0, 0.0

            if auto_state == AUTO_GOTO_T1:
                v_cmd, w_cmd, reached_pos, _ = goto_controller_diffdrive(
                    robot_id, APPROACH1_XY, goal_yaw=None,
                    max_v=ctrl_max_v, max_omega=ctrl_max_w
                )
                if reached_pos:
                    auto_state = AUTO_GOTO_T2
                    print("[AUTO] Reached Table 1 approach. Aligning...")


            elif auto_state == AUTO_GOTO_T2:
                v_cmd, w_cmd, reached_pos, _ = goto_controller_diffdrive(
                    robot_id, APPROACH2_XY, goal_yaw=None,
                    max_v=ctrl_max_v, max_omega=ctrl_max_w
                )
                if reached_pos:
                    auto_state = AUTO_ALIGN_T2
                    print("[AUTO] Reached Table 2 approach. Aligning...")


            # send wheel commands
            set_diffdrive_wheels(
                robot_id, left_wheels, right_wheels,
                v_cmd, w_cmd,
                wheel_radius, half_track,
                max_wheel_speed, base_max_force
            )
        
        
        # if auto_active and auto_state != AUTO_IDLE:
        #     # default: stop base
        #     v_cmd, w_cmd = 0.0, 0.0

        #     if auto_state == AUTO_GOTO_T1:
        #         v_cmd, w_cmd, reached_pos, _ = goto_controller_diffdrive(
        #             robot_id, APPROACH1_XY, goal_yaw=None,
        #             max_v=ctrl_max_v, max_omega=ctrl_max_w
        #         )
        #         if reached_pos:
        #             auto_state = AUTO_ALIGN_T1
        #             print("[AUTO] Reached Table 1 approach. Aligning...")

        #     elif auto_state == AUTO_ALIGN_T1:
        #         v_cmd, w_cmd, _, reached_yaw = goto_controller_diffdrive(
        #             robot_id, APPROACH1_XY, goal_yaw=FACE_TABLE_YAW,
        #             max_v=ctrl_max_v, max_omega=ctrl_max_w
        #         )
        #         if reached_yaw:
        #             auto_state = AUTO_IK_PICK
        #             print("[AUTO] Aligned at Table 1. Running IK to payload...")

        #     elif auto_state == AUTO_GOTO_T2:
        #         v_cmd, w_cmd, reached_pos, _ = goto_controller_diffdrive(
        #             robot_id, APPROACH2_XY, goal_yaw=None,
        #             max_v=ctrl_max_v, max_omega=ctrl_max_w
        #         )
        #         if reached_pos:
        #             auto_state = AUTO_ALIGN_T2
        #             print("[AUTO] Reached Table 2 approach. Aligning...")

        #     elif auto_state == AUTO_ALIGN_T2:
        #         v_cmd, w_cmd, _, reached_yaw = goto_controller_diffdrive(
        #             robot_id, APPROACH2_XY, goal_yaw=FACE_TABLE_YAW,
        #             max_v=ctrl_max_v, max_omega=ctrl_max_w
        #         )
        #         if reached_yaw:
        #             auto_state = AUTO_OPEN
        #             print("[AUTO] Aligned at Table 2. Opening gripper (place)...")

        #     # send wheel commands
        #     set_diffdrive_wheels(
        #         robot_id, left_wheels, right_wheels,
        #         v_cmd, w_cmd,
        #         wheel_radius, half_track,
        #         max_wheel_speed, base_max_force
        #     )
            
            

        else:
            # =========================
            # TELEOP base (i/k + j/l)
            # =========================
            if ord('i') in keys and keys[ord('i')] & p.KEY_IS_DOWN:
                lin_vel += lin_step * dt * 60
            elif ord('k') in keys and keys[ord('k')] & p.KEY_IS_DOWN:
                lin_vel -= lin_step * dt * 60
            else:
                lin_vel *= lin_damp

            if ord('j') in keys and keys[ord('j')] & p.KEY_IS_DOWN:
                ang_vel += ang_step * dt * 60
            elif ord('l') in keys and keys[ord('l')] & p.KEY_IS_DOWN:
                ang_vel -= ang_step * dt * 60
            else:
                ang_vel *= ang_damp

            lin_vel = max(min(lin_vel, max_lin), -max_lin)
            ang_vel = max(min(ang_vel, max_ang), -max_ang)

            set_diffdrive_wheels(
                robot_id, left_wheels, right_wheels,
                lin_vel, ang_vel,
                wheel_radius, half_track,
                max_wheel_speed, base_max_force
            )

        # # =========================
        # # AUTO ARM / GRIPPER actions (blocking)
        # # =========================
        # if auto_active:
        #     if auto_state == AUTO_IK_PICK:
        #         move_ee_to_payload_ik(
        #             robot_id=robot_id,
        #             arm_joints=arm_joints,
        #             ee_link_index=EE_LINK_INDEX,
        #             payload_id=pl_id,
        #             max_force=250.0,
        #             steps=240
        #         )
        #         q_cmd_arm = [p.getJointState(robot_id, j)[0] for j in arm_joints]
        #         auto_state = AUTO_CLOSE
        #         print("[AUTO] IK done. Closing gripper...")

        #     elif auto_state == AUTO_CLOSE:
        #         for _ in range(120):
        #             move_gripper_pos(robot_id, finger_joints, finger_close, return_alpha, force=120)
        #             p.stepSimulation()
        #             time.sleep(dt)
        #         auto_state = AUTO_GOTO_T2
        #         print("[AUTO] Gripper closed. Going to Table 2...")

        #     elif auto_state == AUTO_OPEN:
        #         for _ in range(120):
        #             move_gripper_pos(robot_id, finger_joints, finger_open, return_alpha, force=120)
        #             p.stepSimulation()
        #             time.sleep(dt)
        #         auto_state = AUTO_DONE
        #         print("[AUTO] Gripper opened. Mission complete.")

        #     elif auto_state == AUTO_DONE:
        #         auto_active = False
        #         auto_state = AUTO_IDLE
        #         print("[AUTO] Returned to teleop.")

        # =========================
        # Manual ARM / GRIPPER keys (still available)
        # =========================
        # [1] IK to payload
        if ord('1') in keys and (keys[ord('1')] & p.KEY_IS_DOWN):
            if not one_was_down:
                print("[1] Moving end-effector to payload using IK...")
                move_ee_to_payload_ik(
                    robot_id=robot_id,
                    arm_joints=arm_joints,
                    ee_link_index=EE_LINK_INDEX,
                    payload_id=pl_id,
                    max_force=250.0,
                    steps=240
                )
                q_cmd_arm = [p.getJointState(robot_id, j)[0] for j in arm_joints]
                print("[1] Done.")
                one_was_down = True
        else:
            one_was_down = False

        # [2] go to q_zero (smooth)
        if ord('2') in keys and (keys[ord('2')] & p.KEY_IS_DOWN):
            if not two_was_down:
                print("[2] Returning arm to q_zero (smooth)...")
                # do a short smooth return
                q_start = np.array([p.getJointState(robot_id, j)[0] for j in arm_joints], dtype=float)
                steps = 240
                for t in range(steps):
                    a = (t + 1) / float(steps)
                    q_cmd = (1 - a) * q_start + a * q_zero
                    for idx, j in enumerate(arm_joints):
                        p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL,
                                                targetPosition=float(q_cmd[idx]), force=150)
                    p.stepSimulation()
                    time.sleep(dt)
                q_cmd_arm = [p.getJointState(robot_id, j)[0] for j in arm_joints]
                print("[2] Done.")
                two_was_down = True
        else:
            two_was_down = False

        # [9] open gripper
        if ord('9') in keys and (keys[ord('9')] & p.KEY_IS_DOWN):
            if not nine_was_down:
                move_gripper_pos(robot_id, finger_joints, finger_open, return_alpha, force=120)
                nine_was_down = True
        else:
            nine_was_down = False

        # [0] close gripper
        if ord('0') in keys and (keys[ord('0')] & p.KEY_IS_DOWN):
            if not zero_was_down:
                move_gripper_pos(robot_id, finger_joints, finger_close, return_alpha, force=120)
                zero_was_down = True
        else:
            zero_was_down = False

        # =========================
        # Arm hold (keep last q_cmd_arm)
        # =========================
        for idx, j in enumerate(arm_joints):
            p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL,
                                    targetPosition=float(q_cmd_arm[idx]), force=120)

        p.stepSimulation()
        time.sleep(dt)
