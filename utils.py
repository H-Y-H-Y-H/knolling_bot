import pybullet as p
import time
import pybullet_data
import random
import os
import numpy as np
import csv

filename = "robot_arm1"
m1 = 0.20615
m2 = 0.200
len_wrist = 0.174
height_base = 0.105
theta_1_offset = np.arctan2(0.05, 0.2)
origin_offset = 0.084
height_offset = 0.005

def create_box(body_name: str,
               position: np.ndarray,
               orientation: np.ndarray = np.array([0, 0, 0, 1]),
               rgba_color=None,
               size=None,
               mass=0.1,
               ) -> None:
    """
    Create a box.
    """
    length = size[0]
    width = size[1]
    height = size[2]

    visual_kwargs = {
        "rgbaColor": rgba_color if rgba_color is not None else [np.random.random(), np.random.random(),
                                                                np.random.random(), 1],
        "halfExtents": [length / 2, width / 2, height / 2]
    }
    collision_kwargs = {
        "halfExtents": [length / 2, width / 2, height / 2]
    }

    _create_geometry(body_name,
                          geom_type=p.GEOM_BOX,
                          mass=mass,
                          position=position,
                          orientation=orientation,
                          lateral_friction=1.0,
                          contact_damping=1.0,
                          contact_stiffness=50000,
                          visual_kwargs=visual_kwargs,
                          collision_kwargs=collision_kwargs)


def _create_geometry(
        body_name: str,
        geom_type: int,
        mass: float = 0.0,
        position=None,
        orientation=None,
        ghost: bool = False,
        lateral_friction=None,
        spinning_friction=None,
        contact_damping=None,
        contact_stiffness=None,
        visual_kwargs={},
        collision_kwargs={},
) -> None:
    """Create a geometry.

    Args:
        body_name (str): The name of the body. Must be unique in the sim.
        geom_type (int): The geometry type. See p.GEOM_<shape>.
        mass (float, optional): The mass in kg. Defaults to 0.
        position (np.ndarray, optional): The position, as (x, y, z). Defaults to [0, 0, 0].
        orientation (np.ndarray, optional): The orientation. Defaults to [0, 0, 0, 1]
        ghost (bool, optional): Whether the body can collide. Defaults to False.
        lateral_friction (float or None, optional): Lateral friction. If None, use the default pybullet
            value. Defaults to None.
        spinning_friction (float or None, optional): Spinning friction. If None, use the default pybullet
            value. Defaults to None.
        visual_kwargs (dict, optional): Visual kwargs. Defaults to {}.
        collision_kwargs (dict, optional): Collision kwargs. Defaults to {}.
    """
    position = position if position is not None else np.zeros(3)
    orientation = orientation if orientation is not None else np.array([0, 0, 0, 1])

    baseVisualShapeIndex = p.createVisualShape(geom_type, **visual_kwargs)
    if not ghost:
        baseCollisionShapeIndex = p.createCollisionShape(geom_type, **collision_kwargs)
    else:
        baseCollisionShapeIndex = -1
    box_id = p.createMultiBody(
        baseVisualShapeIndex=baseVisualShapeIndex,
        baseCollisionShapeIndex=baseCollisionShapeIndex,
        baseMass=mass,
        basePosition=position,
        baseOrientation=orientation
    )

def inverse_kinematic(pos, ori, parameters=None):

    if pos.shape[0] == 3:
        pos = pos.reshape(1, 3)
        ori = ori.reshape(1, 3)
    pos[:, 0] += origin_offset
    pos[:, 2] += height_offset
    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]
    yaw = ori[:, 2]
    distance = np.sqrt(x ** 2 + y ** 2)
    z_cal = z + len_wrist - height_base

    theta_2 = np.arccos(((m1 ** 2 - m2 ** 2 - z_cal ** 2 - distance ** 2) / (2 * m2)) / np.sqrt(z_cal ** 2 + distance ** 2)) - np.arccos(z_cal / np.sqrt(z_cal ** 2 + distance ** 2))
    theta_1 = np.arccos((distance - m2 * np.sin(theta_2)) / m1)
    motor_0 = np.arctan2(y, x) + np.pi
    motor_1 = np.pi * 2 - (theta_1 + theta_1_offset + np.pi / 2)
    motor_2 = theta_2 + np.pi / 2 - theta_1 - theta_1_offset + np.pi / 2
    motor_3 = np.pi - theta_2
    motor_4 = np.pi - (yaw - (motor_0 - np.pi))

    motor = np.concatenate((motor_0.reshape(-1, 1), motor_1.reshape(-1, 1), motor_2.reshape(-1, 1), motor_3.reshape(-1, 1), motor_4.reshape(-1, 1)), axis=1)
    motor = motor / (np.pi * 2) * 4096
    motor = np.insert(motor, 1, motor[:, 1], axis=1)
    return motor

def forward_kinematic(motor):

    motor = motor / 4096 * (np.pi * 2)
    motor = np.delete(motor, 2, axis=1)
    # print('this is motor in forward_kinematic', motor)
    theta_0 = motor[:, 0] - np.pi
    theta_1 = np.pi * 2 - motor[:, 1] - theta_1_offset - np.pi / 2
    theta_2 = motor[:, 2] - np.pi / 2 + theta_1 + theta_1_offset - np.pi / 2
    theta_3 = np.pi - motor[:, 3]
    yaw = motor[:, 0] - np.pi + np.pi - motor[:, 4]
    distance = m1 * np.cos(theta_1) + m2 * np.sin(theta_2)
    x = distance * np.cos(theta_0)
    y = distance * np.sin(theta_0)
    z = height_base + (m1 * np.sin(theta_1) - m2 * np.cos(theta_2)) - len_wrist

    pos = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)), axis=1)
    pos[:, 0] -= origin_offset
    pos[:, 2] -= height_offset
    ori = np.concatenate((np.zeros(len(motor)).reshape(-1, 1), np.ones(len(motor)).reshape(-1, 1) * np.pi / 2, yaw.reshape(-1, 1)), axis=1)

    return pos, ori

def reset(robotID):
    p.setJointMotorControlArray(robotID, [0, 1, 2, 3, 4, 7, 8], p.POSITION_CONTROL,
                                targetPositions=[0, -np.pi / 2, np.pi / 2, 0, 0, 0, 0])


def sim_cmd2tarpos(tar_pos_cmds):
    tar_pos_cmds = np.asarray(tar_pos_cmds)
    motion_limit = np.asarray([1 / 180, 1 / 135, 1 / 135, 1 / 90, 1 / 180])
    reset_rad = [0, -np.pi / 2, np.pi / 2, 0, 0]
    reset_cmds = np.asarray([0.5, 0, 1, 0, 0.5])
    cmds_gap = np.subtract(tar_pos_cmds, reset_cmds)
    rad_gap = np.divide(cmds_gap, motion_limit) * np.pi / 180
    tar_pos = np.add(reset_rad, rad_gap)
    return tar_pos


# real robot:motor limit:0-4095(0 to 360 degrees)

def real_cmd2tarpos(tar_pos_cmds):  # sim to real!!!!!!!!!!!!!!!!

    # input: scaled angle of motors in pybullet, basically (0, 1)
    # output: angle of motors in real world, (0, 4096)

    tar_pos_cmds = np.asarray(tar_pos_cmds)
    pos2deg = 4095 / 360
    motion_limit = np.asarray([1 / 180, 1 / 135, 1 / 135, 1 / 90, 1 / 180])
    motion_limit2 = np.divide(motion_limit, pos2deg)
    reset_pos = [3075, 1025, 1050, 2050, 2050]
    reset_cmds = np.asarray([0.5, 0, 1, 0, 0.5])
    cmds_gap = np.subtract(tar_pos_cmds, reset_cmds)
    cmds_gap[2] = -cmds_gap[2]
    cmds_gap[3] = -cmds_gap[3]
    pos_gap = np.divide(cmds_gap, motion_limit2)
    tar_pos = np.add(reset_pos, pos_gap)
    tar_pos2 = np.insert(tar_pos, 2, tar_pos[1])
    # tar_pos2 = tar_pos2.astype(int)
    return tar_pos2


def real_tarpos2cmd(tar_pos):  # real to sim!!!!!!!!!!!

    # input: angle of motors in real world, (0, 4096)
    # output: scaled angle of motors in pybullet, basically (0, 1)

    tar_pos = np.delete(tar_pos, 2)
    tar_pos = np.asarray(tar_pos)
    # print('this is tar from calculation', tar_pos)
    pos2deg = 4095 / 360
    motion_limit = np.asarray([1 / 180, 1 / 135, 1 / 135, 1 / 90, 1 / 180])
    reset_pos = [3075, 1025, 1050, 2050, 2050]
    reset_cmds = np.asarray([0.5, 0, 1, 0, 0.5])
    motion_limit2 = np.divide(motion_limit, pos2deg)
    pos_gap = np.subtract(tar_pos, reset_pos)
    cmds_gap = np.multiply(motion_limit2, pos_gap)
    cmds_gap[2] = -cmds_gap[2]
    cmds_gap[3] = -cmds_gap[3]
    cmds = np.add(cmds_gap, reset_cmds)

    return cmds


def rad2cmd(cur_rad):  # sim to real

    # input: angle of motors in pybullet, (-180, 180)
    # output: scaled angle of motors in pybullet, basically (0, 1)

    cur_rad = np.asarray(cur_rad)
    reset_rad = np.asarray([0, -np.pi / 2, np.pi / 2, 0, 0])
    rad_gap = np.subtract(cur_rad, reset_rad)
    motion_limit = np.asarray([1 / 180, 1 / 135, 1 / 135, 1 / 90, 1 / 180])
    cmds_gap = np.multiply(rad_gap, motion_limit) * 180 / np.pi
    reset_cmds = np.asarray([0.5, 0, 1, 0, 0.5])
    tar_cmds = np.add(reset_cmds, cmds_gap)
    return tar_cmds


def cmd2rad(cur_cmd):  # real to sim

    # input: scaled angle of motors in pybullet, basically (0, 1)
    # output: angle of motors in pybullet, (-180, 180)

    cur_cmd = np.asarray(cur_cmd)
    reset_cmds = np.asarray([0.5, 0, 1, 0, 0.5])
    cmds_gap = np.subtract(cur_cmd, reset_cmds)
    motion_limit = np.asarray([1 / 180, 1 / 135, 1 / 135, 1 / 90, 1 / 180])
    rad_gap = np.true_divide(cmds_gap, motion_limit) * (np.pi / 180)
    reset_rad = np.asarray([0, -np.pi / 2, np.pi / 2, 0, 0])
    tar_rads = np.add(reset_rad, rad_gap)

    return tar_rads


def rad2pos(cur_rad):
    tar_cmds = rad2cmd(cur_rad)
    # print("cmd", tar_cmds)
    pos = real_cmd2tarpos(tar_cmds)
    return pos


def change_sequence(pos_before):
    origin_point = np.array([0, -0.2])
    delete_index = np.where(pos_before == 0)[0]
    distance = np.linalg.norm(pos_before[:, :2] - origin_point, axis=1)
    order = np.argsort(distance)
    return order