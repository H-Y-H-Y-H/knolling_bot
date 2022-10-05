import pybullet as p
import time
import pybullet_data
import random
from math import sqrt, cos, sin, pi
import os
import numpy as np
import csv


def reset(robotID):
    p.setJointMotorControlArray(robotID, [0,1,2,3,4,7,8], p.POSITION_CONTROL, targetPositions=[0,-np.pi/2,np.pi/2,0,0,0,0])


def idk(tar_pos_cmds):
    tar_pos_cmds = np.asarray(tar_pos_cmds)
    motion_limit = np.asarray([1/180, 1/135, 1/135, 1/90, 1/180])
    reset_rad = [0, -np.pi/2, np.pi/2, 0, 0]
    reset_cmds = np.asarray([0.5, 0, 1, 0, 0.5])
    cmds_gap = np.subtract(tar_pos_cmds, reset_cmds)
    rad_gap = np.divide(cmds_gap, motion_limit) * np.pi/180
    tar_pos = np.add(reset_rad, rad_gap)
    return tar_pos

# real robot:motor limit:0-4095(0 to 360 degrees)

if __name__ == "__main__":
    filename = "robot_arm1"

    # or p.DIRECT for non-graphical version
    physicsClient = p.connect(1)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
    p.setGravity(0, 0, -9.8)
    planeId = p.loadURDF("plane.urdf")
    table_scale = 0.7
    table_surface_height = 0.625*table_scale
    startPos = [0, 0, table_surface_height]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])


    boxId = p.loadURDF(filename + ".urdf", startPos, startOrientation, useFixedBase=1, flags=p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
    # boxId2 = p.loadURDF("cube_small.urdf", [0, 0.2, 0.025], startOrientation, useFixedBase=0, flags=flags=p.URDF_USE_SELF_COLLISION)
    boxId3 = p.loadURDF("table/table.urdf", [(0.5-0.16)*table_scale, 0, 0], p.getQuaternionFromEuler([0, 0, np.pi/2]), useFixedBase=1,
                        flags=p.URDF_USE_SELF_COLLISION, globalScaling=table_scale)
    # boxId4 = p.loadURDF("cube_small.urdf", [0, 0.2, table_surface_height], startOrientation, useFixedBase=0, flags=p.URDF_USE_SELF_COLLISION)
    # p.changeDynamics(boxId, 10, lateralFriction=0.99,spinningFriction=0.02,rollingFriction=0.002)
    # p.changeDynamics(boxId, 11, lateralFriction=0.99,spinningFriction=0.02,rollingFriction=0.002)
    # p.changeDynamics(boxId2, -1, lateralFriction=0.99,spinningFriction=0.02,rollingFriction=0.002)
    # num_joints = p.getNumJoints(boxId)
    # print(num_joints)
    # print(p.getLinkState(boxId,12))


    #ee_link index = 9
    #base_rot_joint_index = 0
    #shuold_index = 1
    #elbow_index = 2
    #index = 3
    #index = 4
    #left_gripper = 7, lower="0" upper="0.03202"
    #right_gripper = 8, lower="0" upper="0.03202"
    reset(boxId)
    for i in range(60):
        p.stepSimulation()
        time.sleep(1/240)

    time.sleep(10)
    # for i in range(10):
    #     tar_pos1 = np.random.random_sample(size=5)
    #     print(tar_pos1)
    #     angle1 = idk(tar_pos1)
    #     p.setJointMotorControlArray(boxId, [0, 1, 2, 3, 4], p.POSITION_CONTROL, targetPositions = angle1,targetVelocities = [1,1,1,1,1])
    #     p.setJointMotorControlArray(boxId, [7,8], p.POSITION_CONTROL, targetPositions=[0,0])
    #     for j in range(200):
    #         p.stepSimulation()
    #         time.sleep(1/240)
    #     # input("next")

    tar_pos2 = [0, 90/135, 1-90/135, 0, 0.5]
    angle2 = idk(tar_pos2)
    p.setJointMotorControlArray(boxId, [0, 1, 2, 3, 4], p.POSITION_CONTROL, targetPositions=angle2)

    # ik_angle = p.calculateInverseKinematics(boxId, 9, targetPosition = [0, 0.2, 0.2],maxNumIterations = 200, targetOrientation = p.getQuaternionFromEuler([0,1.57,1.57]))
    # # print(ik_angle)
    # # print(len(ik_angle))
    # p.setJointMotorControlArray(boxId, [0, 1, 2, 3, 4, 5], p.POSITION_CONTROL, targetPositions = ik_angle[0:6])
    # p.setJointMotorControlArray(boxId, [7,8], p.POSITION_CONTROL, targetPositions=[0.032,0.032])
    # p.setJointMotorControl2(boxId, 0, p.POSITION_CONTROL, targetPosition = 1.57, force=2, maxVelocity=100)
    # time.sleep(5)
    # print(p.getBasePositionAndOrientation(boxId))
    # for i in range(200):
    #     p.stepSimulation()
    #     time.sleep(0)
    #
    # print(p.getLinkState(boxId, 9))
    t = 0
    for i in range(20000):
        p.stepSimulation()
        # p.setJointMotorControlArray(boxId, [7, 8], p.POSITION_CONTROL, targetPositions=[0.032*sin(t), 0.032*sin(t)])
        # t+=0.01
        time.sleep(1/240)

    p.disconnect()
