import pybullet as p
import time
import pybullet_data
from math import sqrt, cos, sin, pi
import os
import numpy as np


physicsClient = p.connect(1)
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf")
startPos = [2, 0, 1]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])


cube1 = p.loadURDF("urdf/2x2.urdf", [0, 0, 0],
                   startOrientation, useFixedBase=1)
cube2 = p.loadURDF("urdf/2x3.urdf", [0, 0.02, 0],
                   startOrientation, useFixedBase=1)
cube3 = p.loadURDF("urdf/2x4.urdf", [0, 0.04, 0],
                   startOrientation, useFixedBase=1)


for i in range(20000):
    p.stepSimulation()

    time.sleep(1. / 240)


p.disconnect()
