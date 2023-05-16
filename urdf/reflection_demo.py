import pybullet as p
import time
import math
import pybullet_data
import numpy as np
import random

x_grasp_accuracy=0.2
y_grasp_accuracy=0.2
z_grasp_accuracy=0.2
table_boundary = 0.05
x_low_obs = 0.05
x_high_obs = 0.3
y_low_obs = -0.15
y_high_obs = 0.15
z_low_obs = 0.005
z_high_obs = 0.05
x_grasp_interval = (x_high_obs - x_low_obs) * x_grasp_accuracy
y_grasp_interval = (y_high_obs - y_low_obs) * y_grasp_accuracy
z_grasp_interval = (z_high_obs - z_low_obs) * z_grasp_accuracy

cid = p.connect(p.SHARED_MEMORY)
if (cid < 0):
  p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setPhysicsEngineParameter(numSolverIterations=10)
p.setTimeStep(1. / 120.)
logId = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "visualShapeBench.json")
#useMaximalCoordinates is much faster then the default reduced coordinates (Featherstone)

p.resetDebugVisualizerCamera(cameraDistance=0.7,
                             cameraYaw=45,
                             cameraPitch=-45,
                             cameraTargetPosition=[0.1, 0, 0.4])

baseid = p.loadURDF('plane_zzz.urdf', useMaximalCoordinates=True)
#disable rendering during creation.


p.addUserDebugLine(
    lineFromXYZ=[x_low_obs - table_boundary, y_low_obs - table_boundary, z_low_obs],
    lineToXYZ=[x_high_obs + table_boundary, y_low_obs - table_boundary, z_low_obs])
p.addUserDebugLine(
    lineFromXYZ=[x_low_obs - table_boundary, y_low_obs - table_boundary, z_low_obs],
    lineToXYZ=[x_low_obs - table_boundary, y_high_obs + table_boundary, z_low_obs])
p.addUserDebugLine(
    lineFromXYZ=[x_high_obs + table_boundary, y_high_obs + table_boundary, z_low_obs],
    lineToXYZ=[x_high_obs + table_boundary, y_low_obs - table_boundary, z_low_obs])
p.addUserDebugLine(
    lineFromXYZ=[x_high_obs + table_boundary, y_high_obs + table_boundary, z_low_obs],
    lineToXYZ=[x_low_obs - table_boundary, y_high_obs + table_boundary, z_low_obs])


# if random.uniform(0, 1) > 0.5:
#     p.configureDebugVisualizer(lightPosition=[random.randint(1, 1), random.uniform(0, 1), 2],
#                                shadowMapResolution=8192, shadowMapIntensity=np.random.randint(5, 7) / 10)
# else:
#     p.configureDebugVisualizer(lightPosition=[random.randint(1, 1), random.uniform(-1, 0), 2],
#                                shadowMapResolution=8192, shadowMapIntensity=np.random.randint(5, 7) / 10)


background = np.random.randint(4, 5)
textureId = p.loadTexture(f"../urdf/img_{background}.png")
p.changeVisualShape(baseid, -1, textureUniqueId=textureId, specularColor=[0, 0, 0])

shift = [0, -0.02, 0]
meshScale = [0.1, 0.1, 0.1]
#the visual shape and collision shape can be re-used by all createMultiBody instances (instancing)
visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,
                                    fileName="duck.obj",
                                    rgbaColor=[1, 1, 1, 1],
                                    specularColor=[0, 0, 0],
                                    visualFramePosition=shift,
                                    meshScale=meshScale)
collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                          fileName="duck_vhacd.obj",
                                          collisionFramePosition=shift,
                                          meshScale=meshScale)

rangex = 5
rangey = 5
for i in range(rangex):
  for j in range(rangey):
    p.createMultiBody(baseMass=1,
                      baseInertialFramePosition=[0, 0, 0],
                      baseCollisionShapeIndex=collisionShapeId,
                      baseVisualShapeIndex=visualShapeId,
                      basePosition=[((-rangex / 2) + i) * meshScale[0] * 2,
                                    (-rangey / 2 + j) * meshScale[1] * 2, 1],
                      useMaximalCoordinates=True)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
p.stopStateLogging(logId)
p.setGravity(0, 0, -10)
p.setRealTimeSimulation(1)

colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 1, 1]]
currentColor = 0

while (1):
  time.sleep(1./240.)