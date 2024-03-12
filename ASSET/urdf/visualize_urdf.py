import numpy as np
import pybullet as p
import pybullet_data as pd
import cv2
import random
import os
import csv
# from urdfpy import URDF

class visualize_env():

    def __init__(self, para_dict, knolling_para=None, lstm_dict=None, arrange_dict=None):

        self.para_dict = para_dict
        self.knolling_para = knolling_para

        self.kImageSize = {'width': 480, 'height': 480}
        self.init_pos_range = para_dict['init_pos_range']
        self.init_ori_range = para_dict['init_ori_range']
        self.init_offset_range = para_dict['init_offset_range']
        self.urdf_path = para_dict['urdf_path']
        self.object_urdf_path = para_dict['object_urdf_path']
        self.pybullet_path = pd.getDataPath()
        self.is_render = para_dict['is_render']
        self.save_img_flag = para_dict['save_img_flag']
        self.objects_num = para_dict['objects_num']

        self.x_low_obs = 0.03
        self.x_high_obs = 0.27
        self.y_low_obs = -0.14
        self.y_high_obs = 0.14
        self.z_low_obs = 0.0
        self.z_high_obs = 0.05
        x_grasp_accuracy = 0.2
        y_grasp_accuracy = 0.2
        z_grasp_accuracy = 0.2
        self.x_grasp_interval = (self.x_high_obs - self.x_low_obs) * x_grasp_accuracy
        self.y_grasp_interval = (self.y_high_obs - self.y_low_obs) * y_grasp_accuracy
        self.z_grasp_interval = (self.z_high_obs - self.z_low_obs) * z_grasp_accuracy
        self.table_boundary = 0.03
        self.table_center = np.array([(self.x_low_obs + self.x_high_obs) / 2, (self.y_low_obs + self.y_high_obs) / 2])

        self.gripper_interval = 0.01

        if self.is_render:
            p.connect(p.GUI, options="--width=1280 --height=720")
        else:
            p.connect(p.DIRECT)

        self.camera_parameters = {
            'width': 640.,
            'height': 480,
            'fov': 42,
            'near': 0.1,
            'far': 100.,
            'camera_up_vector':
                [1, 0, 0],  # I really do not know the parameter's effect.
            'light_direction': [
                0.5, 0, 1
            ],  # the direction is from the light source position to the origin of the world frame.

        }
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.150, 0, 0], #0.175
                                                               distance=0.4,
                                                               yaw=90,
                                                               pitch = -90,
                                                               roll=0,
                                                               upAxisIndex=2)
        self.projection_matrix = p.computeProjectionMatrixFOV(fov=self.camera_parameters['fov'],
                                                              aspect=self.camera_parameters['width'] /
                                                                     self.camera_parameters['height'],
                                                              nearVal=self.camera_parameters['near'],
                                                              farVal=self.camera_parameters['far'])
        p.resetDebugVisualizerCamera(cameraDistance=0.5,
                                     cameraYaw=45,
                                     cameraPitch=-45,
                                     cameraTargetPosition=[0.1, 0, 0])
        p.setAdditionalSearchPath(pd.getDataPath())
        # p.setPhysicsEngineParameter(numSolverIterations=10)
        p.setTimeStep(1. / 120.)

        self.grid_size = 5
        self.x_range = (-0.001, 0.001)
        self.y_range = (-0.001, 0.001)
        x_center = 0
        y_center = 0
        x_offset_values = np.linspace(self.x_range[0], self.x_range[1], self.grid_size)
        y_offset_values = np.linspace(self.y_range[0], self.y_range[1], self.grid_size)
        xx, yy = np.meshgrid(x_offset_values, y_offset_values)
        sigma = 0.01
        kernel = np.exp(-((xx - x_center) ** 2 + (yy - y_center) ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
        self.kernel = kernel / np.sum(kernel)

        self.main_demo_epoch = 0

        self.create_entry_num = 0

    def create_scene(self):

        if np.random.uniform(0, 1) > 0.5:
            p.configureDebugVisualizer(lightPosition=[np.random.randint(1, 2), np.random.uniform(0, 1.5), 2],
                                       shadowMapResolution=8192, shadowMapIntensity=np.random.randint(0, 1) / 10)
        else:
            p.configureDebugVisualizer(lightPosition=[np.random.randint(1, 2), np.random.uniform(-1.5, 0), 2],
                                       shadowMapResolution=8192, shadowMapIntensity=np.random.randint(0, 1) / 10)
        self.baseid = p.loadURDF(self.urdf_path + "plane_zzz.urdf", useMaximalCoordinates=True)

        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs - self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs],
            lineToXYZ=[self.x_high_obs + self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs - self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs],
            lineToXYZ=[self.x_low_obs - self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obs + self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs],
            lineToXYZ=[self.x_high_obs + self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obs + self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs],
            lineToXYZ=[self.x_low_obs - self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs])

        background = np.random.randint(1, 5)
        textureId = p.loadTexture(self.urdf_path + f"floor_{background}.png")
        p.changeVisualShape(self.baseid, -1, textureUniqueId=textureId, specularColor=[0, 0, 0])

        p.setGravity(0, 0, -10)

    def create_objects(self, pos_data=None, ori_data=None, lwh_data=None):
        # Randomly select 5-10 URDF files from a list of 300
        all_urdf_files = os.listdir(self.object_urdf_path)
        selected_urdf_files = random.sample(all_urdf_files, random.randint(5, 10))

        # Update self.objects_num to match the number of selected URDF files
        self.objects_num = len(selected_urdf_files)
        self.objects_index = []

        # Function to generate random color variations
        def get_random_color_variation(base_color):
            variation_intensity = 0.7  # Adjust this for more or less color variation
            color_variation = np.random.uniform(1 - variation_intensity, 1 + variation_intensity, size=3)
            if np.array_equal(base_color, np.array([0.7, 0.7, 0.7])):
                color_variation = 1.15
            return np.clip(base_color * color_variation, 0, 1)

        # Define base colors
        base_colors = {
            'red': np.array([1, 0, 0]),
            'black': np.array([0, 0, 0]),
            'blue': np.array([0, 0, 1]),
            'green': np.array([0, 1, 0]),
            'grey': np.array([0.7, 0.7, 0.7])
        }

        # Function to check if a new position is too close to existing objects
        def is_too_close(new_pos, existing_objects, min_distance=0.08):
            for _, obj_pos in existing_objects:
                if np.linalg.norm(np.array(new_pos) - np.array(obj_pos)) < min_distance:
                    return True
            return False

        # Generate random orientations for the objects
        rdm_ori_roll = np.pi / 2 * np.ones((self.objects_num, 1))  # Fixed roll value
        rdm_ori_pitch = np.random.uniform(self.init_ori_range[1][0], self.init_ori_range[1][1], size=(self.objects_num, 1))
        rdm_ori_yaw = np.random.uniform(self.init_ori_range[2][0], self.init_ori_range[2][1], size=(self.objects_num, 1))
        rdm_ori = np.hstack([rdm_ori_roll, rdm_ori_pitch, rdm_ori_yaw])
        rdm_ori = np.zeros(shape=(self.objects_num, 3))
        # rdm_ori[:, 0] = np.pi / 2

        x_offset = np.random.uniform(self.init_offset_range[0][0], self.init_offset_range[0][1])
        y_offset = np.random.uniform(self.init_offset_range[1][0], self.init_offset_range[1][1])

        # Place the objects, ensuring they don't overlap
        for i in range(self.objects_num):
            placement_successful = False
            while not placement_successful:
                # Generate a new position for the object
                rdm_pos_x = np.random.uniform(self.init_pos_range[0][0], self.init_pos_range[0][1])
                rdm_pos_y = np.random.uniform(self.init_pos_range[1][0], self.init_pos_range[1][1])
                rdm_pos_z = np.random.uniform(self.init_pos_range[2][0], self.init_pos_range[2][1])
                new_pos = [rdm_pos_x + x_offset, rdm_pos_y + y_offset, rdm_pos_z]

                # Check if the new position is too close to any existing objects
                if not is_too_close(new_pos, self.objects_index):
                    urdf_file = self.object_urdf_path + selected_urdf_files[i % len(selected_urdf_files)]
                    urdf_file = self.object_urdf_path + 'utilityknife_1_L1.02_T0.98.urdf'
                    obj_id = p.loadURDF(urdf_file,
                                        basePosition=new_pos,
                                        baseOrientation=p.getQuaternionFromEuler(rdm_ori[i]),
                                        flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
                
                    # Select a random base color and apply variation
                    chosen_color = random.choice(list(base_colors.values()))
                    color_variation = get_random_color_variation(chosen_color)
                
                    p.changeVisualShape(obj_id, -1, rgbaColor=[*color_variation, 1])
                    self.objects_index.append((obj_id, new_pos))
                    placement_successful = True

        # Rest of the existing code for object setup
        for _ in range(int(100)):
            p.stepSimulation()
            if self.is_render == True:
                pass
                # time.sleep(1/96)
        # while True:
        #     p.stepSimulation()

        p.changeDynamics(self.baseid, -1, lateralFriction=self.para_dict['base_lateral_friction'],
                        contactDamping=self.para_dict['base_contact_damping'],
                        contactStiffness=self.para_dict['base_contact_stiffness'])

    def create_arm(self):
        self.arm_id = p.loadURDF(self.urdf_path + "robot_arm928/robot_arm.urdf",
                                 basePosition=[-0.08, 0, -0.01], useFixedBase=True,
                                 flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)

        p.changeDynamics(self.arm_id, 7, lateralFriction=self.para_dict['gripper_lateral_friction'],
                         contactDamping=self.para_dict['gripper_contact_damping'],
                         contactStiffness=self.para_dict['gripper_contact_stiffness'])
        p.changeDynamics(self.arm_id, 8, lateralFriction=self.para_dict['gripper_lateral_friction'],
                         contactDamping=self.para_dict['gripper_contact_damping'],
                         contactStiffness=self.para_dict['gripper_contact_stiffness'])
        pass

    def get_images(self, image_count, save_dir, start_index):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)  # Create the directory if it doesn't exist

        for i in range(image_count):
            # Capture and save image
            (width, length, image, _, _) = p.getCameraImage(width=640,
                                                        height=480,
                                                        viewMatrix=self.view_matrix,
                                                        projectionMatrix=self.projection_matrix,
                                                        renderer=p.ER_BULLET_HARDWARE_OPENGL)
            img = image[:, :, :3]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert to BGR for saving with OpenCV
            cv2.imwrite(os.path.join(save_dir, f'image_{start_index + i}.png'), img)  # Save image in the specified directory

    def get_lwh_list(self, lwh_base_dir):
        self.lwh_list = {}

        for root, dirs, files in os.walk(lwh_base_dir):
            for file in files:
                if file.endswith(".csv"):
                    file_path = os.path.join(root, file)
                    with open(file_path, mode = "r") as csvfile:
                        csv_reader = csv.DictReader(csvfile)
                        for row in csv_reader:
                            dimensions = row['BoundingBoxDimensions (cm)'].strip("[]").split(",")
                            if dimensions:
                                self.lwh_list[os.path.splitext(file)[0]] = list(map(float, dimensions))

    def setup(self):
        image_count = 0
        save_dir = 'yolo_new_objects_images'
        lwh_base_dir = 'OpensCAD_generate/generated_stl'
        self.get_lwh_list(lwh_base_dir)
        while image_count < 30:
            p.resetSimulation()
            self.create_scene()
            self.create_arm()
            self.create_objects()
            self.get_images(1, save_dir, image_count)
            image_count += 1

if __name__ == '__main__':

    np.random.seed(4)
    para_dict = {
        'reset_pos': np.array([0, 0, 0.12]), 'reset_ori': np.array([0, np.pi / 2, 0]),
        'save_img_flag': True,
        'init_pos_range': [[0.03, 0.27], [-0.15, 0.15], [0.01, 0.02]], 'init_offset_range': [[-0, 0], [-0, 0]],
        'init_ori_range': [[0, 0], [0, 0], [0, 0]],
        'objects_num': 3,
        'is_render': True,
        'gripper_lateral_friction': 1, 'gripper_contact_damping': 1, 'gripper_contact_stiffness': 50000,
        'box_lateral_friction': 1, 'box_contact_damping': 1, 'box_contact_stiffness': 50000,
        'base_lateral_friction': 1, 'base_contact_damping': 1, 'base_contact_stiffness': 50000,
        'object_urdf_path': 'OpensCAD_generate/urdf_file/',
        'urdf_path': './',
    }

    visualize_env = visualize_env(para_dict=para_dict)
    visualize_env.setup()
