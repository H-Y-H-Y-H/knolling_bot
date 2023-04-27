import sys
sys.path.append('../')
from knolling_lego_yolov8 import *

if __name__ == '__main__':

    command = 'knolling'

    if command == 'knolling':

        num_2x2 = 4
        num_2x3 = 4
        num_2x4 = 4
        total_offset = [0.15, 0.1, 0]
        grasp_order = [0, 1, 2]
        gap_item = 0.015
        gap_block = 0.02
        random_offset = False
        real_operate = False
        obs_order = 'sim_image_obj'
        check_detection_loss = False
        obs_img_from = 'env'
        use_yolo_pos = False


        env = Arm(is_render=False, urdf_path='../urdf/')
        env.get_parameters(num_2x2=num_2x2, num_2x3=num_2x3, num_2x4=num_2x4,
                           total_offset=total_offset, grasp_order=grasp_order,
                           gap_item=gap_item, gap_block=gap_block,
                           real_operate=real_operate, obs_order=obs_order,
                           random_offset=random_offset, check_detection_loss=check_detection_loss,
                           obs_img_from=obs_img_from, use_yolo_pos=use_yolo_pos)
        evaluations = 1

        for i in range(evaluations):
            image_trim = env.change_config()
            image_chaotic = env.reset()
        print(image_trim.shape)
        print(image_chaotic.shape)