import numpy as np

movie = True

if movie == False:
    loss = np.loadtxt('/home/zhizhuo/ADDdisk/Create Machine Lab/knolling_bot/real_world_data_demo/test_yolo_lw_loss/labels_after/label_loss.txt')
    print('mean: ', loss[-1])
    loss = loss[:, :10]
    print('max: ', np.max(loss, axis=0))
    print('min: ', np.min(loss, axis=0))

    print('std: ', np.std(loss, axis=0))

else:
    index_start = 0
    index_end = 100
    one_cfg_num = 10
    cfg_index = 0
    num_box_one_img = 1
    total_loss = []
    for i in range(index_start, index_end):
        if i % one_cfg_num == 0:

            ground_truth_label = input(f'In cfg {cfg_index}, please input the size of boxes (lwh):')
            cfg_index += 1
            ground_truth_label = list(ground_truth_label.split())

            for q in range(i, i + one_cfg_num):

                temp_data = np.array([float(d) for d in ground_truth_label]).reshape(num_box_one_img, -1)
                target = np.copy(temp_data)
                print(target)
                pred_label = np.loadtxt('/home/zhizhuo/ADDdisk/Create Machine Lab/knolling_dataset/yolo_pose4keypoints_tuning/origin_labels/origin_labels_pred/%012d.txt' % q).reshape(-1, 6)
                pred_label_temp = pred_label[:, :3]
                pred_label = pred_label[:, 3:]
                target_exist_list = []
                pred_exist_list = []
                temp_item_lw = []
                temp_item_pos = []
                temp_item_ori = []
                print('this is pred label\n', pred_label)
                print('this is known label\n', target)

                for j in range(len(pred_label)):
                    for m in range(len(target)):
                        if (np.abs(pred_label[j, 0] - target[m, 0]) < 0.002 and np.abs(
                                pred_label[j, 0] - target[m, 0]) < 0.002):
                            if m not in target_exist_list:
                                print(f'pred_label {j} match temp target {m}!')
                                target_exist_list.append(m)
                                pred_exist_list.append(j)
                                target[m, :2] = pred_label[j, :2]
                                # target[m, 2:4] = pred_label[j, :2]
                                temp_item_pos.append(target[m, :2])
                                temp_item_ori.append(target[m, :4])
                                break
                        elif (np.abs(pred_label[j, 1] - target[m, 0]) < 0.002 and np.abs(
                                pred_label[j, 0] - target[m, 1]) < 0.002):
                            if m not in target_exist_list:
                                print(f'pred_label {j} match temp target {m}! reverse')
                                target_exist_list.append(m)
                                pred_exist_list.append(j)
                                temp = pred_label[j][0]
                                pred_label[j][0] = pred_label[j][1]
                                pred_label[j][1] = temp
                                target[m, :2] = pred_label[j, :2]
                                # target[m, 2:4] = pred_label[j, :2]
                                temp_item_pos.append(target[m, :2])
                                temp_item_ori.append(target[m, :4])
                                break

                if len(target_exist_list) != len(target):
                    target_exist_list = np.asarray(target_exist_list)
                    pred_exist_list = np.asarray(pred_exist_list)
                    print(target_exist_list)
                    if len(target_exist_list) == 0:
                        rest_target = np.copy(target)
                        rest_target_index = np.arange(num_box_one_img)
                        rest_pred = pred_label
                        rest_pred_backup = rest_pred[:, [1, 0, 2]]
                        rest_pred_index = np.arange(num_box_one_img)
                        rest_pred_index = np.tile(rest_pred_index, 2)
                    else:
                        rest_target_index = np.delete(np.arange(num_box_one_img), target_exist_list)
                        rest_target = np.copy(np.delete(target, target_exist_list, axis=0))
                        rest_pred = np.delete(pred_label, pred_exist_list, axis=0)
                        rest_pred_backup = rest_pred[:, [1, 0, 2]]
                        rest_pred = np.concatenate((rest_pred, rest_pred_backup), axis=0)
                        rest_pred_index = np.delete(np.arange(num_box_one_img), pred_exist_list)
                        rest_pred_index = np.tile(rest_pred_index, 2)

                    print('rest_pred', rest_pred)
                    print('rest_target', rest_target)
                    for z in range(len(rest_target)):
                        add_index = np.argmin(np.linalg.norm(rest_pred[:, :2] - rest_target[z, :2], axis=1))
                        print('this is add', add_index)
                        if add_index + 1 > int(len(rest_pred) / 2):
                            print(
                                f'target {target[rest_target_index[z], :]} matches pred{pred_label[rest_pred_index[add_index], [1, 0]]}, reverse')
                            target[rest_target_index[z], :2] = pred_label[rest_pred_index[add_index], [1, 0]]
                            print('here target', target)
                            rest_pred[add_index, :2].fill(-2)
                            rest_pred[int(add_index - len(rest_pred) / 2), :2].fill(-2)
                            rest_target[z, :].fill(0)
                        else:
                            print(
                                f'target {target[rest_target_index[z], :]} matches pred{pred_label[rest_pred_index[add_index], [0, 1]]}')
                            target[rest_target_index[z], :2] = pred_label[rest_pred_index[add_index], [0, 1]]
                            print('here target', target)
                            rest_pred[add_index, :2].fill(-2)
                            rest_pred[int(add_index + len(rest_pred) / 2), :2].fill(-2)
                            rest_target[z, :].fill(0)

                loss = np.mean((target[:, :] - temp_data[:, :]), axis=0)
                print('this is temp data\n', temp_data)
                print('this is target\n', target)
                print('this is loss', loss)
                total_loss.append(loss)
                target = np.concatenate((pred_label_temp, temp_data, pred_label[:, 2].reshape(-1, 1)), axis=1)

                np.savetxt('/home/zhizhuo/ADDdisk/Create Machine Lab/knolling_dataset/yolo_pose4keypoints_tuning/origin_labels/%012d.txt' % q,
                           target, fmt='%.04f')

    if movie == True:
        total_loss.append(np.mean(total_loss, axis=0))
        total_loss = np.asarray(total_loss)
        np.savetxt('/home/zhizhuo/ADDdisk/Create Machine Lab/knolling_dataset/yolo_pose4keypoints_tuning/loss_%d.txt' % num_box_one_img,  total_loss)