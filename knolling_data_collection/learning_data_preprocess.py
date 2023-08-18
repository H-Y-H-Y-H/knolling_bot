import numpy as np
import os
from tqdm import tqdm

configuration = np.arange(4, 5)
range_low = 12
range_high = 13

origin_point = np.array([0, -0.2])

for i in configuration:
    target_path = '../../knolling_dataset/learning_data_730/'
    root_path = '../../knolling_dataset/learning_data_730/'
    os.makedirs(target_path, exist_ok=True)
    os.makedirs(root_path, exist_ok=True)
    print('over')
    for j in range(range_low, range_high):
        data = np.loadtxt(root_path + 'labels_after/num_%d.txt' % (j))
        # index = np.loadtxt(root_path + 'index_flag/num_%s_flag.txt' % j).reshape(-1, j * 2)
        new_data = []
        # new_index = []
        for m in tqdm(range(len(data))):
            # print('aaa')
            one_img_data = data[m].reshape(-1, 5)
            distance = np.linalg.norm(one_img_data[:, :2] - origin_point, axis=1)
            order = np.argsort(distance)
            new_data.append(one_img_data[order].reshape(-1, ))
            # one_img_index = index[m].reshape(2, -1)
            # new_index.append(one_img_index[:, order].reshape(-1, ))
        new_data = np.asarray(new_data)
        # new_index = np.asarray(new_index)
        np.savetxt(target_path + 'labels_after/num_%d.txt' % (j), new_data)
        # np.savetxt(target_path + 'index_flag/num_%s_flag.txt' % j, new_index)