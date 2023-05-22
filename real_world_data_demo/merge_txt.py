import numpy as np
configuration = 4
path = './home/zhizhuo/ADDdisk/Create Machine Lab/knolling_bot/real_world_data_demo/demo_3/labels_before/'

total_data = []
total_images = 5

num_box_one_img = 10

for i in range(total_images):
    data_one_img = np.loadtxt('/home/zhizhuo/ADDdisk/Create Machine Lab/knolling_bot/real_world_data_demo/demo_3/labels_before/label_10_0.txt').reshape(-1, )
    print(data_one_img.shape)
    total_data.append(data_one_img)

total_data = np.asarray(total_data)
np.savetxt(path + 'num_%d.txt' % num_box_one_img, total_data)

