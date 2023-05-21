import numpy as np
configuration = 4
path = './cfg_%d_520/labels_after/' % configuration

total_data = []
total_images = 5

num_box_one_img = 10

for i in range(total_images):
    data_one_img = np.loadtxt(path + 'label_%d_%d.txt' % (num_box_one_img, i)).reshape(-1, )
    total_data.append(data_one_img)

total_data = np.asarray(total_data)
np.savetxt(path + 'num_%d.txt' % num_box_one_img, total_data)

