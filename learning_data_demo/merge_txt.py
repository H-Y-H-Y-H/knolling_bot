import numpy as np
configuration = 4
path = './%d/labels_before/' % configuration

total_data = []
total_images = 5

for i in range(total_images):
    data_one_img = np.loadtxt(path + 'label_%d.txt' % i).reshape(-1, )
    total_data.append(data_one_img)

total_data = np.asarray(total_data)
np.savetxt(path + 'num_%d.txt' % configuration, total_data)

