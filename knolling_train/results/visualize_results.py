import os
import numpy as np

name = 'avid-water-128'

DATAROOT = "C:/Users/yuhan/Downloads/learning_data_512_large/"

for cfg in range(4,5):
    dataset_path = DATAROOT + '/cfg_%d/' % cfg
    for NUM_objects in range(10,11):
        print('load data:', NUM_objects)

        raw_data = np.loadtxt(dataset_path + 'labels_after/num_%d.txt' % NUM_objects)

        # raw_data = np.loadtxt(dataset_path + 'real_before/num_%d_d3.txt' % NUM_objects)
        raw_data = raw_data [int(len(raw_data) * 0.8):]
        # raw_data = raw_data[3]
        # raw_data = np.asarray([raw_data]*5)

        results = np.loadtxt(name + '/outputs.csv')

        print(raw_data)


        # num_data = int(len(raw_data))
        # raw_data = raw_data[-num_data:]
        for i in range(NUM_objects):
            raw_data[:, i * 5:i * 5 + 2] = results[:, i * 2:i * 2 + 2]
            raw_data[:,i*5+4] = 0
        log_folder = name + '/cfg_%d/pred_after/' % cfg
        os.makedirs(log_folder, exist_ok=True)
        np.savetxt(log_folder + '/num_%d.txt' % NUM_objects, raw_data)

        print(raw_data)






