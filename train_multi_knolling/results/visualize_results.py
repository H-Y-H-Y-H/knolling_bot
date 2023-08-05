import os
import numpy as np

name = 'neat-eon-16'
DATAROOT = "C:/Users/yuhan/Downloads/learning_data_803_3/"

for cfg in range(1):
    dataset_path = DATAROOT + '/labels_after_%d'%cfg
    for NUM_objects in range(5,6):
        print('load data:', NUM_objects)

        raw_data = np.loadtxt(dataset_path + '/num_%d.txt' % NUM_objects)

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






