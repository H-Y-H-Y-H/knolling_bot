import numpy as np

target_path = '/home/zhizhuo/ADDdisk/Create Machine Lab/knolling_dataset/learning_data_506/cfg_1/'
before_path = target_path + 'labels_before/'
after_path = target_path + 'labels_after/'

configuration = np.arange(1, 2)
num_range = np.arange(9, 11)

for i in range(len(configuration)):

    for j in range(len(num_range)):

        data_1 = np.loadtxt(after_path + 'num_%d_5000.txt' % num_range[j])
        data_2 = np.loadtxt(after_path + 'num_%d_10000.txt' % num_range[j])
        data_3 = np.loadtxt(after_path + 'num_%d_15000.txt' % num_range[j])
        data_4 = np.loadtxt(after_path + 'num_%d_20000.txt' % num_range[j])
        data_5 = np.loadtxt(after_path + 'num_%d_25000.txt' % num_range[j])
        data_6 = np.loadtxt(after_path + 'num_%d_30000.txt' % num_range[j])
        data_7 = np.loadtxt(after_path + 'num_%d_35000.txt' % num_range[j])
        data_8 = np.loadtxt(after_path + 'num_%d_40000.txt' % num_range[j])
        data_9 = np.loadtxt(after_path + 'num_%d_45000.txt' % num_range[j])
        data_10 = np.loadtxt(after_path + 'num_%d_50000.txt' % num_range[j])
        data_11 = np.loadtxt(after_path + 'num_%d_55000.txt' % num_range[j])
        data_12 = np.loadtxt(after_path + 'num_%d_60000.txt' % num_range[j])
        data_13 = np.loadtxt(after_path + 'num_%d_65000.txt' % num_range[j])
        data_14 = np.loadtxt(after_path + 'num_%d_70000.txt' % num_range[j])

        data = np.concatenate((data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9, data_10, data_11, data_12, data_13, data_14), axis=0)
        np.savetxt(after_path + 'num_%d.txt' % num_range[j], data)

        data_1 = np.loadtxt(before_path + 'num_%d_5000.txt' % num_range[j])
        data_2 = np.loadtxt(before_path + 'num_%d_10000.txt' % num_range[j])
        data_3 = np.loadtxt(before_path + 'num_%d_15000.txt' % num_range[j])
        data_4 = np.loadtxt(before_path + 'num_%d_20000.txt' % num_range[j])
        data_5 = np.loadtxt(before_path + 'num_%d_25000.txt' % num_range[j])
        data_6 = np.loadtxt(before_path + 'num_%d_30000.txt' % num_range[j])
        data_7 = np.loadtxt(before_path + 'num_%d_35000.txt' % num_range[j])
        data_8 = np.loadtxt(before_path + 'num_%d_40000.txt' % num_range[j])
        data_9 = np.loadtxt(before_path + 'num_%d_45000.txt' % num_range[j])
        data_10 = np.loadtxt(before_path + 'num_%d_50000.txt' % num_range[j])
        data_11 = np.loadtxt(before_path + 'num_%d_55000.txt' % num_range[j])
        data_12 = np.loadtxt(before_path + 'num_%d_60000.txt' % num_range[j])
        data_13 = np.loadtxt(before_path + 'num_%d_65000.txt' % num_range[j])
        data_14 = np.loadtxt(before_path + 'num_%d_70000.txt' % num_range[j])

        data = np.concatenate((data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9, data_10, data_11, data_12, data_13, data_14), axis=0)
        np.savetxt(before_path + 'num_%d.txt' % num_range[j], data)

        index_1 = np.loadtxt(after_path + 'num_%d_5000_flag.txt' % num_range[j])
        index_2 = np.loadtxt(after_path + 'num_%d_10000_flag.txt' % num_range[j])
        index_3 = np.loadtxt(after_path + 'num_%d_15000_flag.txt' % num_range[j])
        index_4 = np.loadtxt(after_path + 'num_%d_20000_flag.txt' % num_range[j])
        index_5 = np.loadtxt(after_path + 'num_%d_25000_flag.txt' % num_range[j])
        index_6 = np.loadtxt(after_path + 'num_%d_30000_flag.txt' % num_range[j])
        index_7 = np.loadtxt(after_path + 'num_%d_35000_flag.txt' % num_range[j])
        index_8 = np.loadtxt(after_path + 'num_%d_40000_flag.txt' % num_range[j])
        index_9 = np.loadtxt(after_path + 'num_%d_45000_flag.txt' % num_range[j])
        index_10 = np.loadtxt(after_path + 'num_%d_50000_flag.txt' % num_range[j])
        index_11 = np.loadtxt(after_path + 'num_%d_55000_flag.txt' % num_range[j])
        index_12 = np.loadtxt(after_path + 'num_%d_60000_flag.txt' % num_range[j])
        index_13 = np.loadtxt(after_path + 'num_%d_65000_flag.txt' % num_range[j])
        index_14 = np.loadtxt(after_path + 'num_%d_70000_flag.txt' % num_range[j])

        index = np.concatenate((index_1, index_2, index_3, index_4, index_5, index_6, index_7, index_8, index_9, index_10, index_11,
                               index_12, index_13, index_14), axis=0)
        np.savetxt(after_path + 'num_%d.txt' % num_range[j], index)

        index_1 = np.loadtxt(before_path + 'num_%d_5000_flag.txt' % num_range[j])
        index_2 = np.loadtxt(before_path + 'num_%d_10000_flag.txt' % num_range[j])
        index_3 = np.loadtxt(before_path + 'num_%d_15000_flag.txt' % num_range[j])
        index_4 = np.loadtxt(before_path + 'num_%d_20000_flag.txt' % num_range[j])
        index_5 = np.loadtxt(before_path + 'num_%d_25000_flag.txt' % num_range[j])
        index_6 = np.loadtxt(before_path + 'num_%d_30000_flag.txt' % num_range[j])
        index_7 = np.loadtxt(before_path + 'num_%d_35000_flag.txt' % num_range[j])
        index_8 = np.loadtxt(before_path + 'num_%d_40000_flag.txt' % num_range[j])
        index_9 = np.loadtxt(before_path + 'num_%d_45000_flag.txt' % num_range[j])
        index_10 = np.loadtxt(before_path + 'num_%d_50000_flag.txt' % num_range[j])
        index_11 = np.loadtxt(before_path + 'num_%d_55000_flag.txt' % num_range[j])
        index_12 = np.loadtxt(before_path + 'num_%d_60000_flag.txt' % num_range[j])
        index_13 = np.loadtxt(before_path + 'num_%d_65000_flag.txt' % num_range[j])
        index_14 = np.loadtxt(before_path + 'num_%d_70000_flag.txt' % num_range[j])

        index = np.concatenate((index_1, index_2, index_3, index_4, index_5, index_6, index_7, index_8, index_9, index_10, index_11,
             index_12, index_13, index_14), axis=0)
        np.savetxt(before_path + 'num_%d.txt' % num_range[j], index)