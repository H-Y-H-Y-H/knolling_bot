import numpy as np

configuration = np.arange(4, 5)
num_range = np.arange(10, 11)

start_evaluations = 0
end_evaluations =   1000000
step_num = 100
save_point = np.linspace(int((end_evaluations - start_evaluations) / step_num + start_evaluations), end_evaluations, step_num)

for i in configuration:

    target_path = '/home/zhizhuo/ADDdisk/Create Machine Lab/knolling_dataset/learning_data_518_large/cfg_%d/' % i
    before_path = target_path + 'labels_before/'
    after_path = target_path + 'labels_after/'
    index_path = target_path + 'index_flag/'

    for j in range(len(num_range)):

        total_data = []
        for s in save_point:
            data = np.loadtxt(after_path + 'num_%d_%d.txt' % (num_range[j], int(s)))
            total_data.append(data)
        total_data = np.asarray(total_data).reshape(-1, 50)
        np.savetxt(after_path + 'num_%d.txt' % num_range[j], total_data)

            # data_1 = np.loadtxt(after_path + 'num_%d_50000.txt' % num_range[j])
            # data_2 = np.loadtxt(after_path + 'num_%d_100000.txt' % num_range[j])
            # data_3 = np.loadtxt(after_path + 'num_%d_150000.txt' % num_range[j])
            # data_4 = np.loadtxt(after_path + 'num_%d_200000.txt' % num_range[j])
            # data_5 = np.loadtxt(after_path + 'num_%d_250000.txt' % num_range[j])
            # data_6 = np.loadtxt(after_path + 'num_%d_300000.txt' % num_range[j])
            # data_7 = np.loadtxt(after_path + 'num_%d_350000.txt' % num_range[j])
            # data_8 = np.loadtxt(after_path + 'num_%d_400000.txt' % num_range[j])
            # data_9 = np.loadtxt(after_path + 'num_%d_450000.txt' % num_range[j])
            # data_10 = np.loadtxt(after_path + 'num_%d_500000.txt' % num_range[j])
            # data_11 = np.loadtxt(after_path + 'num_%d_550000.txt' % num_range[j])
            # data_12 = np.loadtxt(after_path + 'num_%d_600000.txt' % num_range[j])
            # data_13 = np.loadtxt(after_path + 'num_%d_650000.txt' % num_range[j])
            # data_14 = np.loadtxt(after_path + 'num_%d_700000.txt' % num_range[j])
            # data_15 = np.loadtxt(after_path + 'num_%d_750000.txt' % num_range[j])
            # data_16 = np.loadtxt(after_path + 'num_%d_800000.txt' % num_range[j])
            # data_17 = np.loadtxt(after_path + 'num_%d_850000.txt' % num_range[j])
            # data_18 = np.loadtxt(after_path + 'num_%d_900000.txt' % num_range[j])
            # data_19 = np.loadtxt(after_path + 'num_%d_950000.txt' % num_range[j])
            # data_20 = np.loadtxt(after_path + 'num_%d_1000000.txt' % num_range[j])
            #
            # data = np.concatenate((data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9, data_10,
            #                        data_11, data_12, data_13, data_14, data_15, data_16, data_17, data_18, data_19, data_20), axis=0)
            # np.savetxt(after_path + 'num_%d.txt' % num_range[j], data)

            # index_1 = np.loadtxt(index_path + 'num_%d_50000_flag.txt' % num_range[j])
            # index_2 = np.loadtxt(index_path + 'num_%d_100000_flag.txt' % num_range[j])
            # index_3 = np.loadtxt(index_path + 'num_%d_150000_flag.txt' % num_range[j])
            # index_4 = np.loadtxt(index_path + 'num_%d_200000_flag.txt' % num_range[j])
            # index_5 = np.loadtxt(index_path + 'num_%d_250000_flag.txt' % num_range[j])
            # index_6 = np.loadtxt(index_path + 'num_%d_300000_flag.txt' % num_range[j])
            # index_7 = np.loadtxt(index_path + 'num_%d_350000_flag.txt' % num_range[j])
            # index_8 = np.loadtxt(index_path + 'num_%d_400000_flag.txt' % num_range[j])
            # index_9 = np.loadtxt(index_path + 'num_%d_450000_flag.txt' % num_range[j])
            # index_10 = np.loadtxt(index_path + 'num_%d_500000_flag.txt' % num_range[j])
            # index_11 = np.loadtxt(index_path + 'num_%d_550000_flag.txt' % num_range[j])
            # index_12 = np.loadtxt(index_path + 'num_%d_600000_flag.txt' % num_range[j])
            # index_13 = np.loadtxt(index_path + 'num_%d_650000_flag.txt' % num_range[j])
            # index_14 = np.loadtxt(index_path + 'num_%d_700000_flag.txt' % num_range[j])
            # index_15 = np.loadtxt(index_path + 'num_%d_750000_flag.txt' % num_range[j])
            # index_16 = np.loadtxt(index_path + 'num_%d_800000_flag.txt' % num_range[j])
            # index_17 = np.loadtxt(index_path + 'num_%d_850000_flag.txt' % num_range[j])
            # index_18 = np.loadtxt(index_path + 'num_%d_900000_flag.txt' % num_range[j])
            # index_19 = np.loadtxt(index_path + 'num_%d_950000_flag.txt' % num_range[j])
            # index_20 = np.loadtxt(index_path + 'num_%d_1000000_flag.txt' % num_range[j])
            #
            # index = np.concatenate((index_1, index_2, index_3, index_4, index_5, index_6, index_7, index_8, index_9, index_10,
            #                         index_11, index_12, index_13, index_14, index_15, index_16, index_17, index_18, index_19, index_20), axis=0)
            # # index = np.concatenate((index_1, index_2), axis=0)
            # np.savetxt(index_path + 'num_%d_flag.txt' % num_range[j], index)