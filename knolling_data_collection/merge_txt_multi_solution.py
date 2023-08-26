import numpy as np
from tqdm import tqdm

configuration = np.arange(4, 5)
num_range = np.arange(5, 6)

start_evaluations = 0
end_evaluations =   800000
step_num = 400
solution_num = 5
save_point = np.linspace(int((end_evaluations - start_evaluations) / step_num + start_evaluations), end_evaluations, step_num)

def merge():

    for i in configuration:

        for m in range(solution_num):

            target_path = '../../knolling_dataset/learning_data_824/'
            before_path = target_path + 'labels_before_0/'
            after_path = target_path + 'labels_after_%s/' % m
            index_path = target_path + 'index_flag/'

            for j in tqdm(range(len(num_range))):

                total_data = []
                for s in save_point:
                    data = np.loadtxt(after_path + 'num_%d_%d.txt' % (num_range[j], int(s)))
                    total_data.append(data)
                total_data = np.asarray(total_data).reshape(-1, num_range[j] * 5)
                np.savetxt(after_path + 'num_%d.txt' % num_range[j], total_data)
                if m == 0:
                    total_data = []
                    for s in save_point:
                        data = np.loadtxt(before_path + 'num_%d_%d.txt' % (num_range[j], int(s)))
                        total_data.append(data)
                    total_data = np.asarray(total_data).reshape(-1, num_range[j] * 5)
                    np.savetxt(before_path + 'num_%d.txt' % num_range[j], total_data)

def add():
    base_path = '../../knolling_dataset/learning_data_817/'
    add_path = '../../knolling_dataset/learning_data_817_add/'
    # for m in tqdm(range(solution_num)):
    #     data_base = np.loadtxt(base_path + 'labels_after_%s/num_%d.txt' % (m, num_range[0]))
    #     data_add = np.loadtxt(add_path + 'labels_after_%s/num_%d.txt' % (m, num_range[0]))
    #     data_new = np.concatenate((data_base, data_add), axis=0)
    #     np.savetxt(base_path + 'labels_after_%s/num_%d_new.txt' % (m, num_range[0]), data_new)
    for m in tqdm(range(1)):
        data_base = np.loadtxt(base_path + 'labels_before_%s/num_%d.txt' % (m, num_range[0]))
        data_add = np.loadtxt(add_path + 'labels_before_%s/num_%d.txt' % (m, num_range[0]))
        data_new = np.concatenate((data_base, data_add), axis=0)
        np.savetxt(base_path + 'labels_before_%s/num_%d_new.txt' % (m, num_range[0]), data_new)
merge()
# add()