import numpy as np

configuration = np.arange(4, 5)
num_range = np.arange(5, 6)
solution_num = 3

start_evaluations = 0
end_evaluations =   1000
step_num = 10
save_point = np.linspace(int((end_evaluations - start_evaluations) / step_num + start_evaluations), end_evaluations, step_num)

def merge():

    for i in configuration:

        for m in range(solution_num):

            target_path = '../../knolling_dataset/learning_data_803/'
            before_path = target_path + 'labels_before/'
            after_path = target_path + 'labels_after_%s/' % m
            index_path = target_path + 'index_flag/'

            for j in range(len(num_range)):

                total_data = []
                for s in save_point:
                    data = np.loadtxt(after_path + 'num_%d_%d.txt' % (num_range[j], int(s)))
                    total_data.append(data)
                total_data = np.asarray(total_data).reshape(-1, num_range[j] * 5)
                np.savetxt(after_path + 'num_%d.txt' % num_range[j], total_data)

def add():
    base_path = '../../knolling_dataset/learning_data_730/labels_after/num_12.txt'
    add_path = '../../knolling_dataset/learning_data_730/labels_after/num_12_300000.txt'
    data_base = np.loadtxt(base_path)
    data_add = np.loadtxt(add_path)
    print('here')

    data_new = np.concatenate((data_base, data_add), axis=0)
    np.savetxt('../../knolling_dataset/learning_data_730/labels_after/num_12_new.txt', data_new)
merge()
# add()