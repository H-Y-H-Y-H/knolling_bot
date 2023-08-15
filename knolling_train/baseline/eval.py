import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
# from new_model import *

# model_name = '10cnn2d'
# model_name = '10lstm'
model_name = '10mlp'


# method_id_list = ['cosmic-serenity-151', 'lstm_result_r', 'mlp_result_r']
method_id_list = ['hearty-sweep-10', 'earnest-sweep-9','dazzling-sweep-8', 'splendid-sweep-7',
        'jolly-sweep-6', 'glowing-sweep-5', 'feasible-sweep-4', 'kind-sweep-3', 'ethereal-sweep-2', 'sweepy-sweep-1']
# method_id_list =['fiery-sweep-10','apricot-sweep-9','helpful-sweep-8','generous-sweep-7','stoic-sweep-6',
#             'fearless-sweep-5','effortless-sweep-4','amber-sweep-3','clean-sweep-2','solar-sweep-1']
# method_id_list = ['swept-sweep-10','deep-sweep-9','jumping-sweep-8','rural-sweep-7','flowing-sweep-6',
# 'northern-sweep-5', 'fresh-sweep-4','trim-sweep-3','whole-sweep-2','light-sweep-1']

all_data_log = []
mean_list = []
std_list = []
min_list = []
max_list = []
for method_id in range(10):
    for num_object in range(10,11):
        bl_loss_num = np.loadtxt(model_name+'/result_r_%s/loss_list_num_%d.csv' % (method_id_list[method_id], num_object))

        bl_loss_num = bl_loss_num[:, :num_object*2]

        print(bl_loss_num.shape)

        mean_loss = np.mean(bl_loss_num, 1)
        print(mean_loss.shape)

        n_mean = np.mean(mean_loss)
        # n_std  = np.std(mean_loss)
        # n_min  = np.min(mean_loss)
        # n_max  = np.max(mean_loss)

        mean_list.append(n_mean)
        # std_list.append(n_std)
        # min_list.append(n_min)
        # max_list.append(n_max)

# data_list = np.vstack((mean_list, std_list, min_list, max_list))
# all_data_log.append([mean_list, std_list, min_list, max_list])

# all_data_log = np.vstack((all_data_log))
np.savetxt('eval_data.csv', mean_list)
print(all_data_log)
