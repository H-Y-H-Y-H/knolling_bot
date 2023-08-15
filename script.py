import numpy as np

loss_list = []

name = 'devoted-terrain-29'

for cfg in range(5):
    loss_path = 'train_multi_knolling/results/devoted-terrain-29/' \
                'cfg_%d/test_loss_list_num_10.csv'%cfg
    loss_i = np.loadtxt(loss_path)
    loss_i_mean = np.mean(loss_i,axis=1)
    loss_list.append(loss_i_mean)


loss_list = np.mean(loss_list,axis=0)
print(loss_list.shape)
loss_idx = np.argsort(loss_list)
print(loss_idx[10:100])



