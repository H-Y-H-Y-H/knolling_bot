import numpy as np


method_list = ['flowing-moon-116', 'eternal-sweep-1', 'restful-sweep-1',
               'dazzling-sweep-1',  'sage-sweep-1', 'magic-tree-145']

name_list = ['our method', 'no ll loss','no pos loss', 'no min loss', 'no overlap', 'no entropy loss' ]

loss_list = []
for i in range(6):
    method_loss = []

    for shift_id in range(1):
        method_name = method_list[i]
        ll_loss = np.loadtxt(f'{method_name}_min/{shift_id}/ll_loss10.csv').mean(1)
        pos_loss = np.loadtxt(f'{method_name}_min/{shift_id}/pos_loss10.csv').mean(1)
        min_loss = np.loadtxt(f'{method_name}_min/{shift_id}/ms_min_sample_loss10.csv').mean(1)
        overlap_loss = np.loadtxt(f'{method_name}_min/{shift_id}/overlap_loss10.csv')
        entropy_loss = np.loadtxt(f'{method_name}_min/{shift_id}/v_entropy_loss10.csv').mean(1)
        loss = min_loss + 0.01 *overlap_loss

        all_loss_each_method = [loss,ll_loss,min_loss,overlap_loss,entropy_loss,pos_loss]

        for l in all_loss_each_method:
            method_loss.append(l.mean())
            method_loss.append(l.std())
    loss_list.append(method_loss)
print(np.asarray(loss_list))
np.savetxt('eval_loss.csv', loss_list)




