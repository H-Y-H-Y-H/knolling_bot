import os
import numpy as np
import matplotlib.pyplot as plt


name = 'flowing-moon-116_min'
os.makedirs(f'{name}/comb/',exist_ok=True)
num_solu = 100

def visualize_all_solutions():
    for i in range(num_solu):
        print(i)
        img_list = []
        for pref in range(4):
            img_colum = []
            for config in range(3):
                # img_gt = plt.imread(f'{name}/{pref*3+config}/output_images10/{i}/0.jpg')
                img_pred = plt.imread(f'{name}/{pref*3+config}/output_images10/{i}/1.jpg')
                img_colum.append(img_pred)
            img_colum = np.concatenate((img_colum),axis=1)
            img_list.append(img_colum)
        img_combine = np.concatenate((img_list),axis=0)
        plt.imsave(f'{name}/comb/%d.jpg'%i,img_combine)
# visualize_all_solutions()
def demo_info():
    import shutil
    demo_id = 24
    n_solu = [1,3,6,11]

    for i in n_solu:
        data_info = np.loadtxt(f'{name}/{i}/num_10_pred_1.txt')
        info_state = data_info[demo_id]
        np.savetxt(f'{name}/comb/test{demo_id}({i}).txt',info_state)
        shutil.copy(src=f'{name}/{i}/output_images10/{demo_id}/1.jpg',
                    dst=f'{name}/comb/test{demo_id}({i}).jpg')

demo_info()
import cv2
def comb_all():
    img_list = []
    num_i = 6
    num_j = 9
    for i in range(num_i):
        img_list_sub = []
        for j in range(num_j):
            img_pred = plt.imread(f'{name}/comb/%d.jpg'%(i*num_i+j))
            img_list_sub.append(img_pred)
        img_list.append(np.concatenate((img_list_sub),axis=1))
    img_list = np.concatenate((img_list),axis=0)
    img_list = cv2.resize(img_list,(1920,1080))

    plt.imsave(f'{name}/comb/all_{(num_i*num_j)}.jpg',img_list)


# comb_all()