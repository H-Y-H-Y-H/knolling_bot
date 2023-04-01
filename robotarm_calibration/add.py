import numpy as np

split_flag = False

xyz_nn = np.loadtxt('nn_data_xyz/left_012020_002005_basespring/real_xyz_nn.csv')
print(xyz_nn)
cmd_nn = np.loadtxt('nn_data_xyz/left_012020_002005_basespring/cmd_nn.csv')
print(cmd_nn)
real_nn = np.loadtxt('nn_data_xyz/left_012020_002005_basespring/real_nn.csv')
print(real_nn)

if split_flag == True:
    # cmd_nn[:, 1] = cmd_nn[:, 2]
    # real_nn[:, 1] = real_nn[:, 2]
    cmd_nn_split = np.delete(cmd_nn, [0, 5], axis=1)
    real_nn_split = np.delete(real_nn, [0, 5], axis=1)
    input_data_split = np.concatenate((xyz_nn, real_nn_split), axis=1)
    np.savetxt('nn_data/all_distance_0035006_free/xyz_real_nn_split.csv', input_data_split)
    np.savetxt('nn_data/all_distance_0035006_free/cmd_nn_split.csv', cmd_nn_split)
    np.savetxt('nn_data/all_distance_0035006_free/real_nn_split.csv', real_nn_split)
else:
    input_data = np.concatenate((xyz_nn, real_nn), axis=1)
    print(input_data)
    np.savetxt('nn_data_xyz/left_012020_002005_basespring/xyz_real_nn.csv', input_data)

