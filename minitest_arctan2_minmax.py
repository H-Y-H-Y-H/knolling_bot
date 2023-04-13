import numpy as np
from sklearn.preprocessing import MinMaxScaler


zzz_sin_pred = 0.99728
zzz_cos_pred = -0.073701

zzz_sin_tar = 0.83547
zzz_cos_tar = 0.54954

print(np.arctan2(zzz_sin_pred, zzz_cos_pred) * 0.25 * 180 / np.pi)
print(np.arctan2(zzz_sin_tar, zzz_cos_tar) * 0.25 * 180 / np.pi)

norm_parameters = np.array([[0.032, 0.016, 1, 1],
                            [0.016, 0.016, 0, 0]])

scaler = MinMaxScaler()
scaler.fit(norm_parameters)
print(scaler.data_max_)
print(scaler.data_min_)

test_data = np.array([[0.025, 0.016, 0.5, 0.5],
                      [0.025, 1, 0.5, 0.5]])
print(scaler.transform(test_data))


tar = np.array([[ 2.4000e-02,  1.6000e-02,  9.4498e-01, -3.2712e-01],
        [ 1.6000e-02,  1.6000e-02, -2.9888e-01,  9.5429e-01],
        [ 2.4000e-02,  1.6000e-02,  9.2183e-01, -3.8760e-01],
        [ 3.2000e-02,  1.6000e-02,  1.4701e-01, -9.8914e-01],
        [ 1.6000e-02,  1.6000e-02,  2.6296e-01, -9.6481e-01],
        [ 1.6000e-02,  1.6000e-02,  1.0000e+00,  2.7011e-04],
        [ 1.6000e-02,  1.6000e-02,  9.5252e-01, -3.0446e-01],
        [ 2.4000e-02,  1.6000e-02, -5.6461e-01,  8.2536e-01],
        [ 1.6000e-02,  1.6000e-02,  9.9097e-01, -1.3409e-01],
        [ 2.4000e-02,  1.6000e-02, -9.8618e-01,  1.6570e-01],
        [ 1.6000e-02,  1.6000e-02,  9.9967e-01, -2.5714e-02],
        [ 1.6000e-02,  1.6000e-02,  6.1063e-01, -7.9191e-01]])
pred = np.array([[ 0.0240,  0.0157,  0.9540, -0.2954],
        [ 0.0160,  0.0160, -0.2680,  0.9602],
        [ 0.0240,  0.0160,  0.9223, -0.3882],
        [ 0.0320,  0.0158,  0.1658, -0.9887],
        [ 0.0160,  0.0161,  0.2606, -0.9665],
        [ 0.0160,  0.0160,  1.0001,  0.0012],
        [ 0.0160,  0.0158,  0.9432, -0.3266],
        [ 0.0240,  0.0160, -0.5606,  0.8247],
        [ 0.0160,  0.0163,  0.9796, -0.1709],
        [ 0.0240,  0.0160, -0.9847,  0.1768],
        [ 0.0160,  0.0160,  0.9995, -0.0039],
        [ 0.0160,  0.0159,  0.5938, -0.8046]])

print(np.mean((tar - pred) ** 2))


zzz = np.arange(24, dtype=object).reshape(3, 2, 4) + 20

print(zzz.shape)
print(zzz.reshape(-1, 4))
scaler = np.array([[4, 3, 2, 1],
                   [11, 12, 13, 14]])
print(zzz / scaler[0, :])
# print(zzz + np.array([1, 2, 3, 4]))

# print(np.max(zzz.reshape(-1, 4), axis=0))