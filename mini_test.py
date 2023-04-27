import numpy as np

cur = np.array([-0.027, 0.139, 0.045])
seg = np.array([0, 0.15, 0.049])
a = np.sign(seg - cur)
print(a)

zzz_array = np.arange(1)
zzz_array = np.delete(zzz_array, 0)
print(zzz_array)

lego_dict = {
            '16x16':1,
            '20x16':1,
            '20x20':1,
            '24x16':1,
            '24x20':1,
            '24x24':1,
            '28x16':1,
            '28x20':1,
            '28x24':1,
            '32x16':1,
            '32x20':1,
            '32x24':1,
        }
key_index = []
for i, data in enumerate(lego_dict):
    key_index.append(i)
print(key_index)

print(lego_dict[key_index])