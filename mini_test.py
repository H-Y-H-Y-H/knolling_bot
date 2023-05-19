import numpy as np
from scipy.special import comb, perm
from itertools import combinations, permutations

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


all_num = 7
fac = []  # 定义一个列表存放因子
for i in range(1, all_num + 1):
    if all_num % i == 0:
        fac.append(i)
        continue
# print(fac)

arr = np.array([1,2,3])

index = np.array([1,1,1])
# print(arr[index])


rest_index = np.arange(6)

rest_index = np.delete(rest_index, np.where(rest_index == 2))
print(rest_index)

rest_index = np.delete(rest_index, np.where(rest_index == 4))
print(rest_index)


array = np.arange(12).reshape(3, 4)

print(np.where(array[:, 1] > 0))

print(min(134, 2))

print(50*0.05 + 80*0.2 + 96*0.25 + 115*0.25 + 100*0.25)

print(np.array([0.012] * 10).reshape(10, 1))

A=perm(3,2)
print(A)

box_num = 10
length_range = np.random.uniform(0.016, 0.048, size=(box_num, 1))
width_range = np.random.uniform(0.016, np.minimum(length_range, 0.036), size=(box_num, 1))
print(length_range)
print(width_range)