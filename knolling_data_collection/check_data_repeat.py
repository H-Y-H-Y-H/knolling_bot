import numpy as np

num_configuration = 1
before_after = 'after'
range_low = 8
range_high = 9

sample_num = 1000
sample_time = 1

for i in range(1, num_configuration + 1):
    target_path = '/home/zhizhuo/ADDdisk/Create Machine Lab/knolling_dataset/learning_data_506_30/cfg_%s/' % i
    for j in range(range_low, range_high):

        # data = np.loadtxt(target_path + 'labels_%s/num_%d.txt' % (before_after, j))
        data = np.loadtxt(target_path + 'labels_%s/num_8_1000.txt' % (before_after))
        print(len(data))
        total_repetition_rate = 0
        for m in range(sample_time):
            sample_index = np.random.choice(len(data), sample_num, replace=False)
            tar_line = data[sample_index]

            repeat_num = 0
            for n in range(len(tar_line)):
                data_compare = np.delete(data, sample_index[n], 0)
                result = data_compare - tar_line[n]
                repeat_index = np.where(np.all(result == 0, axis=1))[0]
                if len(repeat_index) == 0:
                    pass
                else:
                    print(data[sample_index[n]].reshape(j, -1))
                    print(data_compare[repeat_index].reshape(j, -1))
                    repeat_num += 1
                    print(f'data index {sample_index[n]} and index {repeat_index} has repeated in num {j} of cfg {i}')
            repetition_rate = repeat_num / sample_num
            total_repetition_rate += repetition_rate

            # print(f'one test end, repetition rate is {repetition_rate}')
        print(f'total repetition rate for num {j} in cfg {i} is {total_repetition_rate / sample_time}')

