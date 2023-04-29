# from simple_pid import PID
# import time
# import matplotlib.pyplot as plt
#
# class heater:
#     def __init__(self):
#         self.temp = 25
#
#     def update(self, power, dt):
#         if power > 0:
#             # 加热时房间温度随变量power和时间变量dt 的变化
#             self.temp += 2 * power * dt
#         # 表示房间的热量损失
#         # self.temp -= 0.5 * dt
#         return self.temp
#
# if __name__ == '__main__':
#     # 将创建的模型写进主函数
#     heater = heater()
#     temp = heater.temp
#     # 设置PID的三个参数，以及限制输出
#     pid = PID(100, 0.0000001, 0.000001, setpoint=temp)
#     pid.output_limits = (0, None)
#     # 用于设置时间参数
#     start_time = time.time()
#     last_time = start_time
#     # 用于输出结果可视化
#     setpoint, y, x = [], [], []
#     # 设置系统运行时间
#     step = 1
#     while time.time() - start_time < 1:
#
#         # 设置时间变量dt
#         current_time = time.time()
#         dt = (current_time - last_time)
#         print('this is dt', dt)
#         dt = 0.000005
#
#         # 变量temp在整个系统中作为输出，变量temp与理想值之差作为反馈回路中的输入，通过反馈回路调节变量power的变化。
#         power = pid(temp)
#         print('this is power', power)
#         temp = heater.update(power, dt)
#
#         # 用于输出结果可视化
#         x += [current_time - start_time]
#         y += [temp]
#         setpoint += [pid.setpoint]
#         # 用于变量temp赋初值
#         if current_time - start_time > 0:
#             print('aaaaaaa')
#             pid.setpoint = 75
#
#         last_time = current_time
#
#         if step > 100000:
#             break
#         else:
#             step += 1
#
#     # 输出结果可视化
#     plt.plot(x, setpoint, label='target')
#     plt.plot(x, y, label='PID')
#     plt.xlabel('time')
#     plt.ylabel('temperature')
#     plt.legend()
#     plt.show()
#
#     print('aaaa')


from simple_pid import PID
import numpy as np
pid = PID(0.5, 0.1, 0.00)
pid.setpoint = 30
print(pid(10))
print('aaa')
pid.setpoint = np.array([3200, 2200, 2700])
# pid.setpoint = 3000
# pid.output_limits = (0, None)

# tar_pos = np.array([3000, 2000, 2500])
# cur_pos = np.array([3200, 2200, 2700])
# # cur_pos = np.array([3000, 2000, 2500])
# # tar_pos = np.array([3200, 2200, 2700])
# for i in range(50):
#     error = tar_pos - cur_pos
#     print('this is cur', cur_pos)
#     print('this is error', error)
#     output = pid(error)
#     print('this is output', output)
#     print('\n')
#     cur_pos = (output + cur_pos) * 0.95