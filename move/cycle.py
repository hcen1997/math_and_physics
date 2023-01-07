# 一个质点被轻杆固定在定轴上
import itertools
from math import sin

import matplotlib.pyplot as plt

theta = 0.00001  # rad # 因为不确定原理, 不能为0
theta_h = 0.00001
g = 9.8  # m/s^2
dthetah_dt = sin(theta) * g  # if f= ma and f = mg then m 被精密的消除了, 那么就是说, mg不是一个力, 是一个属性, 空间属性
dt = 0.0001
t = 0
log = []

for i in range(190000):
    t = t + dt
    dthetah_dt = sin(theta) * g
    dthetah = dt * dthetah_dt
    theta_h = theta_h + dthetah
    theta = theta + dt * theta_h
    # print(theta,theta_h,dthetah_dt)
    log.append((theta, theta_h, dthetah_dt, t))

del log[2-1::2]
del log[2-1::2]
del log[2-1::2]
del log[2-1::2]
del log[2-1::2]
del log[2-1::2]
del log[2-1::2]
theta, theta_h, dthetah_dt = [x[0] for x in log], [x[1] for x in log], [x[2] for x in log]
t = [x[3] for x in log]

if __name__ == '__main__':
    plt.plot(t, theta, label='theta')
    plt.plot(t, theta_h, label='theta_h')
    plt.plot(t, dthetah_dt, label='dthetah_dt')

    plt.show()

    # see some energy
    import numpy as np
    theta = np.array(theta)
    theta_h = np.array(theta_h)
    e_g = g* (np.cos(theta)+1)
    e_p = theta_h* 1* theta_h*0.5
    sum = e_g + e_p
    plt.plot(sum,label= 'sum')
    plt.plot(e_g,label= 'e_g')
    plt.plot(e_p,label= 'e_p')
    plt.show()
