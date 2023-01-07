import itertools
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from cycle import t, theta, theta_h, dthetah_dt

length = len(t)

def data_gen():
    for cnt in itertools.count():
        if cnt == length:
            return None
        yield t[cnt], theta[cnt]


def init():
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlim(-1.5, 1.5)
    del xdata[:]
    del ydata[:]
    line.set_data(xdata, ydata)
    return line,


fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.grid()
xdata, ydata, = [], []


def run(data):
    # update the data
    t, theta = data
    xdata.append(math.sin(theta))
    ydata.append(math.cos(theta))
    xmin, xmax = ax.get_xlim()

    line.set_data(xdata, ydata)

    return line,


ani = animation.FuncAnimation(fig, run, data_gen, interval=1, init_func=init)
plt.show()
