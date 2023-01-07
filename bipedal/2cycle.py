# 通过打印两个圆, 来展示步态

import matplotlib.pyplot as plt

# 基础数据 一个174cm高的两足机器人, 应该以 100步 52秒的速度前进
# 假设机器人的移动速度是1.2m/s
# 那么每一步的距离就是0.624米
import numpy as np
from numpy import pi, sin, cos

robot_h = 1740
sample1_step = 100
sample1_time = 52
robot_v = 1.2
single_step_time = sample1_time / sample1_step
x_p_step = robot_v * single_step_time

# 基础坐标轴  高为2米的线 (2000)
ref_h = 2000
# 参考线 x   长1米
ref_x = 1000
# 参考线位于-300处
ref_offset = -300


def plot_ref_line():
    plt.vlines(ref_offset, 0, ref_h, 'blue', ':', "垂直两米参考线")
    plt.hlines(ref_offset, 0, ref_x, 'black', ':', "水平1米参考线")
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


leg_base_xy = None


def plot_body():
    # 在0.5,1.75处画一个头 半径为0.12
    head_r = 120
    head_x = 500
    head = plt.Circle((0.5 * 1000, 1.75 * 1000 - head_r), head_r, linestyle='-', fill=False)
    plt.gcf().gca().add_patch(head)
    body_h = 700
    body_w = 200
    body = plt.Rectangle((head_x - body_w / 2, 1750 - 2 * head_r - body_h), body_w, body_h, linestyle='-', fill=False)
    plt.gcf().gca().add_patch(body)
    # 画出腿部基点圆
    global leg_base_xy
    leg_base_xy = [head_x, 1750 - 2 * head_r - body_h]
    leg_base = plt.Circle(leg_base_xy, 30, linestyle='-', fill=True, color='red')
    plt.gcf().gca().add_patch(leg_base)


leg_center_angle = pi
leg_front_limit = pi / 6
leg_back_limit = -pi / 12
# 大腿长
leg1_length = 600
leg1_keen_xy = None


def plot_leg11(dtheta=0):
    global leg_base_xy
    # 向上为正
    theta = leg_center_angle + leg_front_limit + dtheta
    leg_xy_00 = [sin(theta) * leg1_length, cos(theta) * leg1_length]
    leg_xy_00 = np.array(leg_xy_00)
    lb = np.array(leg_base_xy)
    # 4个值 两个点
    from matplotlib import collections
    line = [leg_base_xy, leg_xy_00 + lb]
    leg11 = collections.LineCollection([line], colors='orange')
    plt.gcf().gca().add_collection(leg11)
    global leg1_keen_xy
    leg1_keen_xy = leg_xy_00 + lb


leg2_keen_xy = None


def plot_leg21(dtheta=0):
    global leg_base_xy
    # 向上为正
    theta = leg_center_angle + leg_back_limit + dtheta
    leg_xy_00 = [sin(theta) * leg1_length, cos(theta) * leg1_length]
    leg_xy_00 = np.array(leg_xy_00)
    lb = np.array(leg_base_xy)
    # 4个值 两个点
    from matplotlib import collections
    line = [leg_base_xy, leg_xy_00 + lb]
    leg11 = collections.LineCollection([line], colors='cyan')
    plt.gcf().gca().add_collection(leg11)
    global leg2_keen_xy
    leg2_keen_xy = leg_xy_00 + lb


def job_print_robot_walk():
    plt.figure(figsize=(7, 6), )
    plt.axes([0.12, 0.11, 0.90 / 2, 0.88])
    plt.ion()
    perid = 10  # 秒
    dt = 0.1
    for t in np.arange(0, perid, dt):
        plt.cla()
        plot_ref_line()
        plot_body()

        # 在一个单步周期内, 走过整个区间的速度
        w = (leg_front_limit - leg_back_limit) / single_step_time
        # 根据当前时间, 计算dtheta
        dtheta_leg11 = w * t
        plot_leg11(dtheta_leg11)
        dtheta_leg11 = -w * t  # 两腿摆动周期 所以取负值
        plot_leg21(dtheta_leg11)
        plt.text(-240, 1900, f'当前时间={t: 2.1f}秒')
        plt.pause(0.1)
    plt.show()


if __name__ == '__main__':
    job_print_robot_walk()
