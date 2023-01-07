# 通过打印两个圆, 来展示步态

import matplotlib.pyplot as plt
from matplotlib import collections

# 基础数据 一个174cm高的两足机器人, 应该以 100步 52秒的速度前进
# 假设机器人的移动速度是1.2m/s
# 那么每一步的距离就是0.624米
import numpy as np
from numpy import pi, sin, cos, fmod


class State:  # python: class 是个框, 啥都往里装 就是
    def __init__(self):
        self.height = 1740
        sample1_step = 100
        sample1_time = 52
        self.velocity = 1.2
        self.single_step_time = sample1_time / sample1_step
        self.x_p_step = self.velocity * self.single_step_time
        print("单步持续时间", self.single_step_time)
        print("单步运动距离", self.x_p_step)

        self.leg_base_xy = None
        self.leg_center_angle = pi
        self.leg_front_limit = pi / 6
        self.leg_back_limit = -pi / 12
        # 大腿长
        self.leg1_length = 600
        self.leg1_keen_xy = None
        self.leg11_theta = None
        self.leg2_keen_xy = None
        self.leg21_theta = None


state = State()

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
    state.leg_base_xy = [head_x, 1750 - 2 * head_r - body_h]
    leg_base = plt.Circle(state.leg_base_xy, 30, linestyle='-', fill=True, color='red')
    plt.gcf().gca().add_patch(leg_base)


# python 支持 state.leg1_keen_xy  的写法, 但是不建议 因为为了统一信息来源

def plot_leg11(dtheta=0):
    # 向上为正
    theta = state.leg_center_angle + state.leg_front_limit + dtheta
    leg_xy_00 = [sin(theta) * state.leg1_length, cos(theta) * state.leg1_length]
    leg_xy_00 = np.array(leg_xy_00)
    lb = np.array(state.leg_base_xy)
    plot_line([state.leg_base_xy, leg_xy_00 + lb],'orange')
    state.leg1_keen_xy = leg_xy_00 + lb
    state.leg11_theta = theta


def plot_line(line,color):
    leg11 = collections.LineCollection([line], colors=color)
    plt.gcf().gca().add_collection(leg11)


def plot_leg21(dtheta=0):
    theta = state.leg_center_angle + state.leg_back_limit + dtheta
    leg_xy_00 = [sin(theta) * state.leg1_length, cos(theta) * state.leg1_length]
    leg_xy_00 = np.array(leg_xy_00)
    lb = np.array(state.leg_base_xy)
    plot_line([state.leg_base_xy, leg_xy_00 + lb],'cyan')
    state.leg2_keen_xy = leg_xy_00 + lb
    state.leg21_theta = theta


def gen_ctl_dtheta_sig_in_period(ctl_step_duration, ctl_step_dt, ctl_dtheta_sig, ctl_step_period):
    # 在整个周期中 每一步的序号, 分两条腿
    length = int(ctl_step_duration / ctl_step_dt)  # bug 浮点数转int丢失精度
    ctl_dtheta_sig_arr = [[], []]
    t_index = 0
    t_sum = 0
    for i in range(length):
        # 判断在哪个周期内
        now = i * ctl_step_dt
        if not now < t_sum + ctl_step_period[t_index]:
            t_sum = t_sum + ctl_step_period[t_index]
            t_index = t_index + 1
        ctl_dtheta_sig_arr[0].append(ctl_dtheta_sig[0][t_index])
        ctl_dtheta_sig_arr[1].append(ctl_dtheta_sig[1][t_index])
    return ctl_dtheta_sig_arr


def plot_leg12(dtheta):
    plot_line([state.leg1_keen_xy,(200,200)],'orange')


def job_print_robot_walk():
    plt.figure(figsize=(7, 6), )
    plt.axes([0.12, 0.11, 0.90 / 2, 0.88])
    plt.ion()
    perid = 10  # 秒
    dt = 0.02
    dtheta_leg11, dtheta_leg12 = 0.0, 0.0
    print("\n开始脚脚运动模拟")
    for t in np.arange(0, perid, dt):
        plt.cla()
        plot_ref_line()
        plot_body()

        ############################# controller start ###########
        """ 基于ld_四足_pos_controller的控制算法
        1. 定义整个步态周期时间
        2. 定义dt
        3. 定义步态关键参数在周期内的变化矩阵
        4. 定义步态关键周期的持续时间
        5. 生成指示数据
        """
        ctl_step_duration = 2 * state.single_step_time
        ctl_step_dt = dt
        ctl_dtheta_sig = [[-1, 1],
                          [1, -1]]
        ctl_step_period = [state.single_step_time, state.single_step_time]
        # todo codelyze ctl_
        ctl_dtheta_sig_in_period = gen_ctl_dtheta_sig_in_period(ctl_step_duration, ctl_step_dt,
                                                                ctl_dtheta_sig, ctl_step_period)
        # print(ctl_dtheta_sig_in_period)
        ######################### controller stop #############
        # 在一个单步周期内, 走过整个区间的速度
        w = (state.leg_front_limit - state.leg_back_limit) / state.single_step_time
        # 根据当前时间, 计算dtheta
        current_index_in_control_arr = int(fmod(t, ctl_step_duration) / dt)

        dtheta_leg11 = dtheta_leg11 + w * dt * ctl_dtheta_sig_in_period[0][current_index_in_control_arr]
        plot_leg11(dtheta_leg11)
        dtheta_leg12 = dtheta_leg12 + w * dt * ctl_dtheta_sig_in_period[1][current_index_in_control_arr]
        plot_leg21(dtheta_leg12)
        plt.text(-240, 1900, f'当前时间={t: 2.1f}秒')
        print(f'当前时间 {t:.1f} 左膝盖xy {state.leg2_keen_xy} '
              f'左大腿角 {state.leg11_theta} 右膝盖xy {state.leg1_keen_xy} 右大腿角 {state.leg21_theta}')

        plot_leg12(0)
        # plot_leg22(0)
        # todo: 最少需要多少个连杆呢?
        # 先写到legx3看看吧
        plt.pause(dt)
    plt.show()


if __name__ == '__main__':
    job_print_robot_walk()
