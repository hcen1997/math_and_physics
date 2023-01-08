# 通过展示单腿的支撑态, 来确定重心的移动过程
import math

import matplotlib.pyplot as plt
from matplotlib import collections

import numpy as np
from numpy import pi, sin, cos, fmod


class State:
    def __init__(self):
        self.height = 1.740
        self.velocity = 1.2
        self.single_step_time = 0.5
        self.x_p_step = self.velocity * self.single_step_time
        print("单步持续时间", self.single_step_time)
        print("单步运动距离", self.x_p_step)

        self.head_r = 0.261 / 2
        self.head_x = 0.500
        self.body_h = 0.556
        self.body_w = 0.237
        self.leg_base_xy = np.array([self.head_x, self.height - 2 * self.head_r - self.body_h])
        self.leg_center_angle = pi
        self.leg_front_limit = pi / 6
        self.leg_back_limit = -pi / 12
        # 大腿长
        self.leg1_length = 384
        self.leg1_knee_xy = np.array([0, 0])
        self.leg11_theta = 0
        self.leg2_knee_xy = np.array([0, 0])
        self.leg21_theta = 0

        # 小腿长
        self.leg2_length = 0.428
        self.leg1_ankle_xy = np.array([0, 0])
        self.leg12_theta = 0
        self.leg2_ankle_xy = np.array([0, 0])
        self.leg22_theta = 0
        # 脚踝
        self.leg_ankle_R = 0.048
        # 脚掌
        self.leg3_length = 0.237
        self.leg3_back_length = 0.041
        self.leg3_mid_length = 0.140
        self.leg3_front_length = 0.06
        self.leg13_theta = 0
        self.leg13_front_xy = np.array([0, 0])
        self.leg13_back_xy = np.array([0, 0])


class Control:
    def __init__(self):
        self.now = 0.0
        self.dt = 0.05
        self.cnt = 0
        self.control_loop_total = 20
        self.ctl_index = self.cnt % self.control_loop_total

        self.w_leg1 = (state.leg_front_limit - state.leg_back_limit) / state.single_step_time
        self.theta_leg11 = 0
        self.theta_leg12 = 0
        # 质心z轴位置
        self.z_pos = 0.0

    def add_control_index(self):
        self.cnt = self.cnt + 1
        self.ctl_index = self.cnt % self.control_loop_total


state = State()
ctl = Control()


def plot_ref_line():
    # 基础坐标轴  高为2米的线
    ref_h = 2
    # 参考线 x   长1米
    ref_x = 1
    # 参考线位于-300处
    ref_offset = -0.1
    plt.vlines(ref_offset, 0, ref_h, 'blue', ':', "垂直两米参考线")
    # plt.hlines(ref_offset, 0, ref_x, 'black', ':', "水平1米参考线")
    plt.hlines(0, ref_offset - 0.001, ref_x * 1.2, 'black', '--', "地面")
    plt.hlines(0, ref_offset - 0.002, ref_x * 1.2, 'black', '--', )
    plt.hlines(0, ref_offset - 0.003, ref_x * 1.2, 'black', '--', )
    plt.hlines(0, ref_offset - 0.004, ref_x * 1.2, 'black', '--', )

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def plot_body():
    # 画出腿部基点圆
    leg_base = plt.Circle(state.leg_base_xy, 0.03, linestyle='-', fill=True, color='red', )
    plt.gcf().gca().add_patch(leg_base)
    body_xy = state.leg_base_xy - [state.body_w / 2, 0]
    body = plt.Rectangle(body_xy, state.body_w, state.body_h, linestyle='-',
                         fill=False)
    plt.gcf().gca().add_patch(body)
    head_xy = state.leg_base_xy + [0, state.body_h + state.head_r]
    head = plt.Circle(head_xy, state.head_r, linestyle='-', fill=False)
    plt.gcf().gca().add_patch(head)


# python 支持 state.leg1_knee_xy  的写法, 但是不建议 因为为了统一信息来源

def plot_leg11(dtheta=0):
    # 向上为正
    theta = state.leg_center_angle + state.leg_front_limit + dtheta
    leg_xy_00 = [sin(theta) * state.leg1_length, cos(theta) * state.leg1_length]
    leg_xy_00 = np.array(leg_xy_00)
    lb = np.array(state.leg_base_xy)
    plot_line([state.leg_base_xy, leg_xy_00 + lb], 'orange')
    state.leg1_knee_xy = leg_xy_00 + lb
    state.leg11_theta = theta


def plot_line(line, color):
    leg11 = collections.LineCollection([line], colors=color)
    plt.gcf().gca().add_collection(leg11)


def plot_leg21(dtheta=0):
    theta = state.leg_center_angle + state.leg_back_limit + dtheta
    leg_xy_00 = [sin(theta) * state.leg1_length, cos(theta) * state.leg1_length]
    leg_xy_00 = np.array(leg_xy_00)
    lb = np.array(state.leg_base_xy)
    plot_line([state.leg_base_xy, leg_xy_00 + lb], 'cyan')
    state.leg2_knee_xy = leg_xy_00 + lb
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


"""
画线子过程
1. 基点
2. plot_line([base, leg_xy_00 + base], <color_you_should_like>)
"""


def plot_leg12(dtheta):
    # plot_line([state.leg1_knee_xy, (200, 200)], 'orange')

    theta = state.leg11_theta + dtheta
    color = 'orange'
    # 大腿处于+-15度边界的时候, 腿平行, 其他时刻腿垂直
    if abs(state.leg11_theta - state.leg_center_angle) < pi / 12 and state.index_in_loop < 10:
        color = 'red'
        # print(state.index_in_loop)
    else:
        # theta = state.leg_center_angle + dtheta
        color = 'orange'

    leg_xy_00 = np.array([sin(theta) * state.leg2_length, cos(theta) * state.leg2_length])

    state.leg1_ankle_xy = leg_xy_00 + np.array(state.leg1_knee_xy)
    plot_line([state.leg1_knee_xy, state.leg1_ankle_xy], color)  # 咦, 我只需要关心某些点就行了
    t = (state.leg1_ankle_xy - state.leg1_knee_xy) ** 2
    # print("leg2长度", math.sqrt(t[0] + t[1])) # leg2 长度没问题, 只是看起来有变化
    state.leg12_theta = theta


def plot_leg22(dtheta):
    theta = state.leg_center_angle  # + state.leg_back_limit + dtheta
    leg_xy_00 = np.array([sin(theta) * state.leg2_length, cos(theta) * state.leg2_length])
    state.leg2_ankle_xy = leg_xy_00 + np.array(state.leg2_knee_xy)
    plot_line([state.leg2_knee_xy, state.leg2_ankle_xy], 'cyan')  # 咦, 我只需要关心某些点就行了
    state.leg22_theta = theta


# log_
def plot_knee():
    knee1 = plt.Circle(state.leg1_knee_xy, 40, linestyle='-', fill=False, color='green')
    plt.gcf().gca().add_patch(knee1)
    # knee2 = plt.Circle(state.leg2_knee_xy, 40, linestyle='-', fill=False, color='green')
    # plt.gcf().gca().add_patch(knee2)


def plot_leg13(dtheta):
    leg3_back_length = 41
    print('脚踝距地面距离', state.leg1_ankle_xy[1])
    # 脚掌2百3 最大距离240
    # 这里简单写个离地面 20 高的运动情况
    foot_from_ground = state.leg1_ankle_xy[1]
    target_h = foot_from_ground - 80
    target_dtheta = np.arccos(target_h / (state.leg3_length - leg3_back_length))

    theta = state.leg_center_angle + target_dtheta  # + state.leg_back_limit + dtheta
    leg_xy_o = np.array(state.leg1_ankle_xy)
    leg_xy_front = np.array(
        [sin(theta) * (state.leg3_length - leg3_back_length), cos(theta) * (state.leg3_length - leg3_back_length)])
    # state.leg2_ankle_xy = leg_xy_00 + np.array(state.leg2_knee_xy)
    theta = theta - pi
    leg_xy_back = np.array([sin(theta) * (leg3_back_length), cos(theta) * (leg3_back_length)])
    plot_line([leg_xy_front + leg_xy_o, leg_xy_back + leg_xy_o], 'orange')  # 咦, 我只需要关心某些点就行了
    state.leg23_theta = theta + pi
    print("指尖距地面高度", (leg_xy_front + leg_xy_o)[1])


def job_print_robot_support():
    plt.figure(figsize=(7, 6), )
    plt.axes([0.12, 0.11, 0.90 / 2, 0.80])
    plt.ion()
    perid = 50  # 秒
    print("\n开始脚脚运动模拟")
    print("大腿角速度rad/s", ctl.w_leg1)
    info_xy = [0.8, 1.6]
    for t in np.arange(0, perid, ctl.dt):
        plt.cla()
        plot_ref_line()
        plot_body()
        plt.text(-0.05, 1.900, f'当前={t: 2.1f}秒')

        ############################# controller start ###########
        touch_down = [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        da = -pi / 24
        ctl_leg2_sig = [
            [da * 3, da * 2, da * 0.5, 0, 0, 0, 0, 0, 0, da * 1,
             da * 2, da * 3, da * 4, da * 5, da * 6, da * 7, da * 8, da * 7, da * 6, da * 4.5, ],
            [-pi / 2, -pi / 2, -pi / 2, -pi / 2, -pi / 2, -pi / 2, -pi / 2, -pi / 2, -pi / 2, -pi / 2, -pi / 2, -pi / 2,
             -pi / 2, -pi / 2, -pi / 2, -pi / 2, -pi / 2, -pi / 2, -pi / 2, -pi / 2, ]
        ]
        ######################### controller stop #############
        ctl_index = ctl.ctl_index
        v = 0.01 if ctl_index < 10 else -0.01  # m/s
        state.leg_base_xy = state.leg_base_xy + [0, v * ctl.dt]

        # dtheta_leg11 = dtheta_leg11 + w * dt * ctl_leg1_sig[0][current_index_in_control_arr]
        # plot_leg11(dtheta_leg11)
        # dtheta_leg12 = dtheta_leg12 + w * dt * ctl_leg1_sig[1][current_index_in_control_arr]
        # plot_leg21(dtheta_leg12)

        # 画出膝盖和小腿
        # plot_knee()
        # dtheta_leg21 = ctl_leg2_sig[0][current_index_in_control_arr]
        # plot_leg12(dtheta_leg21)
        # dtheta_leg22 = ctl_leg2_sig[1][current_index_in_control_arr]
        # plot_leg22(dtheta_leg22)

        plt.text(info_xy[0], info_xy[1], f'重心高度 {state.leg_base_xy[1]:.3f}')
        # plt.text(info_xy[0], info_xy[1]-0.08, f'脚离地面高度 {state.leg_base_xy[1]:.3f}')
        plt.pause(ctl.dt)
        ctl.add_control_index()
    plt.show()


if __name__ == '__main__':
    job_print_robot_support()
