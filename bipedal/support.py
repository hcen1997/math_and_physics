# 通过展示单腿的支撑态, 来确定重心的移动过程
import math
import time

import matplotlib.pyplot as plt
from matplotlib import collections

import numpy as np
from numpy import pi, sin, cos, fmod

_dt = 0.05


class State:
    def __init__(self):
        self.height = 1.740
        self.velocity = 1.2
        self.single_step_time = 0.5
        self.x_p_step = self.velocity * self.single_step_time
        print("单步持续时间", self.single_step_time)
        print("单步运动距离", self.x_p_step)
        self.x_p_dt = self.x_p_step / self.single_step_time * _dt
        print("足部需要运动过的距离/每个dt", self.x_p_dt)

        self.head_r = 0.261 / 2
        self.head_x = 0.500
        self.body_h = 0.556
        self.body_w = 0.237
        self.leg_base_xy = np.array([self.head_x, self.height - 2 * self.head_r - self.body_h])
        self.leg_center_angle = pi
        self.leg_front_limit = pi / 6
        self.leg_back_limit = -pi / 12
        # 大腿长
        self.leg1_length = 0.384
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

        self.vh = 0
        self.a_mass = 0


class Control:
    def __init__(self):
        self.now = 0.0
        self.dt = _dt
        self.cnt = 0
        self.control_loop_total = 20
        self.ctl_index = self.cnt % self.control_loop_total

        self.w_leg1 = (state.leg_front_limit - state.leg_back_limit) / state.single_step_time
        # 大腿偏离y轴负半轴的角度, 向左为正
        self.theta_leg11 = 0
        # 小腿偏离大腿向小腿延长线的角度, 向左为正(都是负的)
        self.theta_leg12 = 0
        # 脚偏离小腿左向垂线的角度向上为正
        self.theta_leg13 = 0
        self.theta_leg21 = 0
        self.theta_leg22 = 0
        self.theta_leg23 = 0
        # 质心z轴位置
        self.z_pos = 0.0
        # 初始脚位置 相对于center线的ankle点
        self.foot1_xy = np.array([state.x_p_step / 2, 0])
        self.foot2_xy = np.array([0, 0])

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
    state.leg11_theta = state.leg_center_angle + dtheta
    leg_xy_00 = np.array([sin(state.leg11_theta) * state.leg1_length, cos(state.leg11_theta) * state.leg1_length])
    plot_line([state.leg_base_xy, leg_xy_00 + state.leg_base_xy], 'orange')
    state.leg1_knee_xy = leg_xy_00 + state.leg_base_xy


def plot_line(line, color):
    leg11 = collections.LineCollection([line], colors=color)
    plt.gcf().gca().add_collection(leg11)


def plot_leg21(dtheta):
    state.leg21_theta = state.leg_center_angle + dtheta
    leg_xy_00 = np.array([sin(state.leg21_theta) * state.leg1_length, cos(state.leg21_theta) * state.leg1_length])
    plot_line([state.leg_base_xy, leg_xy_00 + state.leg_base_xy], 'cyan')
    state.leg2_knee_xy = leg_xy_00 + state.leg_base_xy


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
    state.leg12_theta = state.leg11_theta + dtheta
    leg_xy_00 = np.array([sin(state.leg12_theta) * state.leg2_length, cos(state.leg12_theta) * state.leg2_length])
    state.leg1_ankle_xy = leg_xy_00 + state.leg1_knee_xy
    plot_line([state.leg1_knee_xy, state.leg1_ankle_xy], 'orange')

    t = (state.leg1_ankle_xy - state.leg1_knee_xy) ** 2
    # print("leg2长度", math.sqrt(t[0] + t[1])) # leg2 长度没问题, 只是看起来有变化


def plot_leg22(dtheta):
    state.leg22_theta = state.leg21_theta + dtheta
    leg_xy_00 = np.array([sin(state.leg22_theta) * state.leg2_length, cos(state.leg22_theta) * state.leg2_length])
    state.leg2_ankle_xy = leg_xy_00 + state.leg2_knee_xy
    plot_line([state.leg2_knee_xy, state.leg2_ankle_xy], 'cyan')


# log_
def plot_knee(n):
    if n == 1:
        knee1 = plt.Circle(state.leg1_knee_xy, 0.040, linestyle='-', fill=False, color='green')
        plt.gcf().gca().add_patch(knee1)
    elif n == 2:
        knee2 = plt.Circle(state.leg2_knee_xy, 0.040, linestyle='-', fill=False, color='green')
        plt.gcf().gca().add_patch(knee2)


def plot_leg13(dtheta):
    theta = state.leg12_theta + pi / 2 + dtheta  # + state.leg_back_limit + dtheta
    leg_xy_front = np.array(
        [sin(theta) * (state.leg3_length - state.leg3_back_length),
         cos(theta) * (state.leg3_length - state.leg3_back_length)])
    theta = theta - pi
    leg_xy_back = np.array([sin(theta) * (state.leg3_back_length), cos(theta) * (state.leg3_back_length)])
    plot_line([leg_xy_front + state.leg1_ankle_xy, leg_xy_back + state.leg1_ankle_xy], 'orange')  # 咦, 我只需要关心某些点就行了
    state.leg13_theta = theta + pi
    # print("指尖距地面高度", (leg_xy_front + leg_xy_o)[1])


def plot_ankle(n):
    if n == 1:
        ankle = plt.Circle(state.leg1_ankle_xy, state.leg_ankle_R / 2, linestyle='-', fill=False, color='green')
        plt.gcf().gca().add_patch(ankle)
    elif n == 2:
        ankle = plt.Circle(state.leg2_ankle_xy, state.leg_ankle_R / 2, linestyle='-', fill=False, color='green')
        plt.gcf().gca().add_patch(ankle)


def plot_leg23(dtheta):
    theta = state.leg22_theta + pi / 2 + dtheta  # + state.leg_back_limit + dtheta
    leg_xy_front = np.array(
        [sin(theta) * (state.leg3_length - state.leg3_back_length),
         cos(theta) * (state.leg3_length - state.leg3_back_length)])
    theta = theta - pi
    leg_xy_back = np.array([sin(theta) * (state.leg3_back_length), cos(theta) * (state.leg3_back_length)])
    plot_line([leg_xy_front + state.leg2_ankle_xy, leg_xy_back + state.leg2_ankle_xy], 'cyan')  # 咦, 我只需要关心某些点就行了
    state.leg23_theta = theta + pi


def plot_point(xy):
    plt.gcf().gca().add_patch(plt.Circle(xy, 0.010, linestyle='-', fill=False, color='black'))


def tri_lll(a, b, c):
    return np.arccos((b * b + c * c - a * a) / (2 * b * c))


def job_print_robot_support():
    plt.figure(figsize=(7, 6), )
    plt.axes([0.12, 0.11, 0.90 / 2, 0.80])
    plt.ion()
    perid = 50  # 秒
    print("\n开始脚脚运动模拟")
    print("大腿角速度rad/s", ctl.w_leg1)
    info_xy = [0.8, 1.6]
    st = time.time()
    last = time.time()
    for t in np.arange(0, perid, ctl.dt):
        st = time.time()
        plt.cla()
        plot_ref_line()
        plt.text(-0.05, 1.900, f'当前={t: 2.1f}秒')

        ################################### 运动开始
        ctl_index = ctl.ctl_index
        plot_body()
        # 以腿永远伸直来规划运动 得到一个角度
        ctl.theta_leg11 = np.arcsin(ctl.foot1_xy[0] / (state.leg1_length + state.leg2_length))
        plot_leg11(ctl.theta_leg11)
        plot_knee(1)
        plot_leg12(ctl.theta_leg12)
        plot_ankle(1)
        ctl.theta_leg13 = -ctl.theta_leg11
        plot_leg13(ctl.theta_leg13)

        plt.text(info_xy[0], info_xy[1], f'重心高度 {state.leg_base_xy[1]:.3f}')
        plt.text(info_xy[0], info_xy[1] - 0.08, f'踝关节离地距离 {state.leg1_ankle_xy[1]:.3f}')

        ######################################################### 开始处理第二条腿
        plot_leg21(ctl.theta_leg21)
        plot_knee(2)
        plot_leg22(ctl.theta_leg22)
        plot_ankle(2)
        plot_leg23(ctl.theta_leg23)

        ## 规划第二条腿路径
        # 1. 判断是否能达到, 不能到就取一个接近的值呀笨蛋
        # 2. 使用边边边公式给出答案
        ctl_foot_xy = np.array([  # 我也不知道这个路径选的怎么样, 但是先算吧
            [1] * 5 + [-1] * 10 + [1] * 5,
            [0.5] * 10 + [-0.5] * 10
        ]) * state.x_p_dt
        ctl.foot2_xy = ctl.foot2_xy + [ctl_foot_xy[0][ctl_index], ctl_foot_xy[1][ctl_index]]
        target_foot_xy = state.leg_base_xy - [0, state.leg1_length + state.leg2_length] + ctl.foot2_xy
        # plot_point(target_foot_xy)
        xy_from_base = target_foot_xy - state.leg_base_xy
        if state.leg1_length + state.leg2_length > np.sqrt(np.sum(xy_from_base * xy_from_base)):
            # target角 大腿偏target角 小腿偏大腿角
            theta1 = - np.arctan(xy_from_base[0] / abs(xy_from_base[1]))
            theta2 = tri_lll(state.leg2_length, state.leg1_length, np.sqrt(np.sum(xy_from_base * xy_from_base)))
            theta3 = tri_lll(np.sqrt(np.sum(xy_from_base * xy_from_base)), state.leg2_length, state.leg1_length)
            # print('t1', np.rad2deg(theta1))
            # print('t2', np.rad2deg(theta2))
            # print('t3', np.rad2deg(theta3))
            ctl.theta_leg21 = theta1+theta2
            ctl.theta_leg22 = theta3-pi

        ######################## 更新下一个时刻的重心和运动位置信息
        ctl_foot_xy = [
            [-1] * 10 + [1] * 10,
            [0] * 20
        ]
        # 下一个时刻的足坐标
        ctl.foot1_xy = ctl.foot1_xy + [state.x_p_dt * ctl_foot_xy[0][ctl_index], 0]
        # 为了保持脚在一个水平面, 所以把身体向上顶
        h_next = np.sqrt((state.leg1_length + state.leg2_length) ** 2 - ctl.foot1_xy[0] ** 2)
        dh = h_next - state.leg_base_xy[1]
        vh = dh / ctl.dt
        state.a_mass = (vh - state.vh) / ctl.dt
        state.vh = dh / ctl.dt
        # print(f"重心高度 {h_next} 重心速度 {vh} 重心加速度 {state.a_mass}")
        state.leg_base_xy = state.leg_base_xy + [0, vh * ctl.dt]

        last = time.time() - st
        print("不包含暂停的所有操作计算时间为(秒)", last)  # 31ms
        plt.pause(ctl.dt)
        ctl.add_control_index()
    plt.show()


if __name__ == '__main__':
    job_print_robot_support()
