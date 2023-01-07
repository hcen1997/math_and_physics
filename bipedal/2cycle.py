# 通过打印两个圆, 来展示步态

import matplotlib.pyplot as plt

# 基础数据 一个174cm高的两足机器人, 应该以 100步 52秒的速度前进
# 假设机器人的移动速度是1.2m/s
# 那么每一步的距离就是0.624米
h = 1740
sample1_step = 100
sample1_time = 52
v = 1.2
x_p_step = v * sample1_time / sample1_step

# 基础坐标轴  高为2米的线 (2000)
ref_h = 2000
# 参考线 x   长1米
ref_x = 1000
# 参考线位于-300处
ref_offset = -300


def plot_ref_line():
    plt.axes([0.12,0.11,0.90/2,0.88])
    plt.vlines(ref_offset, 0, ref_h, 'blue', ':', "垂直两米参考线")
    plt.hlines(ref_offset, 0, ref_x, 'black', ':', "水平1米参考线")
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


if __name__ == '__main__':
    plot_ref_line()
    plt.show()
