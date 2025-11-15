import numpy as np

import constants
from cross_valuation import DATA_PATH
from preprocess_data import read_CSV
import matplotlib.pyplot as plt



def plot_bars_with_custom_color_ratios():
    values = ['a', 'u']
    value = 15
    ratios = [[14, 0, 1], [12, 2, 1]]


    # 创建柱状图
    plt.figure(figsize=(3, 6))
    x = range(len(values))
    width = 0.5

    # 设置颜色
    color_scheme = ['#FF5733', '#33FF57', '#5733FF']  # 三种颜色

    # 遍历 values 和 ratios
    bottom = 0
    for i, (v, r) in enumerate(zip(values, ratios)):
        # 计算每个部分的高度
        total_ratio = sum(r)
        heights = [value * ratio / total_ratio for ratio in r]

        # 绘制柱子
        for j, height in enumerate(heights):
            plt.bar(x[i], height, bottom=bottom, color=color_scheme[j], width=width)
            bottom += height

        bottom = 0  # 重置 bottom

    plt.xticks(x, values)
    plt.axis('off')  # 关闭坐标轴
    plt.show()
    1

def plot_bars_with_custom_color_ratios_1():
    """
    根据给定的 values 和对应的 ratios 绘制水平柱状图，柱子的颜色按比例分配。

    参数：
        values: 值列表
        value: 全部值的总和
        ratios: 颜色比例列表的列表
    """

    # 示例数据
    values = ['a', 'u']
    value = 15
    ratios = [[12, 2, 1], [10, 4, 1]]

    # 创建水平柱状图
    plt.figure(figsize=(10, 6))
    y = range(len(values))
    height = 0.2
    colors = ['#FF5733', '#33FF57', '#5733FF']  # 三种颜色

    # 遍历 values 和 ratios
    for i, (v, ratio) in enumerate(zip(values, ratios)):
        # 计算每个部分的宽度
        total_ratio = sum(ratio)
        widths = [value * r / total_ratio for r in ratio]

        # 绘制柱子
        left = 0

        y_adpat = [0]
        for j, width in enumerate(widths):
            plt.barh(y[i] - i * 0.78, width, left=left, color=colors[j], height=height)
            left += width
        y_adpat.append(y[i] - i * 0.78)

    #plt.yticks(y, values)
    plt.yticks(y_adpat, values)  # 可以在这里调整刻度的位置

    plt.xticks([1, 5, 10, 15])
    plt.xlabel('Values')
    plt.ylabel('Methods')
    plt.show()
    1

if __name__ == '__main__':
    plot_bars_with_custom_color_ratios_1()
    plot_bars_with_custom_color_ratios()

    for name in constants.DataSet_MERGE:
        path = DATA_PATH / 'original' / ('%s.csv' % (name))
        X, y = read_CSV(path)
        unique_labels, counts = np.unique(y, return_counts=True)
        unique_labels

#   '', 'winequality-red-8_vs_6', 'winequality-white-3_vs_7',
#                  'pen.global', 'ecoli-0_vs_1', 'ERA', 'wpbc', 'glass2'
# ecoli-0_vs_1 1.8
# Piam 1.9
# haberman 2.7
# wpbc 3.2
# ecoli 4.25
# new-thyroid1 5.1
# newthyroid2  5.1
# ERA 7.4
# pen.global 8.0
# yeast-2_vs_4 9.1
# led7digit-0-2-4-5-6-7-8-9_vs_1 10.9
# glass2 11.6
# yeast5 32.7
# winequality-red-8_vs_6 35.4
# winequality-white-3_vs_7 44
