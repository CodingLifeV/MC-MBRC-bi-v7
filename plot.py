import os
from itertools import product

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import constants
from cross_valuation import DATA_PATH
from preprocess_data import DATA_RESULT_PATH


def get_comparison_results(path):
    data = pd.read_csv(path)
    # 去掉最后一行（平均排名）
    data = data[:-1]

    # 提取数据集名称和方法名称
    datasets = data['dataset'].tolist()
    methods = data.columns[1:].tolist()

    # 去掉自己的方法
    methods.remove(constants.OUR_METHOD_NAME)

    # 创建一个空的DataFrame来存储比较结果
    comparison_results = pd.DataFrame(index=methods, columns=['Better', 'Equal', 'Worse'])

    for method in methods:
        better_count = 0
        equal_count = 0
        worse_count = 0
        for dataset in datasets:
            target_value = data.loc[data['dataset'] == dataset, constants.OUR_METHOD_NAME].values[0]
            method_value = data.loc[data['dataset'] == dataset, method].values[0]
            if method_value < target_value:
                better_count += 1
            elif method_value == target_value:
                equal_count += 1
            else:
                worse_count += 1

        # 确保方法名没有单引号
        method = method.replace("'", "")

        comparison_results.loc[method] = [better_count, equal_count, worse_count]

    return comparison_results, methods

def plot_comparison_single(path, metric, classifer):
    """
        读取CSV文件并生成比较图, 对单一的classifer和单一的metric绘制一张图
        :param csv_file: CSV文件路径
        :param target_method: 要比较的目标方法名称
        """

    comparison_results, methods = get_comparison_results(path)

    # 绘制条形图
    plt.figure(figsize=(10, 6))
    y = range(len(comparison_results))
    height = 0.15

    #colors = ['#386BA3', '#B03C2B', '#6EA778']
    colors = ['#F8D588', '#7DFB6F', '#5235F5']


    y_adpat = []
    for i, (method, result) in enumerate(comparison_results.iterrows()):
        total_counts = result.values
        # 计算每个部分的高度
        total_ratio = sum(total_counts)
        widths = [total_ratio * ratio / total_ratio for ratio in total_counts]

        # 绘制柱子
        left = 0
        uuu = 0

        for j, width in enumerate(widths):
            uuu = y[i] - i * 0.83
            plt.barh(uuu, width, left=left, color=colors[j], height=height)
            left += width
        y_adpat.append(uuu)

    # plt.yticks(y, methods)
    plt.yticks(y_adpat, methods, fontsize=22)  # 可以在这里调整刻度的位置
    plt.xticks([0, 5, 10, 15], fontsize=22)
    plt.title(f"[{metric} for {classifer}] {constants.OUR_METHOD_NAME} vs. ")  # 将 'Values' 放置在图的上方，设置字体大小为22

    # plt.xlabel('Values', fontsize=22)

    # 调整图形边界
    plt.subplots_adjust(left=0.33, right=0.98, top=0.9, bottom=0.1)

    # plt.ylabel('Methods')
    plt.show()


def plot_comparison_all(path, metric, classifer, ax, col_index):
    """
    读取CSV文件并生成比较图,绘制一张大图，包含len(classifer) * len(metric) 个小图
    :param csv_file: CSV文件路径
    :param target_method: 要比较的目标方法名称
    """

    comparison_results, methods = get_comparison_results(path)

    # 绘制条形图
    #plt.figure(figsize=(10,6))
    y = range(len(comparison_results))
    height = 0.15
    #colors = ['#FFD37C', '#33FF57', '#5733FF']
    #colors = ['#386BA3', '#B03C2B', '#6EA778']
    colors = ['#F8D588', '#7DFB6F', '#5235F5']


    y_adpat = []
    for i, (method, result) in enumerate(comparison_results.iterrows()):
        total_counts = result.values
        # 计算每个部分的高度
        total_ratio = sum(total_counts)
        widths = [total_ratio * ratio / total_ratio for ratio in total_counts]

        # 绘制柱子
        left = 0
        uuu = 0

        for j, width in enumerate(widths):
            uuu = y[i] - i * 0.83
            ax.barh(uuu, width, left=left, color=colors[j], height=height)
            left += width
        y_adpat.append(uuu)

    # 设置y轴刻度位置和标签
    ax.set_yticks(y_adpat)

    # 如果不是第一列，则隐藏y轴刻度
    if col_index != 0:
        ax.set_yticklabels([])
    else:
        labels = []
        for m in methods:
            if m == 'SWIMRBF':
                labels.append(r'SWIM$_{RBF}$')
            elif m == 'A_SUWO':
                labels.append(r'A-SUWO')
            elif m == 'kmeans_SMOTE':
                labels.append(r'Kmeans SMOTE')
            elif m == 'Borderline_SMOTE2':
                labels.append(r'Borderline SMOTE2')
            else:
                labels.append(m)
        ax.set_yticklabels(labels, fontsize=30)
    #ax.set_yticklabels(methods, fontsize=25)  # 可以在这里调整刻度的位置
    ax.set_xticks([0, 5, 10, 15])
    ax.set_xticklabels(ax.get_xticks(), fontsize=30)

    if(classifer == 'CART'):
        classifer = 'DT'
    ax.title.set_text(
        f"[{metric} for {classifer}] MB-OCRE vs.")
        #f"[{metric} for {classifer}] {constants.OUR_METHOD_NAME} vs.")  # 将 'Values' 放置在图的上方，设置字体大小为22
    ax.title.set_fontsize(30)  # 设置标题字体大小为22

def plot_comparison_grid():
    # 创建大图
    fig, axs = plt.subplots(len(constants.CLASSIFIERS_NAME), len(constants.METRICS), figsize=(30, 20))
    fig.subplots_adjust(wspace=0.4, hspace=0.8)  # 调整这里的值来设置子图之间的距离


    for i, (classifer, metric) in enumerate(product(constants.CLASSIFIERS_NAME, constants.METRICS)):
        # 计算当前小图在大图中的行列索引
        row_index = i // len(constants.CLASSIFIERS_NAME)
        col_index = i % len(constants.METRICS)

        # 绘制当前分类器和度量值对应的比较图，并将其放置在大图的相应位置
        path = DATA_RESULT_PATH / 'merge' / ('%s.%s.csv' % (classifer, metric))
        plot_comparison_all(path, metric, classifer, axs[row_index, col_index], col_index)

        # 设置x轴刻度范围，使得最后一个刻度与边框重合
        axs[row_index, col_index].set_xlim(left=0, right=15)

        # 隐藏边框
        axs[row_index, col_index].spines['top'].set_visible(False)
        axs[row_index, col_index].spines['right'].set_visible(False)
        axs[row_index, col_index].spines['bottom'].set_visible(False)
        axs[row_index, col_index].spines['left'].set_visible(False)

    # 调整子图之间的间距
    plt.tight_layout()

    plt.savefig(f'better_equal_worse2.png', dpi=300)

    # 显示大图
    plt.show()
    1


if __name__ == '__main__':

    plot_comparison_grid()

    for classifer in constants.CLASSIFIERS_NAME:
        for metric in constants.METRICS:
            path = DATA_RESULT_PATH / 'merge' / ('%s.%s.csv' % (classifer, metric))
            plot_comparison_single(path, metric, classifer)






