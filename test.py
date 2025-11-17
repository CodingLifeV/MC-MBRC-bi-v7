import numpy as np
import pandas as pd

import constants
from cross_valuation import DATA_PATH, read_data, write_data
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


def testtt():
    cleanData()

    for name in constants.DATA_NAME:
        for partition in range(1, 2):
            for fold in range(1, 2):
                print(f"==read data====dataset name:{name},partition:{partition},fold:{fold}========")
                X_train, y_train = read_data(name, partition, fold, flag='TRAIN')
                print(f"==writedata====dataset name:{name},partition:{partition},fold:{fold}========")

                write_data(name, partition, fold, X_train, y_train, 'TRAIN_RES')

    1



def cleanData():
    import pandas as pd

    # 1. 读取文件
    # header=0 表示第一行是表头（包含 'target' 那一行）
    df = pd.read_csv("kddcup10percent.csv", header=0)

    # 2. 定义清洗函数
    def clean_label(val):
        # 将输入转为字符串并去除空格
        s_val = str(val).strip()

        # 如果是 0 或 '0'，保留为 0
        if s_val == '0' or s_val == '0.0':
            return 0
        # 如果是 1 或 '1'，保留为 1
        elif s_val == '1' or s_val == '1.0':
            return 1
        # 其他所有情况（包括攻击类型标签），全部替换为 1
        else:
            return 1

    # 3. 获取最后一列的列名
    last_col = df.columns[-1]

    # 4. 应用清洗函数
    df[last_col] = df[last_col].apply(clean_label)

    # 5. 强制转换为整数类型
    df[last_col] = df[last_col].astype(int)

    # 6. 保存文件
    output_file = "kddcup10percent_processed.csv"
    df.to_csv(output_file, index=False)

    print(f"✅ 处理完成！文件已保存为: {output_file}")
    print(f"最后一列的唯一值: {df[last_col].unique()}")

if __name__ == '__main__':

    testtt()

    '''
    plot_bars_with_custom_color_ratios_1()
    plot_bars_with_custom_color_ratios()

    for name in constants.DataSet_MERGE:
        path = DATA_PATH / 'original' / ('%s.csv' % (name))
        X, y = read_CSV(path)
        unique_labels, counts = np.unique(y, return_counts=True)
        unique_labels
    '''
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
