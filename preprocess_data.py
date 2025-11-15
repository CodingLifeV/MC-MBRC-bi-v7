import csv
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import constants

DATA_RESULT_PATH = Path(__file__).parent / 'data' / 'results'




def read_CSV(path):
    df = pd.read_csv(path)


    # 将非数值型列转换成数值型
    #df = pd.get_dummies(df)

    matrix = df.to_numpy()
    X, y = matrix[:, :-1], matrix[:, -1]

    #scaler = MinMaxScaler()
    #X_scaled = scaler.fit_transform(X)

    return X, y

def list_to_array(*args):
    arrays = [np.array(arg) for arg in args]
    return arrays


def convert_dtype(arr1, arr2):
    """
    Convert the dtype of arr1 to match the dtype of arr2.

    Parameters:
    arr1 (numpy.ndarray): The first array.
    arr2 (numpy.ndarray): The second array.

    Returns:
    numpy.ndarray: The first array with dtype matching arr2.
    """
    if arr1.dtype != arr2.dtype:
        arr1 = arr1.astype(arr2.dtype)
    return arr1


def calculate_and_write_avg(filename):
    # 读取CSV文件
    df = pd.read_csv(filename)

    # 计算每一列的和（不含第一列）
    sum_values = df.iloc[:, 1:].sum()
    # 计算行数（不包含第一行）
    num_rows = len(df)

    # 计算每一列的平均值
    avg_values = (sum_values / num_rows).round(2)

    # 在原文件中添加一行显示平均值
    avg_row = pd.DataFrame([['Avg'] + avg_values.tolist()], columns=['dataset'] + avg_values.index.tolist())
    df = pd.concat([df, avg_row], ignore_index=True)

    # 将结果写入原文件
    df.to_csv(filename, index=False)


if __name__ == '__main__':



    '''
    for classifer in constants.CLASSIFIERS_NAME:
        path = DATA_RESULT_PATH / constants.OUR_METHOD_NAME / ('%s_results.csv' % (classifer))
        calculate_and_write_avg(path)
    '''























