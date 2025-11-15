
import numpy as np

import constants
from distances import euclidean_distance


# 定义计算类别中心的函数，formula (7)
def calculate_class_centers(data, labels):
    unique_labels = np.unique(labels)
    class_centers = []
    for label in unique_labels:
        class_samples = data[labels == label]
        class_center = np.sum(class_samples, axis=0) / len(class_samples)
        class_centers.append(class_center)
    return np.array(class_centers)


def calculate_membership_degree(data, class_centers, m=constants.M):
    '''

    定义计算隶属度的函数，每一个样本相对每一个类的membership degree
    :param data:
    :param class_centers:
    :param m:
    :return:
    '''
    n_samples = data.shape[0]
    n_classes = class_centers.shape[0]
    membership_matrix = np.zeros((n_samples, n_classes))
    # 计算第i个样本关于第j个类的membership degree，using formula (5)
    for i in range(n_samples): # 第i个样本
        for j in range(n_classes): # 第j个类
            numerator = euclidean_distance(data[i], class_centers[j])
            denominator = np.sum([(numerator / euclidean_distance(data[i], center))**(2 / (m - 1)) for center in class_centers])
            membership_matrix[i][j] = 1 / denominator
    return membership_matrix


def calculate_membership_mean_value(membership_matrix, labels):
    '''
    calculate average membership degree of the sample in class k belonging to class j
    类 k 相对于类 j 的average membership degree
    :param membership_matrix:
    :param labels:
    :return:
    '''
    unique_labels = np.unique(labels)
    n_samples, n_classes = membership_matrix.shape
    membership_mean_value_matrix = np.zeros((n_classes, n_classes))

    for k in range(n_classes):
        # 找到当前类别的样本的索引
        class_samples_indices = np.where(labels == unique_labels[k])[0]
        # 当前类别的样本数量
        n_class_samples = len(class_samples_indices)

        # 对于每个类别 k
        for j in range(n_classes):
            # 计算当前类别样本的隶属度之和
            membership_sum = 0
            for i in class_samples_indices:
                membership_sum += membership_matrix[i][j]
            # 计算平均隶属度并存储在平均隶属度矩阵中
            membership_mean_value_matrix[k][j] = membership_sum / n_class_samples

    return membership_mean_value_matrix

def divide_samples(membership_matrix, membership_mean_value_matrix, labels):
    '''

    :param membership_matrix:
    :param membership_mean_value_matrix:
    :param labels:
    :return: d_safe 安全样本, d_ovsafe overlapping region样本, d_maynoise 可能是 noise 的样本
    '''
    n_samples, n_classes = membership_matrix.shape
    unique_labels = np.unique(labels)
    d_safe, d_ovsafe, d_maynoise = [], [], []

    for j in range(n_classes):  # Iterate over classes
        class_samples_indices = np.where(labels == unique_labels[j])[0]  # Indices of samples in class j
        n_class_samples = len(class_samples_indices)  # Number of samples in class j

        for i in class_samples_indices:  # Iterate over samples in class j
            if membership_matrix[i][j] > membership_mean_value_matrix[j][j]:
                d_safe.append(i)
            else:
                all_greater = True
                for k in range(n_classes):
                    if k != j and membership_matrix[i][k] > membership_mean_value_matrix[k][k]:
                        all_greater = False
                        break
                if all_greater:
                    d_ovsafe.append(i) # the samples in the overlapping region
                else:
                    d_maynoise.append(i)

    return d_safe, d_ovsafe, d_maynoise

def remove_possible_noise(membership_matrix, membership_mean_value_matrix, labels ,d_maynoise):
    n_samples, n_classes = membership_matrix.shape
    unique_labels = np.unique(labels)
    d_noisy = []

    for j in range(n_classes):  # Iterate over classes
        class_samples_indices = [i for i in d_maynoise if
                                 labels[i] == unique_labels[j]]  # Indices of samples in class j
        n_class_samples = len(class_samples_indices)  # Number of samples in class j

        for i in class_samples_indices:  # Iterate over samples in class j
            all_greater = True
            for k in range(n_classes):
                if k != j and membership_matrix[i][k] <= membership_mean_value_matrix[k][k]:
                    all_greater = False
                    break
            if all_greater:
                d_noisy.append(i) # d_noisy 是需要删除的样本

    return d_noisy