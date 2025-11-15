import numpy as np

import constants
from distances import cal_nearst_neighbors






def cal_samples_weight(safe_samples, overlap_samples, sys_nums, membership_matrix, j, labels):
    '''
    为当前类j的safe样本和overlap样本计算样本权重，用于指导实例数合成
    :param safe_samples:
    :param overlap_samples:
    :param sys_nums:
    :param membership_matrix:
    :param j:
    :param labels:
    :return:
    '''
    unique_labels, counts = np.unique(labels, return_counts=True)

    indices_safe_j = safe_samples[np.where((labels[safe_samples] == unique_labels[j]))[0]]
    #indices_overlap_j = overlap_samples[np.where((labels[overlap_samples] == unique_labels[j]))[0]]
    #indices = np.concatenate([indices_safe_j, indices_overlap_j])

    row_sums = np.sum(membership_matrix[indices_safe_j], axis=1)
    row_j = membership_matrix[indices_safe_j, j]
    membership_safe_overlap = row_sums - row_j

    inverse_sum = np.sum(1 / membership_safe_overlap)
    # 计算每个位置的权重
    weights = (1 / membership_safe_overlap) / inverse_sum
    # 计算每一个样本需要合成的样本数
    #nums = np.ceil(sys_nums * weights)
    nums = sys_nums * weights
    print(f"{nums}")

    return nums


def create_samples_around_safe(safe_samples, overlap_samples, j, X, nums, labels):
    '''
    环绕safe样本合成新的样本
    :param safe_samples:
    :param overlap_samples:
    :param j: 类 j
    :param X:
    :param nums: 需要合成的实例数组
    :param labels:
    :return:
    '''

    Syns, Syns_y = [], []
    unique_labels, counts = np.unique(labels, return_counts=True)

    # 为safe样本周边生成样本
    # 计算safe和overlap的集合
    indices_safe_j = safe_samples[np.where((labels[safe_samples] == unique_labels[j]))[0]]
    indices_overlap_j = overlap_samples[np.where((labels[overlap_samples] == unique_labels[j]))[0]]
    indices_safe_overlap = np.concatenate((safe_samples, overlap_samples))
    samples_X = [X[index] for index in indices_safe_overlap]

    indices = np.concatenate([indices_safe_j, indices_overlap_j])

    for i in indices_safe_j:
        distances_in_samples_X, indices_in_samples_X = cal_nearst_neighbors(X[i], samples_X, constants.K1)

        num = int(nums[np.where(indices == i)[0][0]])

        #idx = np.where(indices == i)[0]
        #num_sys_idx = int(idx[0])
        #xxx = int(nums[num_sys_idx])

        for _ in range(0, num):
            candidate_index = indices_in_samples_X[np.random.choice(len(indices_in_samples_X))]
            candidate_sample = samples_X[candidate_index]
            # Interpolation Oversampling
            synthetic_sample = X[i] + np.random.rand() * (X[i] - candidate_sample)
            Syns.append(synthetic_sample)
            Syns_y.append(unique_labels[j])


    return Syns, Syns_y



def create_samples_around_overlap(safe_samples, overlap_samples, X, j, nums, labels, membership_matrix, membership_mean_value_matrix):
    '''
    环绕overlap样本合成新的样本
    :param safe_samples:
    :param overlap_samples:
    :param X: 样本集
    :param j: 类j
    :param nums: 需要合成的样本数
    :param labels:
    :param membership_matrix:
    :param membership_mean_value_matrix:
    :return:
    '''
    unique_labels, counts = np.unique(labels, return_counts=True)
    n_classes = len(unique_labels)

    indices_safe_j = safe_samples[np.where((labels[safe_samples] == unique_labels[j]))[0]]
    indices_overlap_j = overlap_samples[np.where((labels[overlap_samples] == unique_labels[j]))[0]]
    indices = np.concatenate([indices_safe_j, indices_overlap_j])

    Syns, Syns_y = [], []
    # 为overlap样本周边生成样本
    for i in indices_overlap_j:
        samples_X = []
        # print(f"++++++++++++++++++++++++++++++++++++++++++实例:{i}")

        # 拿到membership_matrix[i][k] > membership_mean_value_matrix[k][k]的所有k值 k 为不等于j的类
        greater_k = []
        for k in range(n_classes):
            if k != j and membership_matrix[i][k] > membership_mean_value_matrix[k][k]:
                # greater_k.append(unique_labels[k])
                idx = np.where(unique_labels[k] == labels)[0]
                for yyy in idx:
                    samples_X.append(X[yyy])

        if (len(samples_X) == 0): continue

        distances_in_samples_X, indices_in_samples_X = cal_nearst_neighbors(X[i], samples_X, constants.K3)
        # 拿到最近邻居indices_in_samples_X中属于greater_k类的所有样本,作为候选样本CAR candidate sample set

        # 生成样本
        #idx = np.where(indices == i)[0]
        #num_sys_idx = int(idx[0])
        #xxx = int(nums[num_sys_idx])

        num = int(nums[np.where(indices == i)[0][0]])

        for _ in range(0, num):
            candidate_index = indices_in_samples_X[np.random.choice(len(indices_in_samples_X))]
            candidate_sample = samples_X[candidate_index]
            # Interpolation Oversampling
            synthetic_sample = X[i] + np.random.rand() * (X[i] - candidate_sample)
            Syns.append(synthetic_sample)
            Syns_y.append(unique_labels[j])

    return Syns, Syns_y