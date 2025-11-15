import constants
import numpy as np

from distances import cal_nearst_neighbors
from dar.algorithm_dar import calculate_membership_mean_value, \
    remove_possible_noise
from improve.resampling_pro import fuzzy_membership, cal_membership_degree, resample_pro
from improve.resampling_pro1 import resample_pro1, resample_pro2
from improve.sample_divide_pro import divide_samples_pro
from preprocess_data import read_CSV, list_to_array

#resample(safe_samples, overlap_samples, noise_samples, outlier_samples, X, y)
def resample(safe_samples, overlap_samples, noise_samples, outlier_samples, X, labels):

    safe_samples, overlap_samples, noise_samples, outlier_samples, labels = list_to_array(safe_samples, overlap_samples, noise_samples, outlier_samples, labels)
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    n_overSafe_samples = [0] * len(unique_labels)
    #d_all_data = [d_safe, d_ovsafe, d_unnoise]

    max_samples_nums = 0
    for j in range(n_classes):
        num_overlap, num_noise = [], []
        if len(overlap_samples) != 0:
            num_overlap = np.where((labels[overlap_samples] == unique_labels[j]))[0]
        if len(noise_samples) != 0:
            num_noise = np.where((labels[noise_samples] == unique_labels[j]))[0]
        n_overSafe_samples[j] = len(num_overlap) + len(num_noise)
        max_samples_nums = max(max_samples_nums, n_overSafe_samples[j])

    Syns = [] # 合成的样本集合
    Syns_y = [] # 合成的样本集合 目标值y
    for j in range(n_classes):

        sys_nums = max_samples_nums - n_overSafe_samples[j]
        if(sys_nums == 0): continue

        indices_safe_j = safe_samples[np.where((labels[safe_samples] == unique_labels[j]))[0]]
        n_safe_j = len(indices_safe_j)

        # 计算d_safe 并 d_overlap 的集合
        indices_safe_overlap = np.concatenate((safe_samples, overlap_samples))
        samples_X = [X[index] for index in indices_safe_overlap]

        for i in indices_safe_j:
            distances_in_samples_X, indices_in_samples_X = cal_nearst_neighbors(X[i], samples_X, constants.K1)
            # 围绕该样本和成的样本数
            sys_num = int(sys_nums / n_safe_j)
            print(f"============================sys============================sys_num:{sys_num}")
            #sys_num = int(np.ceil(sys_nums / n_safe_j))
            for _ in range(0, sys_num):
                candidate_index = indices_in_samples_X[np.random.choice(len(indices_in_samples_X))]
                candidate_sample = samples_X[candidate_index]
                # Interpolation Oversampling
                synthetic_sample = X[i] + np.random.rand() * (X[i] - candidate_sample)
                Syns.append(synthetic_sample)
                Syns_y.append(unique_labels[j])

    return Syns, Syns_y


def DAR(data_path):
    '''
    Dividing data into safe, OvSafe and nosie, and Removing noise data
    :param data:
    :return:
    '''

    # 加载数据
    #data = "/Users/wyj/PycharmProjects/MC-MBRC-Pro/data/original/newthyroid.csv"
    X, y = read_CSV(data_path)

    ############################################################### 原来方法membership function 的表示方法 Begin
    # calculate membership matrix of dataset data
    # 计算类别中心
    #class_centers = calculate_class_centers(X, y)
    # 计算隶属度矩阵
    #membership_matrix = calculate_membership_degree(X, class_centers)
    ############################################################### 原来方法membership function 的表示方法 End

    ############################################################### 新方法membership function 的表示方法 Begin
    # 计算 within class fuzzy membership 和 between class fuzzy membership, 并返回fuzzy membership matrix
    fuzzy_membership_matrix = fuzzy_membership(X, y)
    # 计算 membership degree
    print("======================calculate membership degree for each sample======================")
    membership_matrix = cal_membership_degree(X, y, fuzzy_membership_matrix)
    ############################################################### 新方法membership function 的表示方法 Begin
    print("======================calculate membership mean degree for class======================")
    # 计算平均隶属度矩阵
    membership_mean_value_matrix = calculate_membership_mean_value(membership_matrix, y)

    ############################################################### 原来方法样本划分的表示方法 Begin
    # 样本划分
    #d_safe, d_ovsafe, d_maynoise = divide_samples(membership_matrix, membership_mean_value_matrix, y)
    # Remove noise d_noise是需要删除的样本
    #d_noise = remove_possible_noise(membership_matrix, membership_mean_value_matrix, y, d_maynoise)
    # remove noise 之后存在下来的元素
    #d_unnoise = [x for x in d_maynoise if x not in d_noise]
    ############################################################### 原来方法样本划分的表示方法 End

    ############################################################### 新方法样本划分的表示方法 Begin
    # 样本划分
    #d_safe, d_ovsafe, d_unnoise, d_noise = divide_samples_pro(membership_matrix, membership_mean_value_matrix, y)
    print("======================divide samples into four categories======================")
    safe_samples, overlap_samples, noise_samples, outlier_samples = \
        divide_samples_pro(membership_matrix, membership_mean_value_matrix, y)
    print(f"original outliers:{outlier_samples}")
    over_noisy = remove_possible_noise(membership_matrix, membership_mean_value_matrix, y, noise_samples)
    noise_samples = [x for x in noise_samples if x not in over_noisy]
    outlier_samples.extend(over_noisy)
    print(f"safe_samples:{len(safe_samples)},overlap_samples:{len(overlap_samples)},noise_samples:{len(noise_samples)},outlier_samples:{len(outlier_samples)}")

    ############################################################### 新方法样本划分的表示方法 End

    # 创建的合成样本
    #sythetics, sythetics_y = resample_pro(safe_samples, overlap_samples, noise_samples, outlier_samples, X, y, membership_matrix, membership_mean_value_matrix)
    print("======================sythetic new samples======================")
    #sythetics, sythetics_y = resample_pro1(safe_samples, overlap_samples, noise_samples, outlier_samples, X, y, membership_matrix)
    sythetics, sythetics_y = resample_pro2(safe_samples, overlap_samples, noise_samples, outlier_samples, X, y, membership_matrix)

    #sythetics, sythetics_y = resample(safe_samples, overlap_samples, noise_samples, outlier_samples, X, y)
    sythetics, sythetics_y = list_to_array(sythetics, sythetics_y)
    print(f"the number of sythetic samples:{len(sythetics)}")

    # 清理 majority 样本, 合成样本 sythetics
    #X, sythetics = run_cleaning(safe_samples, overlap_samples, noise_samples, outlier_samples, X, y, sythetics, sythetics_y)

    # 移除之前需要删除的outlier
    X = np.delete(X, outlier_samples, axis=0)
    y = np.delete(y, outlier_samples, axis=0)
    # 最终的数据集
    if(len(sythetics) != 0):
        X_final = np.concatenate((X, sythetics), axis=0)
        y_final = np.concatenate((y, sythetics_y), axis=0)
    else:
        X_final = X
        y_final = y

    #print(f"=======================final X=======================")
    #print(f"{X_final}")
    #print(f"=======================final y=======================")
    #print(f"{y_final}")

    return X_final, y_final