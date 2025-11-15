import numpy as np

import constants


def divide_samples_pro(membership_matrix, membership_mean_value_matrix, labels):

    n_samples, n_classes = membership_matrix.shape
    unique_labels = np.unique(labels)
    safe_samples, overlap_samples, noise_samples, outlier_samples = [], [], [], []

    for j in range(n_classes): # class jth
        class_samples_indices = np.where(labels == unique_labels[j])[0]
        for i in class_samples_indices:
            exist_greater = False
            for k in range(n_classes):
                if k != j and membership_matrix[i][k] > membership_mean_value_matrix[k][k]:
                    exist_greater = True
                    break

            if membership_matrix[i][j] > membership_mean_value_matrix[j][j]:
                if exist_greater:
                    overlap_samples.append(i)
                else:
                    safe_samples.append(i)
                #safe_samples.append(i)
            else:
                if exist_greater:
                    noise_samples.append(i)
                else:
                    #outlier_samples.append(i)
                    #overlap_samples.append(i)
                    is_outlier = overlap_or_outlier(i, j, membership_matrix, membership_mean_value_matrix)
                    if is_outlier:
                        outlier_samples.append(i)
                    else:
                        overlap_samples.append(i)

    return safe_samples, overlap_samples, noise_samples, outlier_samples



def overlap_or_outlier(sample_index, class_index, membership_matrix, membership_mean_value_matrix):
    '''
    判断指定样本是overlap样本还是outlier样本
    :param sample_index:
    :param class_index:
    :param membership_matrix:
    :param membership_mean_value_matrix:
    :return:
    '''
    n_samples, n_classes = membership_matrix.shape
    is_outlier = True
    for k in range(n_classes):
        if k != class_index and membership_matrix[sample_index][k] > membership_mean_value_matrix[k][k] / constants.COEF:
            is_outlier = False
            break

    return is_outlier
