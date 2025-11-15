import numpy as np

from scipy.spatial import distance_matrix

import constants
from distances import distance
from preprocess_data import list_to_array, convert_dtype

def clean(minority_points, majority_points, energy):
    minority_points, majority_points = list_to_array(minority_points, majority_points)

    #print(f"minority_points:{minority_points}")
    #print(f"++++++++++++++++++++++++++++++++++++++")
    #print(f"majority_points:{majority_points}")


    distances = distance_matrix(minority_points, majority_points, 1.0)
    radii = np.zeros(len(minority_points))
    translations = np.zeros(majority_points.shape)

    for i in range(len(minority_points)):
        minority_point = minority_points[i]
        remaining_energy = energy

        radius = 0.0
        sorted_distances = np.argsort(distances[i])  #
        n_majority_points_within_radius = 0

    while True:
        if n_majority_points_within_radius == len(majority_points):
            if n_majority_points_within_radius == 0:
                radius_change = remaining_energy / (n_majority_points_within_radius + 1)
            else:
                radius_change = remaining_energy / n_majority_points_within_radius

            radius += radius_change

            break

        radius_change = remaining_energy / (n_majority_points_within_radius + 1)

        if distances[i, sorted_distances[n_majority_points_within_radius]] >= radius + radius_change:
            radius += radius_change

            break
        else:
            if n_majority_points_within_radius == 0:
                last_distance = 0.0
            else:
                last_distance = distances[i, sorted_distances[n_majority_points_within_radius - 1]]

            radius_change = distances[i, sorted_distances[n_majority_points_within_radius]] - last_distance
            radius += radius_change
            remaining_energy -= radius_change * (n_majority_points_within_radius + 1)
            n_majority_points_within_radius += 1

    radii[i] = radius

    for j in range(n_majority_points_within_radius):
        majority_point = majority_points[sorted_distances[j]]
        d = distances[i, sorted_distances[j]]

        while d < 1e-20:
            majority_point += (1e-6 * np.random.rand(len(majority_point)) + 1e-6) * \
                              np.random.choice([-1.0, 1.0], len(majority_point))
            d = distance(minority_point, majority_point)

        translation = (radius - d) / d * (majority_point - minority_point)
        translations[sorted_distances[j]] += translation

    translations = convert_dtype(translations, majority_points)
    majority_points += translations

    return majority_points

def run_cleaning(d_safe, d_ovsafe, d_unnoise, d_noise, X, labels, syns, syn_labels):
    '''

    :param d_safe:
    :param d_ovsafe:
    :param d_unnoise:
    :param d_noise:
    :param X: 原样本集合,在清理过程中，不断更新
    :param labels: 原样本集合目标值
    :param syns: 合成样本
    :param syn_labels: 合成样本目标值
    :return:
    '''
    n_classes, counts = np.unique(labels, return_counts=True)
    sorted_indices = np.argsort(counts)

    for j in range(len(sorted_indices) - 1):

        # minority samples: 当前类的safe样本+noise样本+合成样本中属于类j的实例
        min_idx = []
        n_class = n_classes[sorted_indices[j]] # 当前最小类的目标值
        min_idx.extend(np.array(d_safe)[labels[d_safe] == n_class])
        #min_idx.extend(np.array(d_ovsafe)[labels[d_ovsafe] == n_class])
        min_idx.extend(np.array(d_unnoise)[labels[d_unnoise] == n_class])
        minority_samples = []
        for k in range(len(min_idx)):
            minority_samples.append(X[min_idx[k]])
        syn_indices = np.where(syn_labels == n_class)[0] # 合成实例中属于类j的实例
        for k in syn_indices:
            minority_samples.append(syns[k])

        # 当前少数类为空，不需要执行清理过程
        if(len(minority_samples) == 0):
            continue

        # majority samples: 大于类j实例数的类样本+大于类j实例数的类的合成实例样本
        maj_idx = []
        majority_samples = []
        for i in range(j+1, len(sorted_indices)):
            n_class = n_classes[sorted_indices[i]]
            class_samples_indices = np.where(labels == n_class)[0]
            maj_idx.extend(class_samples_indices)
        for k in range(len(maj_idx)):
            majority_samples.append(X[maj_idx[k]])
        # 拼接合成样本中作为majority类的样本
        syns_idx = []
        for i in range(j+1, len(sorted_indices)):
            n_class = n_classes[sorted_indices[i]]
            class_samples_indices = np.where(syn_labels == n_class)[0]
            syns_idx.extend(class_samples_indices)
            for k in class_samples_indices:
                majority_samples.append(syns[k])

        majority_samples = clean(minority_samples, majority_samples, constants.ENERGY)

        # 更新 X
        for idx, sample_idx in enumerate(maj_idx):
            X[sample_idx] = majority_samples[idx]
        # 更新 syns
        for idx, sample_idx in enumerate(syns_idx):
            indice = idx + len(maj_idx)
            syns[idx] = majority_samples[indice]

    return X, syns








