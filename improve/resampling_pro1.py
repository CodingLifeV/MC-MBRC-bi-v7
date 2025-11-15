import numpy as np

import constants
from distances import cal_nearst_neighbors, euclidean_distance
from preprocess_data import list_to_array

import numpy as np
from sklearn.neighbors import NearestNeighbors


# 假设 list_to_array, constants, 和 ide_clean_region 已经定义
# 假设您已经安装了 scikit-learn

def resample_pro2(safe_samples, overlap_samples, noise_samples, outlier_samples, X, labels,
                               membership_matrix):

    safe_samples, overlap_samples, noise_samples, outlier_samples, labels = list_to_array(safe_samples, overlap_samples,
                                                                                          noise_samples,
                                                                                          outlier_samples, labels)
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    n_overSafe_samples = [0] * len(unique_labels)

    # ... [计算 max_samples_nums 的代码, 与您原始版本相同] ...
    max_samples_nums = 0
    for j in range(n_classes):
        num_overlap, num_noise = [], []
        if len(overlap_samples) != 0:
            num_overlap = np.where((labels[overlap_samples] == unique_labels[j]))[0]
        if len(noise_samples) != 0:
            num_noise = np.where((labels[noise_samples] == unique_labels[j]))[0]
        n_overSafe_samples[j] = len(num_overlap) + len(num_noise)
        max_samples_nums = max(max_samples_nums, n_overSafe_samples[j])

    Syns = []  # 合成的样本集合
    Syns_y = []  # 合成的样本集合 目标值y

    # --- 优化点 1: 将 "majority" 样本的计算移出循环 ---
    # `samples_majority` 在所有类的循环中都是不变的
    indices_safe_overlap = np.concatenate((safe_samples, overlap_samples), axis=0)
    samples_majority = X[indices_safe_overlap]

    for j in range(n_classes):  # 类别循环
        sys_nums = max_samples_nums - n_overSafe_samples[j]
        if (sys_nums == 0): continue

        indices_safe_j = safe_samples[np.where((labels[safe_samples] == unique_labels[j]))[0]]
        n_safe_j = len(indices_safe_j)
        if n_safe_j == 0: continue

        samples_minority_j = X[indices_safe_j]  # 类 j 的所有 "safe" 样本

        # --- 优化点 2: 构建“全局”搜索空间 (每个类j仅一次) ---
        # 搜索空间 = (本类的所有safe样本) U (所有majority样本)
        # 注意：我们这里包含了 "self" 样本
        X_samples_j = np.concatenate((samples_minority_j, samples_majority), axis=0)

        # 记录 "majority" 样本在 X_samples_j 中的起始索引
        maj_indicator = len(samples_minority_j)

        # 如果搜索空间为空，跳过
        if len(X_samples_j) == 0:
            continue

        # [!!!] 警告：如果 X_samples_j 仍然非常大 (如 20万),
        # 且 q 值很小 (如 5)，这里可以优化为只搜索 k = 1000 个邻居。
        # 但为了忠于 discoverCMR 算法(它需要 *所有* 邻居), 我们设置 n_neighbors=len(...)
        # 您的算法要求必须找到 *q* 个 *连续* 的多数类。

        # 为了性能，我们设置一个合理的 k 上限，比如 2000
        # 假设 q 不会很大，且 "wall" 不会在 2000 个邻居之后
        k_search = min(len(X_samples_j), 2000)
        if constants.Q > k_search:
            k_search = min(len(X_samples_j), int(constants.Q * 1.5))  # 确保 k > q

        nn_tree = NearestNeighbors(n_neighbors=k_search, metric='euclidean', n_jobs=-1)
        nn_tree.fit(X_samples_j)

        # --- 优化点 3: 批量 K-NN 查询 ---
        # 一次性查询所有 "safe" 样本的 K-NN
        # all_indices 的 shape 是 (n_safe_j, k_search)
        all_distances, all_indices = nn_tree.kneighbors(samples_minority_j)

        # --- 优化点 4: 内联 discoverCMR ---
        # (替换 for i in indices_safe_j: ... discoverCMR(...) )
        for i_idx in range(n_safe_j):  # 样本循环 (现在非常快)

            current_sample = samples_minority_j[i_idx]

            # --- discoverCMR 的逻辑开始 ---
            # [关键]：跳过索引0 (它是样本自身), 完美模拟 `if i != index`
            indices_in_samples_X = all_indices[i_idx][1:]

            count = 0
            is_minority = []  # 记录第i个样本是minority还是majority
            t = 0  # 候选区域的截止索引 (默认为0)

            # 这个循环现在只迭代 k_search 次，而不是 N 次
            for i in range(len(indices_in_samples_X)):
                neighbor_global_index = indices_in_samples_X[i]

                if (neighbor_global_index >= maj_indicator):  # 邻居是 majority 样本
                    count = count + 1
                    is_minority.append(False)
                    if (count == constants.Q):
                        # [BUG修复] 原始 t = max(1, i-q+1)
                        # 在 i=1, q=2 时 t=1, 会包含一个样本 (错误)
                        # 正确的逻辑是 t = max(0, i-q+1)
                        t = max(0, i - constants.Q + 1)
                        break
                else:  # 邻居是 minority 样本
                    count = 0
                    is_minority.append(True)

                # 如果循环正常结束 (没有找到q个连续的)
                if i == len(indices_in_samples_X) - 1:
                    t = len(indices_in_samples_X)  # 候选区域是所有邻居

            # [关键]：我们只拿到了索引，现在才从 X_samples_j 中提取 *数据*
            candidate_region_indices = indices_in_samples_X[:t]
            candidate_region = X_samples_j[candidate_region_indices]

            is_minority = is_minority[:t]
            # --- discoverCMR 的逻辑结束 ---

            # --- ide_clean_region 的调用 (不变) ---
            clean_subregion = ide_clean_region(current_sample, candidate_region, is_minority)
            if (len(clean_subregion) == 0):
                continue

            sys_num = int(np.ceil(sys_nums / n_safe_j))
            for _ in range(0, sys_num):
                candidate_sample = clean_subregion[np.random.choice(len(clean_subregion))]
                # Interpolation Oversampling
                synthetic_sample = current_sample + np.random.rand() * (current_sample - candidate_sample)
                Syns.append(synthetic_sample)
                Syns_y.append(unique_labels[j])

    return Syns, Syns_y

def resample_pro1(safe_samples, overlap_samples, noise_samples, outlier_samples, X, labels, membership_matrix):

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

        # indices_safe_j 为 minority类
        indices_safe_j = safe_samples[np.where((labels[safe_samples] == unique_labels[j]))[0]]
        n_safe_j = len(indices_safe_j)

        # 计算d_safe 并 d_overlap 的集合 为majority 类
        indices_safe_overlap = np.concatenate((safe_samples, overlap_samples), axis=0)
        samples_majority = [X[index] for index in indices_safe_overlap]

        for i in indices_safe_j:
            samples_minority = [X[index] for index in indices_safe_j if i != index]

            candidate_region, is_minority = discoverCMR(X[i], samples_minority, samples_majority, q=constants.Q)
            clean_subregion = ide_clean_region(X[i], candidate_region, is_minority)
            if (len(clean_subregion) == 0):
                continue

            sys_num = int(np.ceil(sys_nums / n_safe_j))
            for _ in range(0, sys_num):

                candidate_sample = clean_subregion[np.random.choice(len(clean_subregion))]
                # Interpolation Oversampling
                synthetic_sample = X[i] + np.random.rand() * (X[i] - candidate_sample)
                Syns.append(synthetic_sample)
                Syns_y.append(unique_labels[j])

    return Syns, Syns_y

def discoverCMR(sample, minority_samples, majority_samples, q):
    '''
    参考文章:Oversampling With Reliably Expanding Minority Class Regions for Imbalanced Data Learning
    :param sample:
    :param minority_samples:
    :param majority_samples:
    :param q:
    :return:
    '''
    if(len(minority_samples) == 0):
        X_samples = majority_samples.copy()
    else:
        X_samples = np.concatenate((minority_samples, majority_samples), axis=0)

    maj_indicator = len(minority_samples) # X_samples 索引从maj_indicator开始之后的都为majority样本

    distances_in_samples_X, indices_in_samples_X = cal_nearst_neighbors(sample, X_samples, len(X_samples))

    count = 0
    is_minority = [] # 记录第i个样本是minority还是majority
    for i in range(len(indices_in_samples_X)):
        if(indices_in_samples_X[i] >= maj_indicator):  # 位置i处为主要类majority样本
            count = count + 1
            is_minority.append(False)
            if(count == q):
                t = max(1, i - q + 1)
                break
        else: # 位置i处为主要类minority样本
            count = 0
            is_minority.append(True)

    candidate_region = [X_samples[indices_in_samples_X[i]] for i in range(t)]
    is_minority = [is_minority[i] for i in range(t)]

    return candidate_region, is_minority


def ide_clean_region(sample, candidate_region, is_minority):
    clean_subregion = []
    for i in range(len(candidate_region)):
        circle_sam = (sample + candidate_region[i]) / 2
        radiu = euclidean_distance(sample, candidate_region[i]) / 2
        flag_clean = 1
        for l in range(len(candidate_region)):
            if(l != i and is_minority[l] == False and euclidean_distance(circle_sam, candidate_region[l]) <= radiu):
                flag_clean = 0
                break
        if flag_clean:
            clean_subregion.append(candidate_region[i])
    return clean_subregion

def ide_clean_region01(sample, candidate_region, X, current_j, labels):
    '''

    :param sample: 当前样本
    :param candidate_region: 当前样本候选样本集合CAR
    :param current_j: 当前类
    :param labels:  所有样本集标签
    :return:
    '''
    clean_subregion = []
    for i in range(len(candidate_region)):
        circle_sam = (sample + X[candidate_region[i]]) / 2
        radiu = euclidean_distance(sample, X[candidate_region[i]]) / 2
        flag_clean = 1
        for l in range(len(candidate_region)):
            if (l != i and euclidean_distance(circle_sam, X[candidate_region[l]]) < radiu and labels[candidate_region[l]] != current_j):
                flag_clean = 0 # 候选样本集当前样本需要排除
                break
        if flag_clean:
            clean_subregion.append(candidate_region[i])
    return clean_subregion