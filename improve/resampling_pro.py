import numpy as np

import constants
from distances import cal_nearst_neighbors
from improve.create_samples import create_samples_around_safe, cal_samples_weight, create_samples_around_overlap
from preprocess_data import read_CSV, list_to_array
import numpy as np
from sklearn.neighbors import NearestNeighbors


def DAR_PRO(data_path):
    # 加载数据
    X, y = read_CSV(data_path)

    # 计算每一个实例最近的第k个邻居的距离，k个邻居是自身所处的类
    # 计算 within class fuzzy membership 和 between class fuzzy membership, 并返回fuzzy membership matrix
    fuzzy_membership_matrix = fuzzy_membership(X,y)
    # 计算 membership degree
    membership_matrix = cal_membership_degree(X, y, fuzzy_membership_matrix)


def cal_membership_degree(X, y, fuzzy_membership_matrix, m=constants.M):

    unique_labels, counts = np.unique(y, return_counts=True)
    n_classes = len(unique_labels)
    num_instances, _ = X.shape

    membership_matrix = np.zeros((num_instances, n_classes))

    for i in range(num_instances): # 第i个样本
        for j in range(n_classes): # 第j个类
            for element in fuzzy_membership_matrix[i]:
                if(element == 0): continue
                membership_matrix[i][j] += (fuzzy_membership_matrix[i][j] / element) ** (1 / (m - 1))


    '''
    for j in range(n_classes):
        class_samples_indices = np.where(y == unique_labels[j])[0]
        for i in class_samples_indices:
            for k in range(n_classes):
                membership_matrix[i][j] += (fuzzy_membership_matrix[i][j] / fuzzy_membership_matrix[i][k]) ** (1 / (m - 1))
            #membership_matrix[i][j] = 1 / membership_matrix[i][j]
    '''
    return membership_matrix




# 假设 constants.K2 已经定义，例如：
# class constants:
#    K2 = 1.0

def kth_nearest_neigh_distance(X, y):
    '''
    计算样本 i 在自己所属类内的第 k 个最近邻居的距离 within_class_k_distances[i]
    以及所属类 k 的距离倒数之和 distances_sum_reciprocal[k]

    (使用 sklearn.neighbors.NearestNeighbors 进行了批量优化)

    :param X: 特征数据, shape (n_samples, n_features)
    :param y: 标签数据, shape (n_samples,)
    :param constants: 包含 K2 的常量对象
    :return: (within_class_k_distances, distances_sum_reciprocal)
    '''

    unique_labels, counts = np.unique(y, return_counts=True)
    n_classes = len(unique_labels)
    num_instances, _ = X.shape

    # 初始化结果数组
    within_class_k_distances = np.zeros(num_instances)
    distances_sum_reciprocal = np.zeros(n_classes)  # 距离倒数之和

    # 计算每一个目标值 最近邻 K 值
    # 注意：确保 k_vals 中的值至少为 1
    k_vals = np.ceil(np.sqrt(counts) * constants.K2).astype(int)
    k_vals = np.maximum(k_vals, 1)  # 保证 k 至少为 1

    for j in range(n_classes):
        # 1. 提取当前类的所有样本
        class_samples_indices = np.where(y == unique_labels[j])[0]
        samples_within = X[class_samples_indices]
        num_samples_in_class = samples_within.shape[0]

        # 获取当前类别的 k 值
        k_j = k_vals[j]

        # 2. 处理边缘情况：类中只有一个或没有样本
        if num_samples_in_class <= 1:
            within_class_k_distances[class_samples_indices] = 0.0  # 距离为0
            distances_sum_reciprocal[j] = 0.0  # 倒数和为0 (避免除零)
            continue  # 处理下一个类

        # 3. 确定需要查找的邻居数
        # 我们需要第 k_j 个邻居。
        # NearestNeighbors.kneighbors() 会包含样本自身 (距离为0) 作为第0个邻居。
        # 所以我们需要查找 k_j + 1 个邻居，才能拿到第 k_j 个邻居（索引为 k_j）。
        # 同时，请求的邻居数不能超过类中的总样本数。
        n_neighbors_to_find = min(k_j + 1, num_samples_in_class)

        # 4. 使用 NearestNeighbors 批量计算
        # n_jobs=-1 表示使用所有可用的 CPU 核心进行并行计算，极大提速
        nn = NearestNeighbors(n_neighbors=n_neighbors_to_find,
                              metric='euclidean',
                              algorithm='auto',
                              n_jobs=-1)
        nn.fit(samples_within)

        # 批量查询类中所有样本的 k+1 个最近邻
        # distances 的 shape 是 (num_samples_in_class, n_neighbors_to_find)
        distances, _ = nn.kneighbors(samples_within, n_neighbors=n_neighbors_to_find)

        # 5. 提取第 k 个邻居的距离
        # 原始代码的 k 值检查逻辑:
        # idx = int(k_vals[j]) - 1
        # if(idx >= len(nearest_distances)): idx = len(nearest_distances) - 1
        # 这意味着如果 k 大于 (类样本数-1)，则取最远的那个邻居。

        # 在我们的 new distances 数组中：
        # - 第 0 列 (distances[:, 0]) 是样本自身，距离为 0
        # - 第 1 列 (distances[:, 1]) 是第 1 个邻居
        # - 第 k_j 列 (distances[:, k_j]) 是第 k_j 个邻居

        # 我们要找的列索引是 k_j。但如果 k_j 太大 (大于最大索引 n_neighbors_to_find - 1)，
        # 我们就取最后一列，即最远的邻居。
        target_col_idx = min(k_j, n_neighbors_to_find - 1)

        # 提取所有样本的第 k 个邻居距离
        k_th_distances = distances[:, target_col_idx]

        # 6. 赋值结果
        # [核心优化]：使用索引进行矢量化赋值，而不是循环
        within_class_k_distances[class_samples_indices] = k_th_distances

        # 7. 计算距离倒数之和
        # [鲁棒性]：必须处理距离为 0 的情况（例如样本点重合），避免除零错误
        non_zero_mask = k_th_distances > 0
        if np.any(non_zero_mask):
            distances_sum_reciprocal[j] = np.sum(1.0 / k_th_distances[non_zero_mask])
        else:
            distances_sum_reciprocal[j] = 0.0  # 所有距离都是0

    return within_class_k_distances, distances_sum_reciprocal
'''
def kth_nearest_neigh_distance(X, y):
    
    #计算样本 i 在自己所属类内的第 k 个最近邻居的距离within_class_k_distances[i] 以及所属类k的距离倒数之和distances_sum_reciprocal[k]
    #:param X:
    #:param y:
    #:return:
    

    unique_labels, counts = np.unique(y, return_counts=True)
    n_classes = len(unique_labels)
    num_instances, _ = X.shape
    within_class_k_distances = np.zeros(num_instances)
    distances_sum_reciprocal = np.zeros(n_classes)  # 距离之和

    # 计算每一个目标值 最近邻 K 值
    k_vals = np.ceil(np.sqrt(counts) * constants.K2)

    for j in range(n_classes):
        class_samples_indices = np.where(y == unique_labels[j])[0]
        samples_within = X[class_samples_indices]
        for i in class_samples_indices:
            sam = X[i]
            nearest_distances, _ = cal_nearst_neighbors(sam, samples_within, int(k_vals[j]))
            # 求最近邻居平均值，存储在最后一个元素
            #nearest_distances[int(k_vals[j]) - 1] = sum(nearest_distances) / int(k_vals[j])

            #if(int(k_vals[j]) - 1 >= len(nearest_distances)):
            #    k_vals[j] = len(nearest_distances)

            # 没有最近邻居,说明当前类只有一个实例
            #if(len(nearest_distances) == 0):

            # 最近邻居只有1个的情况
            idx = int(k_vals[j]) - 1
            if(idx >= len(nearest_distances)):
                idx = len(nearest_distances) - 1

            if(len(nearest_distances) == 0):    # 没有最近邻居的情况,说明当前类只有一个实例
                within_class_k_distances[i] = 0
                #distances_sum_reciprocal[j] = 1
            else:
                within_class_k_distances[i] = nearest_distances[idx]
                distances_sum_reciprocal[j] += 1 / within_class_k_distances[i]

    return within_class_k_distances, distances_sum_reciprocal
'''

import numpy as np
from sklearn.neighbors import NearestNeighbors


# -----------------------------------------------------------------
# 假设您已经有了上一问的优化函数，这里需要用到它
# 并且假设 constants.K2 已经定义
#
# def kth_nearest_neigh_distance_optimized(X, y, constants):
#     ... (上一问的优化代码)
# -----------------------------------------------------------------


def fuzzy_membership(X, y):
    '''
    计算样本i相对于自己类 within class 的 fuzzy membership 和
    相对于其他类 between class 的 fuzzy membership

    (使用 sklearn.neighbors.NearestNeighbors 和 NumPy 矢量化进行了全面优化)

    :param X: 特征数据, shape (n_samples, n_features)
    :param y: 标签数据, shape (n_samples,)
    :param constants: 包含 K2 的常量对象
    :param kth_nearest_neigh_distance_optimized: 上一问中优化后的k-NN距离函数
    :return: fuzzy_membership_matrix
    '''

    # 1. 基础设置
    unique_labels, counts = np.unique(y, return_counts=True)
    n_classes = len(unique_labels)
    num_instances, _ = X.shape

    # 计算每一个目标值 最近邻 K 值
    k_vals = np.ceil(np.sqrt(counts) * constants.K2).astype(int)
    k_vals = np.maximum(k_vals, 1)  # 保证 k 至少为 1

    # 2. [关键] 调用优化后的 k-NN 函数
    # 这已为我们准备好了所有 within-class 计算所需的数据
    within_class_k_distances, distances_sum_reciprocal = kth_nearest_neigh_distance(X, y)

    # 3. 初始化结果矩阵
    fuzzy_membership_matrix = np.zeros((num_instances, n_classes))

    # 4. 主循环：遍历每一个类 j (作为 "自己" 的类)
    for j in range(n_classes):
        class_j_indices = np.where(y == unique_labels[j])[0]
        class_j_samples = X[class_j_indices]
        n_samples_j = len(class_j_indices)

        if n_samples_j == 0:
            continue  # 如果这个类没有样本，跳过

        # -----------------------------------------------------------------
        # 优化点 1: 矢量化 "Within-Class Membership" (替换第一个 for-i 循环)
        # -----------------------------------------------------------------

        # 获取类 j 中所有样本的 k-NN 距离
        within_distances_j = within_class_k_distances[class_j_indices]
        dist_sum_recip_j = distances_sum_reciprocal[j]

        if dist_sum_recip_j != 0:
            # 使用 NumPy 矢量化批量计算
            membership_j = (1.0 / within_distances_j) / dist_sum_recip_j

            # 处理原始逻辑：within_class_k_distances[i] == 0 时，隶属度为 1
            # (这通常发生在类中只有一个实例时)
            zero_dist_mask = (within_distances_j == 0)
            membership_j[zero_dist_mask] = 1.0
        else:
            # 如果 sum_reciprocal 为 0, 说明所有 k-NN 距离都为 0 (例如类中只有一个实例)
            # 遵循原始逻辑，将隶属度设为 1
            membership_j = np.ones(n_samples_j)

        # 矢量化赋值
        fuzzy_membership_matrix[class_j_indices, j] = membership_j

        # -----------------------------------------------------------------
        # 优化点 2: 批量化 "Between-Class Membership" (替换第二个 for-i 循环)
        # -----------------------------------------------------------------

        # 遍历所有其他类 k
        for k in range(n_classes):
            if k == j:
                continue

            class_k_indices = np.where(y == unique_labels[k])[0]
            class_k_samples = X[class_k_indices]
            n_samples_k = len(class_k_indices)
            k_k = k_vals[k]  # 获取类 k 对应的 k 值
            dist_sum_recip_k = distances_sum_reciprocal[k]

            # 边缘情况处理：如果类 k 没有样本，或其倒数和为0
            if n_samples_k == 0 or dist_sum_recip_k == 0:
                # 遵循原始逻辑，隶属度设为 0
                fuzzy_membership_matrix[class_j_indices, k] = 0.0
                continue

            # 确定需要查找的邻居数
            # 原始逻辑是取第 k_k-1 个索引，如果 k_k 太大，则取最后一个
            # 我们通过 min(k_k, n_samples_k) 来实现这一点
            n_neighbors_to_find = min(k_k, n_samples_k)

            # 批量计算类 j 中所有样本 到 类 k 的最近邻
            nn = NearestNeighbors(n_neighbors=n_neighbors_to_find,
                                  metric='euclidean',
                                  algorithm='auto',
                                  n_jobs=-1)
            nn.fit(class_k_samples)

            # 批量查询！
            # distances_jk 的 shape 是 (n_samples_j, n_neighbors_to_find)
            distances_jk, _ = nn.kneighbors(class_j_samples, n_neighbors=n_neighbors_to_find)

            # 提取第 k 个邻居的距离
            # 原始逻辑是取索引 k_k-1，如果 k_k > N_k，则取最后一个
            # 我们的 `distances_jk` 最后一列 (索引 -1) 正好是第 n_neighbors_to_find 个邻居
            # 这完美匹配了原始逻辑
            between_k_distances = distances_jk[:, -1]

            # 矢量化计算隶属度
            # [鲁棒性] 原始代码的 "active" 部分没有处理 between_k_distances == 0 的情况
            # 这会导致 1/0 = inf。
            # 原始代码的 "commented" 部分 (if between_class_k_distance != 0: ... else: ... = 1)
            # 似乎更健壮。我们采用 "commented" 部分的逻辑：

            membership_k = np.ones(n_samples_j)  # 默认隶属度为 1 (对应 distance == 0)

            # 仅对距离 > 0 的进行计算
            valid_mask = (between_k_distances > 0)
            if np.any(valid_mask):
                membership_k[valid_mask] = (1.0 / between_k_distances[valid_mask]) / dist_sum_recip_k

            # 矢量化赋值
            fuzzy_membership_matrix[class_j_indices, k] = membership_k

    return fuzzy_membership_matrix
'''
def fuzzy_membership(X, y):
    
    #计算样本i相对于自己类 within class 的 fuzzy membership 和 相对于其他类 between class 的 fuzzy membership
    #:param X:
    #:param y:
    #:return:
    
    # 计算每个唯一标签的实例数量
    unique_labels, counts = np.unique(y, return_counts=True)
    n_classes = len(unique_labels)
    num_instances, _ = X.shape

    # 计算每一个目标值 最近邻 K 值
    k_vals = np.ceil(np.sqrt(counts) * constants.K2)

    within_class_k_distances, distances_sum_reciprocal = kth_nearest_neigh_distance(X, y)
    fuzzy_membership_matrix = np.zeros((num_instances, n_classes))

    for j in range(n_classes):
        class_samples_indices = np.where(y == unique_labels[j])[0]
        # 计算 within class fuzzy membership
        for i in class_samples_indices:
            if(within_class_k_distances[i] != 0):
                fuzzy_membership_matrix[i][j] = 1 / within_class_k_distances[i] / distances_sum_reciprocal[j]
            else:
                # 当前类只有1个实例,within_class_k_distances[i]=0,fuzzy_membership_matrix[i][j] = 1或者0
                fuzzy_membership_matrix[i][j] = 1

        # 计算 between class fuzzy membership
        for k in range(n_classes):
            if (k == j): continue

            #if(counts[j] <= counts[k]): # 类j的实例为少数类实例(positive class)
            #    imbalance_ratio = counts[k] / counts[j]
            #else:
            #    imbalance_ratio = counts[j] / counts[k]

            indices = np.where(y == unique_labels[k])[0]
            samples_between = X[indices]
            for i in class_samples_indices:
                sam = X[i]
                nearest_distances, sorted_indices = cal_nearst_neighbors(sam, samples_between, int(k_vals[k]))
                # 求最近邻居平均值，存储在最后一个元素
                #nearest_distances[int(k_vals[k]) - 1] = sum(nearest_distances) / int(k_vals[k])

                between_class_k_distance = nearest_distances[int(k_vals[k]) - 1]
                
                #if(between_class_k_distance != 0):
                #    fuzzy_membership_matrix[i][k] = 1 / between_class_k_distance / distances_sum_reciprocal[k] # * imbalance_ratio
                #else:
                #    fuzzy_membership_matrix[i][k] = 1
                
                if (distances_sum_reciprocal[k] != 0):
                    fuzzy_membership_matrix[i][k] = 1 / between_class_k_distance / distances_sum_reciprocal[k]  # * imbalance_ratio
                else:
                    fuzzy_membership_matrix[i][k] = 0

    return fuzzy_membership_matrix
    '''

def resample_pro(safe_samples, overlap_samples, noise_samples, outlier_samples, X, labels, membership_matrix, membership_mean_value_matrix):

    safe_samples, overlap_samples, noise_samples, outlier_samples, labels = list_to_array(safe_samples, overlap_samples, noise_samples, outlier_samples, labels)
    #unique_labels = np.unique(labels)
    unique_labels, counts = np.unique(labels, return_counts=True)
    n_classes = len(unique_labels)
    n_overSafe_samples = [0] * len(unique_labels)

    # 计算每一个目标值 最近邻 K 值
    k_vals = np.ceil(np.sqrt(counts) * constants.K2)

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

        #print(f"++++++++++++++++++++++++++++++++++++++++++类:{j}")

        sys_nums = max_samples_nums - n_overSafe_samples[j]
        if(sys_nums == 0): continue


        #indices_safe_j = safe_samples[np.where((labels[safe_samples] == unique_labels[j]))[0]]
        #indices_overlap_j = overlap_samples[np.where((labels[overlap_samples] == unique_labels[j]))[0]]
        #indices = np.concatenate([indices_safe_j, indices_overlap_j])


        nums = cal_samples_weight(safe_samples, overlap_samples, sys_nums, membership_matrix, j, labels)

        Syns_safe, Syns_y_safe = create_samples_around_safe(safe_samples, overlap_samples, j, X, nums, labels)

        #Syns_overlap, Syns_y_overlap = create_samples_around_overlap(safe_samples, overlap_samples, X, j, nums, labels, membership_matrix,
        #                                  membership_mean_value_matrix)

        Syns.extend(Syns_safe)
        #Syns.extend(Syns_overlap)
        Syns_y.extend(Syns_y_safe)
        #Syns_y.extend(Syns_y_overlap)


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

        indices_safe_j = safe_samples[np.where((labels[safe_samples] == unique_labels[j]))[0]]
        indices_overlap_j = overlap_samples[np.where((labels[overlap_samples] == unique_labels[j]))[0]]
        indices = np.concatenate([indices_safe_j, indices_overlap_j])

        n_safe_j = len(indices_safe_j)
        n_overlap_j = len(indices_overlap_j)
        n_safe_overlap = n_safe_j + n_overlap_j

        # 计算d_safe 并 d_overlap 的集合
        indices_safe_overlap = np.concatenate((safe_samples, overlap_samples))
        samples_X = [X[index] for index in indices_safe_overlap]

        #for i in indices_safe_j:
        for i in indices:
            distances_in_samples_X, indices_in_samples_X = cal_nearst_neighbors(X[i], samples_X, constants.K1)
            # 围绕该样本和成的样本数
            #sys_num = int(sys_nums / n_safe_j)
            sys_num = int(sys_nums / n_safe_overlap)
            for _ in range(0, sys_num):
                candidate_index = indices_in_samples_X[np.random.choice(len(indices_in_samples_X))]
                candidate_sample = samples_X[candidate_index]
                # Interpolation Oversampling
                synthetic_sample = X[i] + np.random.rand() * (X[i] - candidate_sample)
                Syns.append(synthetic_sample)
                Syns_y.append(unique_labels[j])

    return Syns, Syns_y

def weight_safe_and_overlap(class_index, safe_j, overlap_j, membership_matrix):
    


    1