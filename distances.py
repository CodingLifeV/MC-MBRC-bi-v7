import numpy as np
from scipy.spatial import distance
from scipy.spatial.distance import euclidean


# 定义计算距离的函数
def euclidean_distance(x, c):
    '''
    :param x:
    :param c:
    :return:
    '''
    return np.sqrt(np.sum((x - c)**2))

# 计算某个样本的最近k个邻居
def cal_nearst_neighbors(sam, samples, k):
    '''
    样本sam距离样本集合samples的最近的k个邻居
    :param sam: 待查询的样本
    :param samples: 样本集合
    :param k: 最近邻居的数量
    :return: 最近邻居的索引和对应的距离
    '''
    distances = []
    indices = []

    if(len(samples) < k):
        k = len(samples)

    # 计算样本sam与样本集合samples中每个样本的距离
    for i, sample in enumerate(samples):

        #print(f"j:{i}")

        if np.array_equal(sample, sam):  # 如果当前样本是sam本身，则跳过
            continue


        dist = euclidean(sam, sample)  # 欧几里得距离
        distances.append(dist)
        indices.append(i)

    # 对距离进行排序并取出前k个最近邻居
    k = int(k)
    sorted_indices = np.argsort(distances)[:k]
    nearest_distances = [distances[i] for i in sorted_indices]
    #nearest_indices = [indices[i] for i in sorted_indices]

    return nearest_distances, sorted_indices

def distance(x, y, p_norm=1):
    return np.sum(np.abs(x - y) ** p_norm) ** (1 / p_norm)
