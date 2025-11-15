
import numpy as np

def custom_scorer_multi(conf_matrix):
    AvAcc = calculate_avacc(conf_matrix)
    CEN = calculate_CEN(conf_matrix)
    CBA = calculate_CBA(conf_matrix)
    fbm = calculate_fbm(1, conf_matrix)
    #     final_results = {'dataset': [], 'AvAcc': [], 'CEN': [], 'CBA': [], 'mGM': []}
    return {'AvAcc': AvAcc, 'CEN': CEN, 'CBA': CBA, 'fbm': fbm}



# macro-F1 MG MAUC




def calculate_avacc(z):
    '''
    计算 avacc
    :param z:  conf_matrix
    :return:
    '''
    corr = np.diag(z)
    t = np.sum(z, axis=1)
    p = np.sum(z, axis=0)
    res = 0

    for i in range(len(corr)):
        res += sum(corr) / (sum(corr) + t[i] + p[i] - 2 * corr[i])

    res = float(res / len(corr))
    return res

def calculate_CEN(conf_matrix):
    '''
    calculate CEN,  the misclassification information involves both the information of how
    the samples with true class label cl i have been misclassified to the other N classes and the information
    of how the samples of the other N classes have been misclassified to class cli .
    :param conf_matrix:
    :return:
    '''
    num_classes = conf_matrix.shape[0]

    row_sums = np.sum(conf_matrix, axis=1)
    col_sums = np.sum(conf_matrix, axis=0)
    res = 0

    for j in range(num_classes):
        P = (row_sums[j] + col_sums[j]) / (2 * np.sum(conf_matrix))
        res += P * calculate_cen_class(conf_matrix, j)

    #print("res:", res)
    return float(res)

def calculate_cen_class(z, j):
    n = z.shape[0]
    row_sums = np.sum(z, axis=1)
    col_sums = np.sum(z, axis=0)
    probs = np.full((n, n), np.nan)

    for k in range(n):
        if k != j:
            #print("row_sums[j] + col_sums[k]:", row_sums[j] + col_sums[k])
            # p_jk
            probs[j, k] = z[j, k] / (row_sums[j] + col_sums[j])
            # p_kj
            probs[k, j] = z[k, j] / (row_sums[j] + col_sums[j])
        else:
            # p_jj
            probs[j, j] = 0

    if (row_sums[j] + col_sums[j]) == 0:
        res = np.nan
        return res
    else:
        res = 0
        for k in set(range(n)) - {j}:
            l1 = 0
            l2 = 0
            if probs[j, k] != 0:
                l1 = probs[j, k] * np.log(probs[j, k]) / np.log(2 * (n - 1))
            if probs[k, j] != 0:
                l2 = probs[k, j] * np.log(probs[k, j]) / np.log(2 * (n - 1))
            res += l1 + l2

    return -res

def calculate_CBA(z):
    '''
    计算 CBA
    :param z:
    :return:
    '''
    n = z.shape[0]

    def across(u, v, t):
        if np.sum(u) == 0 and np.sum(v) == 0:
            return 0
        else:
            return t / max(np.sum(u), np.sum(v))

    xyacross = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            xyacross[i, j] = across(z[i, :], z[:, j], z[i, j])

    return np.mean(np.diag(xyacross))


def calculate_mavg(z):
    '''
    计算mGM
    :param z:
    :return:
    '''
    corr = np.diag(z)
    t = np.sum(z, axis=1)
    res = 1
    for i in range(len(corr)):
        if t[i] == 0:
            continue
        res *= corr[i] / t[i]

    res = float(res ** (1 / len(corr)))
    return res

## Arithmetic Macro-average of precision in each class
def mava(confusion_matrix):
    # 获取混淆矩阵的对角线元素（即每个类别的正确预测数量）
    correct_predictions = np.diag(confusion_matrix)
    # 计算每个类别的总预测数量
    total_predictions = np.sum(confusion_matrix, axis=1)

    # 计算每个类别的召回率
    recalls = correct_predictions / total_predictions

    # 计算宏平均召回率
    macro_avg_recall = np.mean(recalls)

    return macro_avg_recall


def precM(confusion_matrix):
    # 获取混淆矩阵的对角线元素（即每个类别的正确预测数量）
    correct_predictions = np.diag(confusion_matrix)

    # 计算每个类别的总预测数量
    total_predictions = np.sum(confusion_matrix, axis=0)

    # 移除预测数量为 0 的类别
    idx = np.where(total_predictions != 0)[0]
    if len(idx) == 0:
        return np.nan

    total_predictions

    # 计算每个类别的精确率
    precisions = correct_predictions[idx] / total_predictions[idx]

    # 计算宏平均精确率
    macro_avg_precision = np.mean(precisions)

    return macro_avg_precision

# Mean F𝛽 measure
def calculate_fbm(beta, confusion_matrix):
    rec = mava(confusion_matrix)
    prec = precM(confusion_matrix)
    res = ((1 + beta**2) * prec * rec) / (beta**2 * prec + rec)
    return res

# Test
if __name__ == '__main__':
    # 定义给定的值
    values = [5, 2, 0, 10, 200, 1, 3, 5, 300]
    # 将值转换为数组
    conf_matrix = np.array(values)
    # 调整数组的形状为 3x3
    conf_matrix = conf_matrix.reshape(3, 3)
    xxx = calculate_fbm(1, conf_matrix)
    print(f"{xxx}")





