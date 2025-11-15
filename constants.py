

#
DATA_NAME = ['haberman', 'new-thyroid1', 'ecoli', 'newthyroid2', 'yeast5',
                 'yeast-2_vs_4', 'led7digit-0-2-4-5-6-7-8-9_vs_1', 'Pima', 'winequality-red-8_vs_6', 'winequality-white-3_vs_7',
                 'pen.global', 'ecoli-0_vs_1', 'ERA', 'wpbc', 'glass2']
DATA_NAME = ['creditcard1']


#DataSet_MERGE = ['glass1']
#最终的数据集:
DataSet_MERGE = ['haberman', 'new-thyroid1', 'ecoli', 'newthyroid2', 'yeast5',
                 'yeast-2_vs_4', 'led7digit-0-2-4-5-6-7-8-9_vs_1', 'Pima', 'winequality-red-8_vs_6', 'winequality-white-3_vs_7',
                 'pen.global', 'ecoli-0_vs_1', 'ERA', 'wpbc', 'glass2']

# new dataset: diabetes(其实是pima) Biodeg seeds Parkinsons Transfusion ERA

# 分类器名称 'RF', 'MLP', 'CART', 'NB', 'KNN'
CLASSIFIERS_NAME = ['RF', 'CART', 'NB']

# 评估矩阵
METRICS = ['auc', 'g_mean','f1']
#METRICS = ['f1']

# 交叉验证次数
N_TIMES = 10

# 交叉验证折数
FOLD = 2


# 方法名字
OUR_METHOD_NAME='MC-MBRC-Pro'

# 删除  ADASYN MWMOTE CCR G-SMOTE
#ALL_METHOD_NAMES = ['Borderline_SMOTE2', 'CCR', 'MWMOTE', 'G-SMOTE', 'OREM', 'PAIO', OUR_METHOD_NAME]
# Borderline_SMOTE2(2005)  kmeans_SMOTE(2018) A_SUWO(2016) OREM(2022)
ALL_METHOD_NAMES = ['MWMOTE' ,'Borderline_SMOTE2', 'kmeans_SMOTE', 'NRAS', 'A_SUWO', 'OREM', 'SWIMRBF', 'CCO', 'MLOS', OUR_METHOD_NAME]

ALL_METHOD_NAMES = ['No_Sampling', 'SMOTE', 'MWMOTE', 'Borderline_SMOTE2', 'kmeans_SMOTE', 'NRAS', 'A_SUWO', 'OREM', 'SWIMRBF', 'CCO', 'MLOS', 'cWGAN-OS', 'B2BGAN', OUR_METHOD_NAME]


#ALL_METHOD_NAMES = [ 'MWMOTE', 'NRAS', OUR_METHOD_NAME]

# membership degree M
M = 2

# K1个邻居
#K1 = 5
#K3 = 10

# clean energy
# newthyroid : 0.5  1.0  10 0.25
#ENERGY = 0.25

# 系数
K2 = 1

# 指定样本是overlap或者是outlier样本系数, 默认为3, 数值越小,越容易变为outlier样本
COEF = 3

# 寻找清理子区域,用于合成子样本,连续 Q 个 samples是主要类样本, 默认 3
Q = 3