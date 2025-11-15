import numpy as np
import pandas as pd

import constants
from preprocess_data import DATA_RESULT_PATH
from scipy.stats import friedmanchisquare, rankdata, wilcoxon


def avg_rank_shaffer():
    '''
    使用 Friedman test 计算平均排名
    :return:
    '''
    for classifer in constants.CLASSIFIERS_NAME:
        for metric in constants.METRICS:

            path = DATA_RESULT_PATH / 'merge' / ('%s.%s.csv' % (classifer, metric))
            df = pd.read_csv(path)

            selData = df.values
            selData = selData[:, 1:]

            # 计算Friedman ranking test
            _, p_value = friedmanchisquare(*selData.T)
            #_, p_value = friedmanchisquare(*selData)

            print(f"classifer:{classifer}, metric:{metric}, p_value:{p_value}")


            meanRvar = []
            # 如果 p_value 较小，拒绝零假设，说明至少有一个方法的性能与其他方法不同
            if p_value < 0.05:
            #if p_value < 0.6:

                # 计算排名
                ranks = rankdata(-selData, axis=1)
                #if(metric == 'CEN'):
                #    ranks = rankdata(selData, axis=1)
                #else:
                #    ranks = rankdata(-selData, axis=1)

                # 计算每个方法的平均排名
                meanRvar = ranks.mean(axis=0)

                # 计算每一个selData最后一列与其他剩余列的p值
                p_values_pairwise = []
                for i in range(selData.shape[1] - 1):
                    #xxx  = selData[:, i]
                    #yyy = selData[:, -1]
                    _, p_value_pairwise = wilcoxon(selData[:, i], selData[:, -1])
                    p_values_pairwise.append(p_value_pairwise)
                    print(f"classifier: {classifer}, metric: {metric}, pairwise p_value with column {i}: {p_value_pairwise}")


            else:
                ranks = rankdata(-selData, axis=1)
                # 计算每个方法的平均排名
                meanRvar = ranks.mean(axis=0)

                # 计算每一个selData最后一列与其他剩余列的p值
                p_values_pairwise = []
                for i in range(selData.shape[1] - 1):
                    _, p_value_pairwise = wilcoxon(selData[:, i], selData[:, -1])
                    p_values_pairwise.append(p_value_pairwise)
                    print(f"classifier: {classifer}, metric: {metric}, pairwise p_value with column {i}: {p_value_pairwise}")

                # 检查 'Avg_rank' 行是否已存在
                head = df.columns[0]
                if 'Avg_rank' in df[head].values:
                    # 如果存在，删除该行
                    df = df[df[head] != 'Avg_rank']

                # 在 DataFrame 的最后一行添加 meanRvar
                df.loc[len(df)] = ['Avg_rank'] + [round(value, 2) for value in meanRvar]

                # 将更新后的 DataFrame 写回文件
                df.to_csv(path, index=False)

                # 保存到excel文件
                # 将数据保存为Excel文件
                excel_path = DATA_RESULT_PATH / 'xlsx' / ('%s.%s.xlsx' % (classifer, metric))
                df.to_excel(excel_path, index=False)

            # 将每个值保留两位小数
            meanRvar = np.round(meanRvar, 2)
            print("Mean ranks for each method:", meanRvar)

            # 检查 'Avg_rank' 行是否已存在
            head = df.columns[0]
            if 'Avg_rank' in df[head].values:
                # 如果存在，删除该行
                df = df[df[head] != 'Avg_rank']

            # 在 DataFrame 的最后一行添加 meanRvar
            df.loc[len(df)] = ['Avg_rank'] + [round(value, 2) for value in meanRvar]

            # 将更新后的 DataFrame 写回文件
            df.to_csv(path, index=False)

            # 保存到excel文件
            # 将数据保存为Excel文件
            excel_path = DATA_RESULT_PATH / 'xlsx' / ('%s.%s.xlsx' % (classifer, metric))
            df.to_excel(excel_path, index=False)

if __name__ == '__main__':
    avg_rank_shaffer()