from pathlib import Path

import pandas as pd

import constants
from run import DATA_RES_PATH
from statistic_test import avg_rank_shaffer

DATA_RES_MERGE_PATH = Path(__file__).parent / 'data' / 'results' / 'merge'

def merge_data():

    for cls in constants.CLASSIFIERS_NAME:

        for metric in constants.METRICS:
            data_frames = []
            # 通过循环读取和提取 metric 列
            for methodName in constants.ALL_METHOD_NAMES:
                # ALL_METHOD_NAMES = ['ADASYN', 'MWMOTE', 'G-SMOTE', 'A_SUWO', 'OREM', 'PAIO', OUR_METHOD_NAME]
                if methodName == 'ADASYN':
                    path = DATA_RES_PATH / 'ADASYN' / ('%s_results.csv' % cls)
                if methodName == 'MWMOTE':
                    path = DATA_RES_PATH / 'MWMOTE' / ('%s_results.csv' % cls)
                if methodName == 'OREM':
                    path = DATA_RES_PATH / 'OREM' / ('%s_results.csv' % cls)
                if methodName == 'RBO':
                    path = DATA_RES_PATH / 'RBO' / ('%s_results.csv' % cls)
                if methodName == 'A_SUWO':
                    path = DATA_RES_PATH / 'A_SUWO' / ('%s_results.csv' % cls)
                if methodName == 'PAIO':
                    path = DATA_RES_PATH / 'PAIO' / ('%s_results.csv' % cls)
                if methodName == 'G-SMOTE':
                    path = DATA_RES_PATH / 'G-SMOTE' / ('%s_results.csv' % cls)
                if methodName == 'CCR':
                    path = DATA_RES_PATH / 'CCR' / ('%s_results.csv' % cls)
                if methodName == 'Borderline_SMOTE2':
                    path = DATA_RES_PATH / 'Borderline_SMOTE2' / ('%s_results.csv' % cls)
                if methodName == 'kmeans_SMOTE':
                    path = DATA_RES_PATH / 'kmeans_SMOTE' / ('%s_results.csv' % cls)
                if methodName == 'SYMPROD':
                    path = DATA_RES_PATH / 'SYMPROD' / ('%s_results.csv' % cls)
                if methodName == 'NRAS':
                    path = DATA_RES_PATH / 'NRAS' / ('%s_results.csv' % cls)
                if methodName == 'SWIMRBF':
                    path = DATA_RES_PATH / 'SWIMRBF' / ('%s_results.csv' % cls)
                if methodName == 'CCO':
                    path = DATA_RES_PATH / 'CCO' / ('%s_results.csv' % cls)
                if methodName == 'MLOS':
                    path = DATA_RES_PATH / 'MLOS' / ('%s_results.csv' % cls)
                if methodName == 'No_Sampling':
                    path = DATA_RES_PATH / 'No_Sampling' / ('%s_results.csv' % cls)
                if methodName == 'SMOTE':
                    path = DATA_RES_PATH / 'SMOTE' / ('%s_results.csv' % cls)
                if methodName == 'cWGAN-OS':
                    path = DATA_RES_PATH / 'cWGAN_OS' / ('%s_results.csv' % cls)
                if methodName == 'B2BGAN':
                    path = DATA_RES_PATH / 'B2BGAN' / ('%s_results.csv' % cls)
                if methodName == constants.OUR_METHOD_NAME:
                    path = DATA_RES_PATH / constants.OUR_METHOD_NAME / ('%s_results.csv' % cls)

                data_frames.append(read_and_extract_avacc(path, metric, methodName))

            # 过滤出符合条件的数据集
            filtered_data_frames = [df[df['dataset'].isin(constants.DataSet_MERGE)] for df in data_frames]
            # 合并数据
            filtered_data_frames[0].drop_duplicates(subset=['dataset'], inplace=True)
            filtered_data_frames[1].drop_duplicates(subset=['dataset'], inplace=True)
            # 合并数据
            merged_data = pd.merge(filtered_data_frames[0], filtered_data_frames[1], on='dataset')
            for i in range(2, len(filtered_data_frames)):
                merged_data = pd.merge(merged_data, filtered_data_frames[i], on='dataset',
                                       suffixes=('', f'_{constants.ALL_METHOD_NAMES[i]}'))

            '''
            # 合并数据
            data_frames[0].drop_duplicates(subset=['dataset'], inplace=True)
            data_frames[1].drop_duplicates(subset=['dataset'], inplace=True)
            merged_data = pd.merge(data_frames[0], data_frames[1], on='dataset')

            for i in range(2, len(data_frames)):
                merged_data = pd.merge(merged_data, data_frames[i], on='dataset',
                                       suffixes=('', f'_{constants.ALL_METHOD_NAMES[i]}'))
            '''
            # 保存结果到 CSV 文件
            path = DATA_RES_MERGE_PATH / ('%s.%s.csv' % (cls, metric))
            merged_data[['dataset'] + [f'{method}' for method in constants.ALL_METHOD_NAMES]].to_csv(
                path, index=False)

    avg_rank_shaffer()


def read_and_extract_avacc(file_path, metric, suffix):
    data = pd.read_csv(file_path)
    return data[['dataset', metric]].rename(columns={metric: f'{suffix}'})


if __name__ == '__main__':
    merge_data()