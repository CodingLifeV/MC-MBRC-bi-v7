from pathlib import Path

import numpy as np
import pandas as pd
import smote_variants
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler

import constants
from classifers import RF_eval, CART_eval, KNN_eval, MLP_eval, NB_eval, SVM_eval
from cross_valuation import read_data, write_data
from evaluation import custom_scorer_multi
import smote_variants as sv

from evaluation_binary import custom_scorer

DATA_RES_PATH = Path(__file__).parent / 'data' / 'results'



def create_data():
    for name in constants.DATA_NAME:
        for partition in range(1, constants.N_TIMES + 1):
            for fold in range(1, constants.FOLD + 1):
                print(f"====dataset name:{name},partition:{partition},fold:{fold}========")

                X_train, y_train = read_data(name, partition, fold, flag='TRAIN')
                print(f"==============read data finished....============")

                #oversampler = sv.MDO()
                #oversampler = sv.SMOTE()
                #oversampler = sv.MWMOTE()
                #oversampler = sv.Borderline_SMOTE2()
                #oversampler = sv.kmeans_SMOTE()
                #oversampler = sv.NRAS()
                oversampler = sv.A_SUWO(n_jobs=-1)
                print(f"==============Oversampling begin....============")
                X_train_res, y_train_res = oversampler.fit_resample(X_train, y_train)
                print(f"==============Oversampling End....============")

                print(f"==============write data Begin....============")
                write_data(name, partition, fold, X_train_res, y_train_res, 'TRAIN_RES')
                print(f"==============write data End....============")
# ADASYN
def evaluate_Others(classifer):

    final_results = {'dataset': [], 'F1': [], 'AUC': [], 'G-mean': []}

    for name in constants.DATA_NAME:
        results = []
        for partition in range(1, constants.N_TIMES + 1):
            for fold in range(1, constants.FOLD + 1):
                print(f"======classifer:{classifer},dataset name:{name},partition:{partition},fold:{fold}========")

                X_test, y_test = read_data(name, partition, fold, flag='TEST')
                X_train, y_train = read_data(name, partition, fold, flag='TRAIN')

                # 归一化数据
                scaler1 = MinMaxScaler().fit(X_test)
                X_test = np.c_[scaler1.transform(X_test)]
                scaler2 = MinMaxScaler().fit(X_train)
                X_train = np.c_[scaler2.transform(X_train)]

                #oversampler = smote_variants.ADASYN()
                #oversampler = smote_variants.MWMOTE()
                #oversampler = smote_variants.A_SUWO()
                oversampler = smote_variants.SMOTE()
                #oversampler = smote_variants.G_SMOTE()
                #oversampler = smote_variants.CCR()
                #oversampler = smote_variants.Borderline_SMOTE2()
                #oversampler = smote_variants.kmeans_SMOTE()
                #oversampler = smote_variants.SYMPROD()
                #oversampler = smote_variants.NRAS()

                X_train_res, y_train_res = oversampler.sample(X_train, y_train)

                if classifer == 'RF':
                    y_pred = RF_eval(X_train_res, y_train_res, X_test)
                if classifer == 'CART':
                    y_pred = CART_eval(X_train_res, y_train_res, X_test)
                if classifer == 'KNN':
                    y_pred = KNN_eval(X_train_res, y_train_res, X_test)
                if classifer == 'MLP':
                    y_pred = MLP_eval(X_train_res, y_train_res, X_test)
                if classifer == 'NB':
                    y_pred = NB_eval(X_train_res, y_train_res, X_test)

                result = custom_scorer(y_test, y_pred)
                results.append(result)

        average_f1 = np.mean([result['f1'] for result in results])
        average_auc = np.mean([result['auc'] for result in results])
        average_g_mean = np.mean([result['g_mean'] for result in results])
        print(f"f1:{average_f1},auc:{average_auc},g_mean:{average_g_mean}")

        final_results['dataset'].append(name)
        final_results['F1'].append(round(average_f1, 4))
        final_results['G-mean'].append(round(average_g_mean, 4))
        final_results['AUC'].append(round(average_auc, 4))

    # 创建DataFrame以存储结果
    results_df = pd.DataFrame(final_results)
    #path = DATA_RES_PATH / constants.OUR_METHOD_NAME
    path = DATA_RES_PATH / 'SMOTE'
    path = path / (f'{classifer}_results.csv')
    #results_df.to_csv(path, index=False)
    # 不覆盖原文件
    results_df.to_csv(path, mode='a', header=False, index=False)


def evaluate(classifer):
    final_results = {'dataset': [], 'F1': [], 'AUC': [], 'G-mean': []}

    # 读取训练之后的数据，进行分类器验证
    for name in constants.DATA_NAME:
        results = []
        for partition in range(1, constants.N_TIMES + 1):
            for fold in range(1, constants.FOLD + 1):
                print(f"======classifer:{classifer},dataset name:{name},partition:{partition},fold:{fold}========")

                X_test, y_test = read_data(name, partition, fold, flag='TEST')
                #X_train_res, y_train_res = read_data(name, partition, fold, flag='TRAIN_RES')
                # No Sampling
                X_train_res, y_train_res = read_data(name, partition, fold, flag='TRAIN')

                # 归一化数据
                scaler1 = MinMaxScaler().fit(X_train_res)
                X_train_res = np.c_[scaler1.transform(X_train_res)]
                scaler1 = MinMaxScaler().fit(X_test)
                X_test = np.c_[scaler1.transform(X_test)]

                #if(partition == 22 and fold == 1):
                #    partition

                if classifer == 'RF':
                    y_pred = RF_eval(X_train_res, y_train_res, X_test)
                if classifer == 'SVM':
                    y_pred = SVM_eval(X_train_res, y_train_res, X_test)
                if classifer == 'CART':
                    y_pred = CART_eval(X_train_res, y_train_res, X_test)
                if classifer == 'KNN':
                    y_pred = KNN_eval(X_train_res, y_train_res, X_test)
                if classifer == 'MLP':
                    y_pred = MLP_eval(X_train_res, y_train_res, X_test)
                if classifer == 'NB':
                    y_pred = NB_eval(X_train_res, y_train_res, X_test)

                result = custom_scorer(y_test, y_pred)
                results.append(result)

        average_f1 = np.mean([result['f1'] for result in results])
        average_auc = np.mean([result['auc'] for result in results])
        average_g_mean = np.mean([result['g_mean'] for result in results])
        print(f"f1:{average_f1},auc:{average_auc},g_mean:{average_g_mean}")

        final_results['dataset'].append(name)
        final_results['F1'].append(round(average_f1, 4))
        final_results['G-mean'].append(round(average_g_mean, 4))
        final_results['AUC'].append(round(average_auc, 4))

    # 创建DataFrame以存储结果
    results_df = pd.DataFrame(final_results)
    #path = DATA_RES_PATH / constants.OUR_METHOD_NAME
    path = DATA_RES_PATH / 'cWGAN‑OS'
    path = DATA_RES_PATH / 'B2BGAN'
    path = DATA_RES_PATH / 'No_Sampling'


    path = path / (f'{classifer}_results.csv')
    #results_df.to_csv(path, index=False)
    # 不覆盖原文件
    results_df.to_csv(path, mode='a', header=False, index=False)



if __name__ == '__main__':
    #for classifer in constants.CLASSIFIERS_NAME:
    #    evaluate(classifer)

    for classifer in constants.CLASSIFIERS_NAME:
        create_data()
        #evaluate_Others(classifer)