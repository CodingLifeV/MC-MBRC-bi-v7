from pathlib import Path

import pandas as pd
from sklearn.model_selection import KFold

import constants
from resampling import DAR
from preprocess_data import read_CSV

DATA_PATH = Path(__file__).parent / 'data'
DATA_ORIGINAL_PATH = Path(__file__).parent / 'data' / 'original'
DATA_FOLD_PATH = Path(__file__).parent / 'data' / 'folds'
DATA_FOLD_PATH_OREM = Path(__file__).parent / 'data' / 'folds-OREM'
DATA_FOLD_PATH_RBO = Path(__file__).parent / 'data' / 'folds-RBO'
DATA_FOLD_PATH_PAIO = Path(__file__).parent / 'data' / 'folds-PAIO'
DATA_FOLD_PATH_MWMOTE = Path(__file__).parent / 'data' / 'folds-MWMOTE'
DATA_FOLD_PATH_MLOS = Path(__file__).parent / 'data' / 'folds-MLOS'

DATA_FOLD_PATH_SWIMRBF = Path(__file__).parent / 'data' / 'folds-SWIMRBF'
DATA_FOLD_PATH_CCO = Path(__file__).parent / 'data' / 'folds-CCO'
DATA_FOLD_PATH_SMOTE = Path(__file__).parent / 'data' / 'folds-SMOTE'
DATA_FOLD_PATH_cWGAN_OS = Path(__file__).parent / 'data' / 'folds-cWGAN_OS'
DATA_FOLD_PATH_B2BGAN = Path(__file__).parent / 'data' / 'folds-B2BGAN'
DATA_FOLD_PATH_Borderline_SMOTE2 = Path(__file__).parent / 'data' / 'folds-Borderline_SMOTE2'
DATA_FOLD_PATH_kmeans_SMOTE = Path(__file__).parent / 'data' / 'folds-kmeans_SMOTE'
DATA_FOLD_PATH_NRAS = Path(__file__).parent / 'data' / 'folds-NRAS'
DATA_FOLD_PATH_A_SUWO = Path(__file__).parent / 'data' / 'folds-A_SUWO'

def get_cross_val_data(X, y, dataset_name):
    '''
    constants.N_TIMES times constants.FOLD-fold cross validation
    默认30折 2-fold交叉验证
    :param X:
    :param y:
    :return:
    '''
    for partition in range(1, constants.N_TIMES + 1):
        k_fold = KFold(n_splits=constants.FOLD, shuffle=True, random_state=42)
        fold = 0 # 折数
        for train_index, test_index in k_fold.split(X):
            fold += 1
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # 输出 train test 数据到 csv
            write_data(dataset_name, partition, fold, X_train, y_train, 'TRAIN')
            write_data(dataset_name, partition, fold, X_test, y_test, 'TEST')

            #path = DATA_FOLD_PATH / dataset_name / ('%s.%d.%d.train.csv' % (dataset_name, partition, fold))
            #X_final, y_final = DAR(path)
            #write_data(dataset_name, partition, fold, X_final, y_final, 'TRAIN_RES')


def get_cross_train_res_data(dataset_name):
    for partition in range(1, constants.N_TIMES + 1):
        for fold in range(1, constants.FOLD + 1):
            print(f"===========dataset name:{dataset_name},partition:{partition},fold:{fold}===========")
            path = DATA_FOLD_PATH / dataset_name / ('%s.%d.%d.train.csv' % (dataset_name, partition, fold))
            X_final, y_final = DAR(path)
            write_data(dataset_name, partition, fold, X_final, y_final, 'TRAIN_RES')


def write_data(name, partition, fold, X, y, flag):
    '''
    输出数据到指定文件夹
    :param name:
    :param partition:
    :param fold:
    :param X:
    :param y:
    :param flag:
    :return:
    '''
    if(flag == 'TRAIN'):
        path = DATA_FOLD_PATH / name / ('%s.%d.%d.train.csv' % (name, partition, fold))
    if(flag == 'TEST'):
        path = DATA_FOLD_PATH / name / ('%s.%d.%d.test.csv' % (name, partition, fold))
    if(flag == 'TRAIN_RES'):
        #path = DATA_FOLD_PATH_MDO / name / ('%s.%d.%d.train_res.csv' % (name, partition, fold))
        path = DATA_FOLD_PATH / name / ('%s.%d.%d.train_res.csv' % (name, partition, fold))
        path = DATA_FOLD_PATH_SMOTE / name / ('%s.%d.%d.train_res.csv' % (name, partition, fold))
        path = DATA_FOLD_PATH_MWMOTE / name / ('%s.%d.%d.train_res.csv' % (name, partition, fold))
        path = DATA_FOLD_PATH_Borderline_SMOTE2 / name / ('%s.%d.%d.train_res.csv' % (name, partition, fold))
        path = DATA_FOLD_PATH_kmeans_SMOTE / name / ('%s.%d.%d.train_res.csv' % (name, partition, fold))
        path = DATA_FOLD_PATH_NRAS / name / ('%s.%d.%d.train_res.csv' % (name, partition, fold))
        path = DATA_FOLD_PATH_A_SUWO / name / ('%s.%d.%d.train_res.csv' % (name, partition, fold))
        path = DATA_FOLD_PATH_MLOS / name / ('%s.%d.%d.train_res.csv' % (name, partition, fold))

    # Convert X to DataFrame if it's a NumPy array
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
        y = pd.DataFrame(y, columns=["target"])

    # Check if the directory exists, create it if not
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.concat([X, y], axis=1)
    df.to_csv(path, index=False)

    # Convert X to DataFrame if it's a NumPy array


def read_data(name, partition, fold, flag):

    if (flag == 'TRAIN'):
        path = DATA_FOLD_PATH / name / ('%s.%d.%d.train.csv' % (name, partition, fold))
    if (flag == 'TEST'):
        path = DATA_FOLD_PATH / name / ('%s.%d.%d.test.csv' % (name, partition, fold))
    if (flag == 'TRAIN_RES'):
        path = DATA_FOLD_PATH / name / ('%s.%d.%d.train_res.csv' % (name, partition, fold))
        #path = DATA_FOLD_PATH_RBO / name / ('%s.%d.%d.train_res.0v1.csv' % (name, partition, fold))
        #path = DATA_FOLD_PATH_OREM / name / ('%s.%d.%d.train_res.csv' % (name, partition, fold))
        #path = DATA_FOLD_PATH_PAIO / name / ('%s.%d.%d.train_res.csv' % (name, partition, fold))
        path = DATA_FOLD_PATH_SWIMRBF / name / ('%s.%d.%d.train_res.csv' % (name, partition, fold))
        path = DATA_FOLD_PATH_MLOS / name / ('%s.%d.%d.train_res.csv' % (name, partition, fold))
        path = DATA_FOLD_PATH_CCO / name / ('%s.%d.%d.train_res.csv' % (name, partition, fold))
        path = DATA_FOLD_PATH_SMOTE / name / ('%s.%d.%d.train_res.csv' % (name, partition, fold))
        path = DATA_FOLD_PATH_cWGAN_OS / name / ('%s.%d.%d.train_res.csv' % (name, partition, fold))
        path = DATA_FOLD_PATH_B2BGAN / name / ('%s.%d.%d.train_res.csv' % (name, partition, fold))

    # 读取test数据
    X, y = read_CSV(path)
    return X, y

def read_CSV_non_numeric(name):

    fold_path = DATA_FOLD_PATH / name

    if not fold_path.exists():
        fold_path.mkdir(parents=True, exist_ok=True)

    original_path = DATA_ORIGINAL_PATH / ('%s.csv' % name)

    metadata_lines = 0

    with open(original_path) as f:
        for line in f:
            if line.startswith('@'):
                metadata_lines += 1

                if line.startswith('@input'):
                    inputs = [l.strip() for l in line[8:].split(',')]
                elif line.startswith('@output'):
                    outputs = [l.strip() for l in line[8:].split(',')]
            else:
                break
    #metadata_lines = 15
    df = pd.read_csv(original_path, skiprows=metadata_lines, header=None)
    df.columns = inputs + outputs
    df = pd.concat([pd.get_dummies(df[inputs]), df[outputs]], axis=1)

    matrix = df.to_numpy()

    # 将 NumPy 数组转换为 DataFrame
    df = pd.DataFrame(matrix)

    # 将 DataFrame 写入 CSV 文件
    df.to_csv(original_path, index=False, header=False)  # 禁用索引和列名



if __name__ == '__main__':


    # 把存在非数值型的数据集转为数值型数据
    #for name in ['lymphography', 'flare', 'car']:
    '''
    for name in ['abalone']:
        read_CSV_non_numeric(name)
    '''

    '''
    print("write train data, test data, and train result data:")

    # 生成 10 times 2 折 train 和 test 数据
    for name in constants.DATA_NAME:
        path = DATA_ORIGINAL_PATH / ('%s.csv' % (name))
        X, y = read_CSV(path)
        get_cross_val_data(X, y, name)
    '''


    # 生成训练结果数据
    for name in constants.DATA_NAME:
        get_cross_train_res_data(name)






