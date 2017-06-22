import pandas as pd
from pandas import DataFrame as df
from sklearn import preprocessing


def one_hot_process():
    origin_data = pd.read_csv('./origin_data/DIABETIC_DATA.csv')
    diabetic_label = origin_data['READMITTED'].replace(['NO', '<30', '>30'], [0, 1, 0])
    origin_data = origin_data.drop('READMITTED', 1)
    origin_data = origin_data.drop('ENCOUNTER_ID', 1)
    origin_data = origin_data.drop('PATIENT_NBR', 1)
    origin_data = origin_data.drop('WEIGHT', 1)
    origin_data = origin_data.drop('PAYER_CODE', 1)
    origin_data = origin_data.drop('MEDICAL_SPECIALTY', 1)
    one_hot_data = pd.get_dummies(origin_data)
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit_transform(one_hot_data)
    one_hot_data['READMITTED'] = diabetic_label
    save_path = './generate_data/one_hot_data.csv'
    one_hot_data.to_csv(save_path, index=False)


def common_process():
    origin_data = pd.read_csv('./origin_data/DIABETIC_DATA.csv')
    common_data = df()
    diabetic_label = origin_data['READMITTED'].replace(['NO', '<30', '>30'], [0, 1, 0])
    # origin_data = origin_data.drop('READMITTED', 1)
    # origin_data = origin_data.drop('ENCOUNTER_ID', 1)
    # origin_data = origin_data.drop('PATIENT_NBR', 1)
    # origin_data = origin_data.drop('WEIGHT', 1)
    # origin_data = origin_data.drop('PAYER_CODE', 1)
    # origin_data = origin_data.drop('MEDICAL_SPECIALTY', 1)
    common_data['RACE'] = origin_data['RACE'].replace(['NO', '<30', '>30'], [0, 1, 0])
    origin_data['READMITTED'] = diabetic_label
    save_path = './generate_data/common_data.csv'
    origin_data.to_csv(save_path, index=False)


if __name__ == '__main__':
    common_process()
