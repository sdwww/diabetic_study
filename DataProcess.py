import time

import pandas as pd
from pandas import DataFrame as df
from sklearn import preprocessing


def data_neural_process():
    origin_data = pd.read_csv('./origin_data/DIABETIC_DATA.csv')
    diabetic_label = origin_data['READMITTED'].replace(['NO', '<30', '>30'], [0, 1, 0])
    origin_data = origin_data.drop('READMITTED', 1)
    origin_data = origin_data.drop('ENCOUNTER_ID', 1)
    origin_data = origin_data.drop('PATIENT_NBR', 1)
    category_variable = ['RACE', 'GENDER', 'WEIGHT', 'ADMISSION_TYPE_ID', 'DISCHARGE_DISPOSITION_ID',
                         'ADMISSION_SOURCE_ID',
                         'PAYER_CODE', 'MEDICAL_SPECIALTY', 'DIAG_1', 'DIAG_2', 'DIAG_3',
                         'MAX_GLU_SERUM', 'A1CRESULT', 'METFORMIN', 'REPAGLINIDE', 'NATEGLINIDE', 'CHLORPROPAMIDE',
                         'GLIMEPIRIDE', 'ACETOHEXAMIDE', 'GLIPIZIDE', 'GLYBURIDE', 'TOLBUTAMIDE', 'PIOGLITAZONE',
                         'ROSIGLITAZONE', 'ACARBOSE', 'MIGLITOL', 'TROGLITAZONE', 'TOLAZAMIDE', 'EXAMIDE',
                         'CITOGLIPTON', 'INSULIN', 'GLYBURIDE-METFORMIN', 'GLIPIZIDE-METFORMIN',
                         'GLIMEPIRIDE-PIOGLITAZONE', 'METFORMIN-ROSIGLITAZONE', 'METFORMIN-PIOGLITAZONE']
    numerical_variable = ['AGE', 'TIME_IN_HOSPITAL', 'NUM_LAB_PROCEDURES', 'NUM_PROCEDURES',
                          'NUM_MEDICATIONS', 'NUMBER_OUTPATIENT', 'NUMBER_EMERGENCY', 'NUMBER_INPATIENT',
                          'NUMBER_DIAGNOSES', 'CHANGE', 'DIABETESMED']
    # 处理数值变量
    numerical_data = df()
    numerical_data['AGE'] = origin_data['AGE'] \
        .replace(['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)',
                  '[60-70)', '[70-80)', '[80-90)', '[90-100)'],
                 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    numerical_data['TIME_IN_HOSPITAL'] = origin_data['TIME_IN_HOSPITAL']
    numerical_data['NUM_LAB_PROCEDURES'] = origin_data['NUM_LAB_PROCEDURES']
    numerical_data['NUM_PROCEDURES'] = origin_data['NUM_PROCEDURES']
    numerical_data['NUM_MEDICATIONS'] = origin_data['NUM_MEDICATIONS']
    numerical_data['NUMBER_OUTPATIENT'] = origin_data['NUMBER_OUTPATIENT']
    numerical_data['NUMBER_EMERGENCY'] = origin_data['NUMBER_EMERGENCY']
    numerical_data['NUMBER_INPATIENT'] = origin_data['NUMBER_INPATIENT']
    numerical_data['NUMBER_DIAGNOSES'] = origin_data['NUMBER_DIAGNOSES']
    numerical_data['CHANGE'] = origin_data['CHANGE'].replace(['No', 'Ch'], [0, 1])
    numerical_data['DIABETESMED'] = origin_data['DIABETESMED'].replace(['No', 'Yes'], [0, 1])

    # 处理类别变量
    category_data = df()
    for i in category_variable:
        count = 0
        number_dict = {}
        for j in origin_data[i]:
            if j not in number_dict:
                number_dict[j] = count
                count += 1
        category_data[i] = origin_data[i].replace(number_dict.keys(), number_dict.values())

    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit_transform(numerical_data)
    save_path = './generate_data/numerical_data.csv'
    numerical_data.to_csv(save_path, index=False)

    min_max_scaler.fit_transform(category_data)
    save_path = './generate_data/category_data.csv'
    category_data.to_csv(save_path, index=False)

    save_path = './generate_data/diabetic_label.csv'
    diabetic_label.to_csv(save_path, index=True)


def common_process():
    origin_data = pd.read_csv('./origin_data/DIABETIC_DATA.csv')
    common_data = df()
    common_data['RACE'] = origin_data['RACE']
    common_data['GENDER'] = origin_data['GENDER'].replace(['Female', 'Male'], [0, 1])
    common_data['AGE'] = origin_data['AGE'] \
        .replace(['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)',
                  '[60-70)', '[70-80)', '[80-90)', '[90-100)'],
                 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    common_data['WEIGHT'] = origin_data['WEIGHT'].replace(['[0-25)', '[25-50)', '[50-75)', ''], [])
    common_data['ADMISSION_TYPE_ID'] = origin_data['ADMISSION_TYPE_ID']
    common_data['DISCHARGE_DISPOSITION_ID'] = origin_data['DISCHARGE_DISPOSITION_ID']
    common_data['ADMISSION_SOURCE_ID'] = origin_data['ADMISSION_SOURCE_ID']
    common_data['TIME_IN_HOSPITAL'] = origin_data['TIME_IN_HOSPITAL']
    common_data['NUM_LAB_PROCEDURES'] = origin_data['NUM_LAB_PROCEDURES']
    common_data['NUM_PROCEDURES'] = origin_data['NUM_PROCEDURES']
    common_data['NUM_MEDICATIONS'] = origin_data['NUM_MEDICATIONS']
    common_data['NUMBER_OUTPATIENT'] = origin_data['NUMBER_OUTPATIENT']
    common_data['NUMBER_EMERGENCY'] = origin_data['NUMBER_EMERGENCY']
    common_data['NUMBER_INPATIENT'] = origin_data['NUMBER_INPATIENT']
    common_data['DIAG_1'] = origin_data['DIAG_1']
    common_data['DIAG_2'] = origin_data['DIAG_2']
    common_data['DIAG_3'] = origin_data['DIAG_3']
    common_data['NUMBER_DIAGNOSES'] = origin_data['NUMBER_DIAGNOSES']
    common_data['MAX_GLU_SERUM'] = origin_data['MAX_GLU_SERUM'] \
        .replace(['None', 'Norm', '>200', '>300'], [0, 1, 2, 3])

    # common_data['RACE'] = origin_data['RACE'].replace(['NO', '<30', '>30'], [0, 1, 0])
    # common_data['RACE'] = origin_data['RACE'].replace(['NO', '<30', '>30'], [0, 1, 0])
    # common_data['RACE'] = origin_data['RACE'].replace(['NO', '<30', '>30'], [0, 1, 0])
    # common_data['RACE'] = origin_data['RACE'].replace(['NO', '<30', '>30'], [0, 1, 0])
    # common_data['RACE'] = origin_data['RACE'].replace(['NO', '<30', '>30'], [0, 1, 0])
    # common_data['RACE'] = origin_data['RACE'].replace(['NO', '<30', '>30'], [0, 1, 0])
    # common_data['RACE'] = origin_data['RACE'].replace(['NO', '<30', '>30'], [0, 1, 0])
    # common_data['RACE'] = origin_data['RACE'].replace(['NO', '<30', '>30'], [0, 1, 0])
    # common_data['RACE'] = origin_data['RACE'].replace(['NO', '<30', '>30'], [0, 1, 0])

    common_data['READMITTED'] = origin_data['READMITTED'].replace(['NO', '<30', '>30'], [0, 1, 0])
    save_path = './generate_data/common_data.csv'
    common_data.to_csv(save_path, index=False)


def word2number():
    origin_data = pd.read_csv('./origin_data/DIABETIC_DATA.csv')
    data_length = len(origin_data)
    origin_data['READMITTED'] = origin_data['READMITTED'].replace(['NO', '<30', '>30'], [0, 1, 0])
    origin_data = origin_data.drop('ENCOUNTER_ID', 1)
    origin_data = origin_data.drop('PATIENT_NBR', 1)
    for i in origin_data:
        count = 0
        number_dict = {}
        for j in origin_data[i]:
            if j not in number_dict:
                number_dict[j] = count
                count += 1
        origin_data[i] = origin_data[i].replace(number_dict.keys(), number_dict.values())
    train_y = origin_data['READMITTED'][0:int(data_length * 0.8)]
    test_y = origin_data['READMITTED'][int(data_length * 0.8):]
    origin_data = origin_data.drop('READMITTED', 1)
    train_x = origin_data[0:int(data_length * 0.8)]
    test_x = origin_data[int(data_length * 0.8):]
    return [train_x, train_y, test_x, test_y]


if __name__ == "__main__":
    start = time.clock()
    data_neural_process()
    print(time.clock() - start)
