import time

import numpy as np
import pandas as pd
import tensorflow as tf
from pandas import Series

from DiabeticModel import DiabeticModel


def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]


def get_config():
    config = dict()
    config['n_numerical_input'] = 11
    config['n_category_input'] = ['RACE', 'GENDER', 'WEIGHT', 'ADMISSION_TYPE_ID', 'DISCHARGE_DISPOSITION_ID',
                                  'ADMISSION_SOURCE_ID', 'PAYER_CODE', 'MEDICAL_SPECIALTY', 'DIAG_1',
                                  'DIAG_2', 'DIAG_3', 'MAX_GLU_SERUM', 'A1CRESULT', 'METFORMIN',
                                  'REPAGLINIDE', 'NATEGLINIDE', 'CHLORPROPAMIDE', 'GLIMEPIRIDE',
                                  'ACETOHEXAMIDE', 'GLIPIZIDE', 'GLYBURIDE', 'TOLBUTAMIDE',
                                  'PIOGLITAZONE', 'ROSIGLITAZONE', 'ACARBOSE', 'MIGLITOL', 'TROGLITAZONE',
                                  'TOLAZAMIDE', 'EXAMIDE', 'CITOGLIPTON', 'INSULIN', 'GLYBURIDE-METFORMIN',
                                  'GLIPIZIDE-METFORMIN', 'GLIMEPIRIDE-PIOGLITAZONE',
                                  'METFORMIN-ROSIGLITAZONE', 'METFORMIN-PIOGLITAZONE']
    config['n_embed'] = 5
    config['n_concat'] = 191
    config['n_hidden_1'] = 100
    config['n_hidden_2'] = 50
    config['n_output'] = 1
    config['pre_train_epoch'] = 10
    config['fine_tune_epoch'] = 20
    config['n_sample'] = 101766
    config['init_scale'] = 0.1
    config['batch_size'] = 128
    config['display_step'] = 1
    return config


def load_data(config):
    numerical_data = pd.read_csv('./generate_data/numerical_data.csv')
    x_numerical_train, x_numerical_test = numerical_data[0:int(config['n_sample'] * 0.8)], \
                                          numerical_data[int(config['n_sample'] * 0.8):]
    category_data = pd.read_csv('./generate_data/category_data.csv')
    x_cate_train, x_cate_test = category_data[0:int(config['n_sample'] * 0.8)], \
                                category_data[int(config['n_sample'] * 0.8):]
    x_category_train = {}
    x_category_test = {}
    for i in range(len(config['n_category_input'])):
        x_category_train[config['n_category_input'][i]] = x_cate_train.iloc[:, i].as_matrix()
        x_category_test[config['n_category_input'][i]] = x_cate_test.iloc[:, i].as_matrix()
    diabetic_label = Series.from_csv('./generate_data/diabetic_label.csv').as_matrix().reshape(config['n_sample'],1)
    y_train, y_test = diabetic_label[0:int(config['n_sample'] * 0.8)], \
                      diabetic_label[int(config['n_sample'] * 0.8):]
    return x_numerical_train, x_numerical_test, x_category_train, x_category_test, y_train, y_test


def model_pre_train(diabetic_model, config):
    x_numerical_train, x_numerical_test, x_category_train, x_category_test, y_train, y_test = load_data(config=config)
    for epoch in range(config['pre_train_epoch']):
        avg_cost = 0.
        total_batch = int(config['n_sample'] / config['batch_size'])
        for i in range(total_batch):
            batch_numerical_x = x_numerical_train[i * config['batch_size']:(i + 1) * config['batch_size']]
            batch_category_x = {}
            for name, item in x_category_train.items():
                batch_category_x[name] = item[i * config['batch_size']:(i + 1) * config['batch_size']]
            cost = diabetic_model.pre_train(batch_numerical_x,batch_category_x)
            avg_cost += cost / config['n_sample'] * config['batch_size']

        if epoch % config['display_step'] == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))


def model_fine_tune(diabetic_model, config):
    x_numerical_train, x_numerical_test, x_category_train, x_category_test, y_train, y_test = load_data(config=config)
    for epoch in range(config['fine_tune_epoch']):
        avg_cost = 0.
        total_batch = int(config['n_sample'] / config['batch_size'])
        for i in range(total_batch):
            batch_numerical_x = x_numerical_train[i * config['batch_size']:(i + 1) * config['batch_size']]
            batch_category_x = {}
            for name, item in x_category_train.items():
                batch_category_x[name] = item[i * config['batch_size']:(i + 1) * config['batch_size']]
            batch_y = y_train[i * config['batch_size']:(i + 1) * config['batch_size']]
            cost = diabetic_model.fine_tune(batch_numerical_x, batch_category_x, batch_y)
            avg_cost += cost / config['n_sample'] * config['batch_size']

        if epoch % config['display_step'] == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

    print("结果为" + str(diabetic_model.show_predict(x_numerical_test, x_category_test, y_test)))


def main(_):
    config = get_config()
    diabetic_model = DiabeticModel(n_numerical_input=config['n_numerical_input'],
                                   n_category_input=config['n_category_input'], n_embed=config['n_embed'],
                                   n_concat=config['n_concat'], n_hidden_1=config['n_hidden_1'],
                                   n_hidden_2=config['n_hidden_2'],
                                   n_output=config['n_output'], init_scale=config['init_scale'])
    model_pre_train(diabetic_model, config)
    model_fine_tune(diabetic_model, config)


if __name__ == '__main__':
    start = time.clock()
    tf.app.run()
    print('运行时间：', time.clock() - start)
