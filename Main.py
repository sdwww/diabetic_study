import numpy as np
import pandas as pd
import tensorflow as tf
import time

from DiabeticModel import DiabeticModel


def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]


def get_config():
    config = dict()
    config['n_input'] = 2443
    config['n_hidden_1'] = 500
    config['n_hidden_2'] = 200
    config['n_output'] = 1
    config['max_epoch'] = 10
    config['n_sample'] = 101766
    config['init_scale'] = 0.01
    config['batch_size'] = 128
    config['display_step'] = 1
    return config


def load_data(config):
    one_hot_data = pd.read_csv('./generate_data/one_hot_data.csv')
    x_train, x_test = one_hot_data[0:int(config['n_sample'] * 0.8)], one_hot_data[int(config['n_sample'] * 0.8):]
    origin_data = pd.read_csv('./origin_data/DIABETIC_DATA.csv')

    diabetic_label = origin_data['READMITTED'].replace(['NO', '<30', '>30'], [0, 1, 1]).as_matrix() \
        .reshape((config['n_sample'], 1))
    y_train, y_test = diabetic_label[0:int(config['n_sample'] * 0.8)], diabetic_label[int(config['n_sample'] * 0.8):]
    return x_train, x_test, y_train, y_test


def model_pre_train(diabetic_model, config):
    x_train, x_test, _, _ = load_data(config=config)
    for epoch in range(config['max_epoch']):
        avg_cost = 0.
        total_batch = int(config['n_sample'] / config['batch_size'])
        for i in range(total_batch):
            batch_x = x_train[i * config['batch_size']:(i + 1) * config['batch_size']]
            cost = diabetic_model.pre_train(batch_x)
            avg_cost += cost / config['n_sample'] * config['batch_size']

        if epoch % config['display_step'] == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

    print("Total cost: " + str(diabetic_model.calc_encoder_cost(x_test)))


def model_fine_tune(diabetic_model, config):
    x_train, x_test, y_train, y_test = load_data(config=config)
    for epoch in range(config['max_epoch']):
        avg_cost = 0.
        total_batch = int(config['n_sample'] / config['batch_size'])
        for i in range(total_batch):
            batch_x = x_train[i * config['batch_size']:(i + 1) * config['batch_size']]
            batch_y = y_train[i * config['batch_size']:(i + 1) * config['batch_size']]
            cost = diabetic_model.fine_tune(batch_x, batch_y)
            avg_cost += cost / config['n_sample'] * config['batch_size']

        if epoch % config['display_step'] == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

    print("结果为" + str(diabetic_model.show_predict(x_test, y_test)))


def main(_):
    config = get_config()
    diabetic_model = DiabeticModel(n_input=config['n_input'], n_hidden_1=config['n_hidden_1'],
                                   n_hidden_2=config['n_hidden_2'], n_output=config['n_output'],
                                   init_scale=config['init_scale'])
    #model_pre_train(diabetic_model, config)
    model_fine_tune(diabetic_model, config)


if __name__ == '__main__':
    start = time.clock()
    tf.app.run()
    print('运行时间：',time.clock() - start)
