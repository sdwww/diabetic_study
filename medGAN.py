import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.layers import l2_regularizer
from matplotlib import pyplot as plt

_VALIDATION_RATIO = 0.1


class Medgan(object):
    def __init__(self,
                 data_type='binary',
                 input_dim=100,
                 embedding_dim=128,
                 random_dim=128,
                 generator_dims=(128, 128),
                 discriminator_dims=(256, 128, 1),
                 compress_dims=(),
                 decompress_dims=(),
                 bn_decay=0.99,
                 l2_scale=0.001):
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.generator_dims = list(generator_dims) + [embedding_dim]
        self.random_dim = random_dim
        self.data_type = data_type

        if data_type == 'binary':
            self.ae_activation = tf.nn.tanh
        else:
            self.ae_activation = tf.nn.relu

        self.generator_activation = tf.nn.relu
        self.discriminator_activation = tf.nn.relu
        self.discriminator_dims = discriminator_dims
        self.compress_dims = list(compress_dims) + [embedding_dim]
        self.decompress_dims = list(decompress_dims) + [input_dim]
        self.bn_decay = bn_decay
        self.l2_scale = l2_scale

    def load_data(self, data_path):
        data = np.load(data_path)['arr_0']

        if self.data_type == 'binary':
            data = np.clip(data, 0, 1)

        trainX, validX = train_test_split(data, test_size=_VALIDATION_RATIO, random_state=0)
        return trainX, validX

    def build_autoencoder(self, x_input):
        decode_variables = {}
        with tf.variable_scope('autoencoder', regularizer=l2_regularizer(self.l2_scale)):
            temp_vec = x_input
            temp_dim = self.input_dim
            i = 0
            for compress_dim in self.compress_dims:
                W = tf.get_variable('aee_W_' + str(i), shape=[temp_dim, compress_dim])
                b = tf.get_variable('aee_b_' + str(i), shape=[compress_dim])
                temp_vec = self.ae_activation(tf.add(tf.matmul(temp_vec, W), b))
                temp_dim = compress_dim
                i += 1

            i = 0
            for decompressDim in self.decompress_dims[:-1]:
                W = tf.get_variable('aed_W_' + str(i), shape=[temp_dim, decompressDim])
                b = tf.get_variable('aed_b_' + str(i), shape=[decompressDim])
                temp_vec = self.ae_activation(tf.add(tf.matmul(temp_vec, W), b))
                temp_dim = decompressDim
                decode_variables['aed_W_' + str(i)] = W
                decode_variables['aed_b_' + str(i)] = b
                i += 1
            W = tf.get_variable('aed_W_' + str(i), shape=[temp_dim, self.decompress_dims[-1]])
            b = tf.get_variable('aed_b_' + str(i), shape=[self.decompress_dims[-1]])
            decode_variables['aed_W_' + str(i)] = W
            decode_variables['aed_b_' + str(i)] = b

            if self.data_type == 'binary':
                x_reconst = tf.nn.sigmoid(tf.add(tf.matmul(temp_vec, W), b))
                loss = tf.reduce_mean(-tf.reduce_sum(
                    x_input * tf.log(x_reconst + 1e-12) + (1. - x_input) * tf.log(1. - x_reconst + 1e-12), 1), 0)
            else:
                x_reconst = tf.nn.relu(tf.add(tf.matmul(temp_vec, W), b))
                loss = tf.reduce_mean((x_input - x_reconst) ** 2)
        return loss, decode_variables

    def build_generator(self, x_input, bn_train):
        temp_vec = x_input
        temp_dim = self.random_dim
        with tf.variable_scope('generator', regularizer=l2_regularizer(self.l2_scale)):
            for i, gen_dim in enumerate(self.generator_dims[:-1]):
                W = tf.get_variable('W_' + str(i), shape=[temp_dim, gen_dim])
                h = tf.matmul(temp_vec, W)
                h2 = batch_norm(h, decay=self.bn_decay, scale=True, is_training=bn_train, updates_collections=None)
                h3 = self.generator_activation(h2)
                temp_vec = h3 + temp_vec
                temp_dim = gen_dim
            W = tf.get_variable('W' + str(i), shape=[temp_dim, self.generator_dims[-1]])
            h = tf.matmul(temp_vec, W)
            h2 = batch_norm(h, decay=self.bn_decay, scale=True, is_training=bn_train, updates_collections=None)

            if self.data_type == 'binary':
                h3 = tf.nn.tanh(h2)
            else:
                h3 = tf.nn.relu(h2)

            output = h3 + temp_vec
        return output

    def build_generator_test(self, x_input, bn_train):
        temp_vec = x_input
        temp_dim = self.random_dim
        with tf.variable_scope('generator', regularizer=l2_regularizer(self.l2_scale)):
            for i, genDim in enumerate(self.generator_dims[:-1]):
                W = tf.get_variable('W_' + str(i), shape=[temp_dim, genDim])
                h = tf.matmul(temp_vec, W)
                h2 = batch_norm(h, decay=self.bn_decay, scale=True, is_training=bn_train, updates_collections=None,
                                trainable=False)
                h3 = self.generator_activation(h2)
                temp_vec = h3 + temp_vec
                temp_dim = genDim
            W = tf.get_variable('W' + str(i), shape=[temp_dim, self.generator_dims[-1]])
            h = tf.matmul(temp_vec, W)
            h2 = batch_norm(h, decay=self.bn_decay, scale=True, is_training=bn_train, updates_collections=None,
                            trainable=False)

            if self.data_type == 'binary':
                h3 = tf.nn.tanh(h2)
            else:
                h3 = tf.nn.relu(h2)

            output = h3 + temp_vec
        return output

    def get_discriminator_results(self, x_input, keep_rate, reuse=False):
        batch_size = tf.shape(x_input)[0]
        input_mean = tf.reshape(tf.tile(
            tf.reduce_mean(x_input, 0), [batch_size]), (batch_size, self.input_dim))
        temp_vec = tf.concat([x_input, input_mean], axis=1)
        temp_dim = self.input_dim * 2
        with tf.variable_scope('discriminator', reuse=reuse, regularizer=l2_regularizer(self.l2_scale)):
            for i, disc_dim in enumerate(self.discriminator_dims[:-1]):
                W = tf.get_variable('W_' + str(i), shape=[temp_dim, disc_dim])
                b = tf.get_variable('b_' + str(i), shape=[disc_dim])
                h = self.discriminator_activation(tf.add(tf.matmul(temp_vec, W), b))
                h = tf.nn.dropout(h, keep_rate)
                temp_vec = h
                temp_dim = disc_dim
            W = tf.get_variable('W', shape=[temp_dim, 1])
            b = tf.get_variable('b', shape=[1])
            y_hat = tf.squeeze(tf.nn.sigmoid(tf.add(tf.matmul(temp_vec, W), b)))
        return y_hat

    def build_discriminator(self, x_real, x_fake, keepRate, decodeVariables, bn_train):
        # Discriminate for real samples
        y_hat_real = self.get_discriminator_results(x_real, keepRate, reuse=False)
        # Decompress, then discriminate for real samples
        temp_vec = x_fake
        i = 0
        for _ in self.decompress_dims[:-1]:
            temp_vec = self.ae_activation(
                tf.add(tf.matmul(temp_vec, decodeVariables['aed_W_' + str(i)]), decodeVariables['aed_b_' + str(i)]))
            i += 1

        if self.data_type == 'binary':
            x_decoded = tf.nn.sigmoid(
                tf.add(tf.matmul(temp_vec, decodeVariables['aed_W_' + str(i)]), decodeVariables['aed_b_' + str(i)]))
        else:
            x_decoded = tf.nn.relu(
                tf.add(tf.matmul(temp_vec, decodeVariables['aed_W_' + str(i)]), decodeVariables['aed_b_' + str(i)]))

        y_hat_fake = self.get_discriminator_results(x_decoded, keepRate, reuse=True)

        loss_d = -tf.reduce_mean(tf.log(y_hat_real + 1e-12)) - tf.reduce_mean(tf.log(1. - y_hat_fake + 1e-12))
        loss_g = -tf.reduce_mean(tf.log(y_hat_fake + 1e-12))

        return loss_d, loss_g, y_hat_real, y_hat_fake

    def print2file(self, buf, out_file):
        outfd = open(out_file, 'a')
        outfd.write(buf + '\n')
        outfd.close()

    def generate_data(self,
                      n_samples=100,
                      model_file='model',
                      batch_size=100,
                      out_file='out'):
        x_dummy = tf.placeholder('float', [None, self.input_dim])
        _, decode_variables = self.build_autoencoder(x_dummy)
        x_random = tf.placeholder('float', [None, self.random_dim])
        bn_train = tf.placeholder('bool')
        x_emb = self.build_generator_test(x_random, bn_train)
        temp_vec = x_emb
        i = 0
        for _ in self.decompress_dims[:-1]:
            temp_vec = self.ae_activation(
                tf.add(tf.matmul(temp_vec, decode_variables['aed_W_' + str(i)]), decode_variables['aed_b_' + str(i)]))
            i += 1

        if self.data_type == 'binary':
            x_reconst = tf.nn.sigmoid(
                tf.add(tf.matmul(temp_vec, decode_variables['aed_W_' + str(i)]), decode_variables['aed_b_' + str(i)]))
        else:
            x_reconst = tf.nn.relu(
                tf.add(tf.matmul(temp_vec, decode_variables['aed_W_' + str(i)]), decode_variables['aed_b_' + str(i)]))

        np.random.seed(1234)
        saver = tf.train.Saver()
        output_vec = []
        burn_in = 1000
        with tf.Session() as sess:
            saver.restore(sess, model_file)
            print('burning in')
            for i in range(burn_in):
                random_x = np.random.normal(size=(batch_size, self.random_dim))
                output = sess.run(x_reconst, feed_dict={x_random: random_x, bn_train: True})

            print('generating')
            n_batches = int(np.ceil(float(n_samples)) / float(batch_size))
            for i in range(n_batches):
                random_x = np.random.normal(size=(batch_size, self.random_dim))
                output = sess.run(x_reconst, feed_dict={x_random: random_x, bn_train: False})
                output_vec.extend(output)

        output_mat = np.array(output_vec)
        np.save(out_file, output_mat)
        output_mat = output_mat // 0.5
        output_mat = np.transpose(output_mat)
        result=[]
        for i in output_mat:
            result.append(np.sum(i) / 5000)
        plt.scatter(np.arange(20,70),result , s=25, alpha=0.4, marker='o')
        plt.show()

    def calculate_disc_auc(self, preds_real, preds_fake):
        preds = np.concatenate([preds_real, preds_fake], axis=0)
        labels = np.concatenate([np.ones((len(preds_real))), np.zeros((len(preds_fake)))], axis=0)
        auc = roc_auc_score(labels, preds)
        return auc

    def calculate_disc_accuracy(self, preds_real, preds_fake):
        total = len(preds_real) + len(preds_fake)
        hit = 0
        for pred in preds_real:
            if pred > 0.5: hit += 1
        for pred in preds_fake:
            if pred < 0.5: hit += 1
        acc = float(hit) / float(total)
        return acc

    def train(self,
              data_path='',
              model_path='',
              out_path='',
              n_epochs=500,
              discriminator_train_period=2,
              generator_train_period=1,
              pretrain_batch_size=100,
              batch_size=100,
              pretrain_epochs=100,
              save_max_keep=0):
        x_raw = tf.placeholder('float', [None, self.input_dim])
        x_random = tf.placeholder('float', [None, self.random_dim])
        keep_prob = tf.placeholder('float')
        bn_train = tf.placeholder('bool')

        loss_ae, decode_variables = self.build_autoencoder(x_raw)
        x_fake = self.build_generator(x_random, bn_train)
        loss_d, loss_g, y_hat_real, y_hat_fake = self.build_discriminator(x_raw, x_fake, keep_prob, decode_variables,
                                                                          bn_train)
        train_x, valid_x = self.load_data(data_path)

        t_vars = tf.trainable_variables()
        ae_vars = [var for var in t_vars if 'autoencoder' in var.name]
        d_vars = [var for var in t_vars if 'discriminator' in var.name]
        g_vars = [var for var in t_vars if 'generator' in var.name]

        all_regs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        optimize_ae = tf.train.AdamOptimizer().minimize(loss_ae + sum(all_regs), var_list=ae_vars)
        optimize_d = tf.train.AdamOptimizer().minimize(loss_d + sum(all_regs), var_list=d_vars)
        optimize_g = tf.train.AdamOptimizer().minimize(loss_g + sum(all_regs),
                                                       var_list=g_vars.append(decode_variables.values()))

        init_op = tf.global_variables_initializer()

        n_batches = int(np.ceil(float(train_x.shape[0]) / float(batch_size)))
        saver = tf.train.Saver(max_to_keep=save_max_keep)
        log_file = out_path + '.log'

        with tf.Session() as sess:
            # create a log writer. run 'tensorboard --logdir=./logs'
            writer = tf.summary.FileWriter("./logs", sess.graph)  # for 1.0
            if model_path == '':
                sess.run(init_op)
            else:
                saver.restore(sess, model_path)
            n_train_batches = int(np.ceil(float(train_x.shape[0])) / float(pretrain_batch_size))
            n_valid_batches = int(np.ceil(float(valid_x.shape[0])) / float(pretrain_batch_size))

            # 训练自动编码器
            if model_path == '':
                for epoch in range(pretrain_epochs):
                    # 乱序排列
                    idx = np.random.permutation(train_x.shape[0])
                    train_loss_vec = []
                    for i in range(n_train_batches):
                        batch_x = train_x[idx[i * pretrain_batch_size:(i + 1) * pretrain_batch_size]]
                        _, loss = sess.run([optimize_ae, loss_ae], feed_dict={x_raw: batch_x})
                        train_loss_vec.append(loss)
                    idx = np.random.permutation(valid_x.shape[0])
                    valid_loss_vec = []
                    for i in range(n_valid_batches):
                        batch_x = valid_x[idx[i * pretrain_batch_size:(i + 1) * pretrain_batch_size]]
                        loss = sess.run(loss_ae, feed_dict={x_raw: batch_x})
                        valid_loss_vec.append(loss)
                    buf = 'Pretrain_Epoch:%d, trainLoss:%f, validLoss:%f ' \
                          % (epoch, np.mean(train_loss_vec), np.mean(valid_loss_vec))
                    print(buf)
                    self.print2file(buf, log_file)

            # 训练生成器和判别器
            idx = np.arange(train_x.shape[0])
            for epoch in range(n_epochs):
                d_loss_vec = []
                g_loss_vec = []
                for i in range(n_batches):
                    for _ in range(discriminator_train_period):
                        batch_idx = np.random.choice(idx, size=batch_size, replace=False)
                        batch_x = train_x[batch_idx]
                        random_x = np.random.normal(size=(batch_size, self.random_dim))
                        _, disc_loss = sess.run([optimize_d, loss_d],
                                                feed_dict={x_raw: batch_x, x_random: random_x, keep_prob: 1.0,
                                                           bn_train: False})
                        d_loss_vec.append(disc_loss)
                    for _ in range(generator_train_period):
                        random_x = np.random.normal(size=(batch_size, self.random_dim))
                        _, generator_loss = sess.run([optimize_g, loss_g],
                                                     feed_dict={x_random: random_x, keep_prob: 1.0,
                                                                bn_train: True})
                        g_loss_vec.append(generator_loss)

                # 验证集进行判别器验证
                idx = np.arange(len(valid_x))
                n_valid_batches = int(np.ceil(float(len(valid_x)) / float(batch_size)))
                valid_acc_vec = []
                valid_auc_vec = []
                for i in range(n_valid_batches):
                    batch_idx = np.random.choice(idx, size=batch_size, replace=False)
                    batch_x = valid_x[batch_idx]
                    random_x = np.random.normal(size=(batch_size, self.random_dim))
                    preds_real, preds_fake, = sess.run([y_hat_real, y_hat_fake],
                                                       feed_dict={x_raw: batch_x, x_random: random_x, keep_prob: 1.0,
                                                                  bn_train: False})
                    valid_acc = self.calculate_disc_accuracy(preds_real, preds_fake)
                    valid_auc = self.calculate_disc_auc(preds_real, preds_fake)
                    valid_acc_vec.append(valid_acc)
                    valid_auc_vec.append(valid_auc)
                buf = 'Epoch:%d, d_loss:%f, g_loss:%f, accuracy:%f, AUC:%f' % (
                    epoch, np.mean(d_loss_vec), np.mean(g_loss_vec), np.mean(valid_acc_vec), np.mean(valid_auc_vec))
                print(buf)
                self.print2file(buf, log_file)
                save_path = saver.save(sess, out_path, global_step=epoch)
        print(save_path)


def get_config():
    model_config = dict()
    # The dimension size of the embedding, which will be generated by the generator.
    model_config['embed_size'] = 128
    # The dimension size of the random noise, on which the generator is conditioned.
    model_config['noise_size'] = 128
    # The dimension size of the generator. Note that another layer of size "--embed_size" is always added.
    model_config['generator_size'] = (128, 128)
    # The dimension size of the discriminator.
    model_config['discriminator_size'] = (256, 128, 1)
    # The dimension size of the encoder of the autoencoder.
    # Note that another layer of size "embed_size" is always added. Therefore this can be a blank tuple.
    model_config['compressor_size'] = ()
    # The dimension size of the decoder of the autoencoder.
    # Note that another layer, whose size is equal to the dimension of the <patient_matrix>,
    # is always added. Therefore this can be a blank tuple.
    model_config['decompressor_size'] = ()
    # The input data type. The <patient matrix> could either contain binary values or count values.
    model_config['data_type'] = 'binary'
    # Decay value for the moving average used in Batch Normalization.
    model_config['batchnorm_decay'] = 0.99
    # L2 regularization coefficient for all weights.
    model_config['L2'] = 0.001
    # The path to the numpy matrix containing aggregated patient records.
    model_config['data_file'] = './random_data.npz'
    # The path to the output models.
    model_config['out_file'] = './medGAN_result/result'
    # The path to the model file, in case you want to continue training.
    model_config['model_file'] = ''
    # The number of epochs to pre-train the autoencoder.
    model_config['n_pretrain_epoch'] = 100
    # The number of epochs to train medGAN.
    model_config['n_epoch'] = 200
    # The number of times to update the discriminator per epoch
    model_config['n_discriminator_update'] = 2
    # The number of times to update the generator per epoch.
    model_config['n_generator_update'] = 1
    # The size of a single mini-batch for pre-training the autoencoder.
    model_config['pretrain_batch_size'] = 10
    # The size of a single mini-batch for training medGAN.
    model_config['batch_size'] = 100
    # The number of models to keep. Setting this to 0 will save models for every epoch.
    model_config['save_max_keep'] = 10
    return model_config


if __name__ == '__main__':
    config = get_config()
    simulated_data = np.zeros([5000, 50])
    simulated_data = np.transpose(simulated_data)
    rate = 1
    for i in simulated_data:
        idx = np.random.choice(50, rate)
        rate += 1
        i[idx] = 1
    simulated_data = np.transpose(simulated_data)
    np.savez(config['data_file'], simulated_data)
    data = np.load(config['data_file'])['arr_0']
    inputDim = data.shape[1]

    mg = Medgan(data_type=config['data_type'],
                input_dim=inputDim,
                embedding_dim=config['embed_size'],
                random_dim=config['noise_size'],
                generator_dims=config['generator_size'],
                discriminator_dims=config['discriminator_size'],
                compress_dims=config['compressor_size'],
                decompress_dims=config['decompressor_size'],
                bn_decay=config['batchnorm_decay'],
                l2_scale=config['L2'])

    mg.train(data_path=config['data_file'],
             model_path=config['model_file'],
             out_path=config['out_file'],
             pretrain_epochs=config['n_pretrain_epoch'],
             n_epochs=config['n_epoch'],
             discriminator_train_period=config['n_discriminator_update'],
             generator_train_period=config['n_generator_update'],
             pretrain_batch_size=config['pretrain_batch_size'],
             batch_size=config['batch_size'],
             save_max_keep=config['save_max_keep'])
    #mg.generate_data(n_samples=5000, model_file='./medGAN_result/result-199')
