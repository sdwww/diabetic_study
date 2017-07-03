import tensorflow as tf
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score


class DiabeticModel(object):
    def __init__(self, n_numerical_input, n_category_input, n_embed, n_concat, n_hidden_1, n_hidden_2, n_output,
                 init_scale,
                 optimizer=tf.train.AdadeltaOptimizer(0.05)):
        self.n_numerical_input = n_numerical_input
        self.n_category_input = n_category_input
        self.n_embed = n_embed
        self.n_concat = n_concat
        self.n_hidden_1 = n_hidden_1
        self.n_hidden_2 = n_hidden_2
        self.n_output = n_output
        self.init_scale = init_scale

        self.weights = self.__initialize_weights()

        # model
        self.numerical_x = tf.placeholder(tf.float32, [None, self.n_numerical_input])
        self.category_x = self.__initialize_category_x()
        self.y = tf.placeholder(tf.float32, [None, self.n_output])
        self.embed = {}
        for name in self.n_category_input:
            self.embed[name] = tf.cast(tf.nn.embedding_lookup(self.weights[name + '_embed'], self.category_x[name]),
                                       tf.float32)
        embed_list = []
        for i in self.embed.values():
            embed_list.append(i)
        embed_list.append(self.numerical_x)
        self.concat = tf.concat(values=embed_list, axis=1)
        self.hidden_1 = tf.nn.relu(tf.add(tf.matmul(self.concat, self.weights['w1']), self.weights['b1']))
        self.reconstruction = tf.nn.relu(
            tf.add(tf.matmul(self.hidden_1, self.weights['w_recon']), self.weights['b_recon']))
        self.hidden_2 = tf.nn.relu(tf.add(tf.matmul(self.hidden_1, self.weights['w2']), self.weights['b2']))
        self.y_ = tf.nn.sigmoid(tf.add(tf.matmul(self.hidden_2, self.weights['w_output']), self.weights['b_output']))

        # cost
        self.encoder_cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.concat), 2.0))
        self.encoder_optimizer = optimizer.minimize(self.encoder_cost)
        self.class_cost = tf.reduce_sum(tf.pow(tf.subtract(self.y, self.y_), 2.0))
        self.class_optimizer = optimizer.minimize(self.class_cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def __initialize_weights(self):
        all_weights = dict()
        for name in self.n_category_input:
            all_weights[name + "_embed"] = tf.Variable(
                tf.random_normal([800, self.n_embed], stddev=self.init_scale, dtype=tf.float32))
        all_weights['w1'] = tf.Variable(
            tf.random_normal([self.n_concat, self.n_hidden_1], stddev=self.init_scale, dtype=tf.float32))
        all_weights['b1'] = tf.Variable(
            tf.random_normal([self.n_hidden_1], stddev=self.init_scale, dtype=tf.float32))
        all_weights['w_recon'] = tf.Variable(
            tf.random_normal([self.n_hidden_1, self.n_concat], stddev=self.init_scale, dtype=tf.float32))
        all_weights['b_recon'] = tf.Variable(
            tf.random_normal([self.n_concat], stddev=self.init_scale, dtype=tf.float32))
        all_weights['w2'] = tf.Variable(
            tf.random_normal([self.n_hidden_1, self.n_hidden_2], stddev=self.init_scale, dtype=tf.float32))
        all_weights['b2'] = tf.Variable(
            tf.random_normal([self.n_hidden_2], stddev=self.init_scale, dtype=tf.float32))
        all_weights['w_output'] = tf.Variable(
            tf.random_normal([self.n_hidden_2, self.n_output], stddev=self.init_scale, dtype=tf.float32))
        all_weights['b_output'] = tf.Variable(
            tf.random_normal([self.n_output], stddev=self.init_scale, dtype=tf.float32))
        return all_weights

    def __initialize_category_x(self):
        category_x = {}
        for name in self.n_category_input:
            category_x[name] = tf.placeholder(tf.int32, [None])
        return category_x

    def pre_train(self, numerical_x, category_x):
        cost, opt = self.sess.run((self.encoder_cost, self.encoder_optimizer),
                                  feed_dict={self.numerical_x: numerical_x,
                           self.category_x[self.n_category_input[0]]: category_x[self.n_category_input[0]],
                           self.category_x[self.n_category_input[1]]: category_x[self.n_category_input[1]],
                           self.category_x[self.n_category_input[2]]: category_x[self.n_category_input[2]],
                           self.category_x[self.n_category_input[3]]: category_x[self.n_category_input[3]],
                           self.category_x[self.n_category_input[4]]: category_x[self.n_category_input[4]],
                           self.category_x[self.n_category_input[5]]: category_x[self.n_category_input[5]],
                           self.category_x[self.n_category_input[6]]: category_x[self.n_category_input[6]],
                           self.category_x[self.n_category_input[7]]: category_x[self.n_category_input[7]],
                           self.category_x[self.n_category_input[8]]: category_x[self.n_category_input[8]],
                           self.category_x[self.n_category_input[9]]: category_x[self.n_category_input[9]],
                           self.category_x[self.n_category_input[10]]: category_x[self.n_category_input[10]],
                           self.category_x[self.n_category_input[11]]: category_x[self.n_category_input[11]],
                           self.category_x[self.n_category_input[12]]: category_x[self.n_category_input[12]],
                           self.category_x[self.n_category_input[13]]: category_x[self.n_category_input[13]],
                           self.category_x[self.n_category_input[14]]: category_x[self.n_category_input[14]],
                           self.category_x[self.n_category_input[15]]: category_x[self.n_category_input[15]],
                           self.category_x[self.n_category_input[16]]: category_x[self.n_category_input[16]],
                           self.category_x[self.n_category_input[17]]: category_x[self.n_category_input[17]],
                           self.category_x[self.n_category_input[18]]: category_x[self.n_category_input[18]],
                           self.category_x[self.n_category_input[19]]: category_x[self.n_category_input[19]],
                           self.category_x[self.n_category_input[20]]: category_x[self.n_category_input[20]],
                           self.category_x[self.n_category_input[21]]: category_x[self.n_category_input[21]],
                           self.category_x[self.n_category_input[22]]: category_x[self.n_category_input[22]],
                           self.category_x[self.n_category_input[23]]: category_x[self.n_category_input[23]],
                           self.category_x[self.n_category_input[24]]: category_x[self.n_category_input[24]],
                           self.category_x[self.n_category_input[25]]: category_x[self.n_category_input[25]],
                           self.category_x[self.n_category_input[26]]: category_x[self.n_category_input[26]],
                           self.category_x[self.n_category_input[27]]: category_x[self.n_category_input[27]],
                           self.category_x[self.n_category_input[28]]: category_x[self.n_category_input[28]],
                           self.category_x[self.n_category_input[29]]: category_x[self.n_category_input[29]],
                           self.category_x[self.n_category_input[30]]: category_x[self.n_category_input[30]],
                           self.category_x[self.n_category_input[31]]: category_x[self.n_category_input[31]],
                           self.category_x[self.n_category_input[32]]: category_x[self.n_category_input[32]],
                           self.category_x[self.n_category_input[33]]: category_x[self.n_category_input[33]],
                           self.category_x[self.n_category_input[34]]: category_x[self.n_category_input[34]],
                           self.category_x[self.n_category_input[35]]: category_x[self.n_category_input[35]]})
        return cost

    def fine_tune(self, numerical_x, category_x, y):
        cost, opt = self.sess. \
            run((self.class_cost, self.class_optimizer),
                feed_dict={self.numerical_x: numerical_x,
                           self.category_x[self.n_category_input[0]]: category_x[self.n_category_input[0]],
                           self.category_x[self.n_category_input[1]]: category_x[self.n_category_input[1]],
                           self.category_x[self.n_category_input[2]]: category_x[self.n_category_input[2]],
                           self.category_x[self.n_category_input[3]]: category_x[self.n_category_input[3]],
                           self.category_x[self.n_category_input[4]]: category_x[self.n_category_input[4]],
                           self.category_x[self.n_category_input[5]]: category_x[self.n_category_input[5]],
                           self.category_x[self.n_category_input[6]]: category_x[self.n_category_input[6]],
                           self.category_x[self.n_category_input[7]]: category_x[self.n_category_input[7]],
                           self.category_x[self.n_category_input[8]]: category_x[self.n_category_input[8]],
                           self.category_x[self.n_category_input[9]]: category_x[self.n_category_input[9]],
                           self.category_x[self.n_category_input[10]]: category_x[self.n_category_input[10]],
                           self.category_x[self.n_category_input[11]]: category_x[self.n_category_input[11]],
                           self.category_x[self.n_category_input[12]]: category_x[self.n_category_input[12]],
                           self.category_x[self.n_category_input[13]]: category_x[self.n_category_input[13]],
                           self.category_x[self.n_category_input[14]]: category_x[self.n_category_input[14]],
                           self.category_x[self.n_category_input[15]]: category_x[self.n_category_input[15]],
                           self.category_x[self.n_category_input[16]]: category_x[self.n_category_input[16]],
                           self.category_x[self.n_category_input[17]]: category_x[self.n_category_input[17]],
                           self.category_x[self.n_category_input[18]]: category_x[self.n_category_input[18]],
                           self.category_x[self.n_category_input[19]]: category_x[self.n_category_input[19]],
                           self.category_x[self.n_category_input[20]]: category_x[self.n_category_input[20]],
                           self.category_x[self.n_category_input[21]]: category_x[self.n_category_input[21]],
                           self.category_x[self.n_category_input[22]]: category_x[self.n_category_input[22]],
                           self.category_x[self.n_category_input[23]]: category_x[self.n_category_input[23]],
                           self.category_x[self.n_category_input[24]]: category_x[self.n_category_input[24]],
                           self.category_x[self.n_category_input[25]]: category_x[self.n_category_input[25]],
                           self.category_x[self.n_category_input[26]]: category_x[self.n_category_input[26]],
                           self.category_x[self.n_category_input[27]]: category_x[self.n_category_input[27]],
                           self.category_x[self.n_category_input[28]]: category_x[self.n_category_input[28]],
                           self.category_x[self.n_category_input[29]]: category_x[self.n_category_input[29]],
                           self.category_x[self.n_category_input[30]]: category_x[self.n_category_input[30]],
                           self.category_x[self.n_category_input[31]]: category_x[self.n_category_input[31]],
                           self.category_x[self.n_category_input[32]]: category_x[self.n_category_input[32]],
                           self.category_x[self.n_category_input[33]]: category_x[self.n_category_input[33]],
                           self.category_x[self.n_category_input[34]]: category_x[self.n_category_input[34]],
                           self.category_x[self.n_category_input[35]]: category_x[self.n_category_input[35]],
                           self.y: y})
        return cost

    def show_predict(self, numerical_x, category_x, y):
        y_ = self.sess.\
            run(self.y_, feed_dict={self.numerical_x: numerical_x,
                                    self.category_x[self.n_category_input[0]]: category_x[self.n_category_input[0]],
                                    self.category_x[self.n_category_input[1]]: category_x[self.n_category_input[1]],
                                    self.category_x[self.n_category_input[2]]: category_x[self.n_category_input[2]],
                                    self.category_x[self.n_category_input[3]]: category_x[self.n_category_input[3]],
                                    self.category_x[self.n_category_input[4]]: category_x[self.n_category_input[4]],
                                    self.category_x[self.n_category_input[5]]: category_x[self.n_category_input[5]],
                                    self.category_x[self.n_category_input[6]]: category_x[self.n_category_input[6]],
                                    self.category_x[self.n_category_input[7]]: category_x[self.n_category_input[7]],
                                    self.category_x[self.n_category_input[8]]: category_x[self.n_category_input[8]],
                                    self.category_x[self.n_category_input[9]]: category_x[self.n_category_input[9]],
                                    self.category_x[self.n_category_input[10]]: category_x[self.n_category_input[10]],
                                    self.category_x[self.n_category_input[11]]: category_x[self.n_category_input[11]],
                                    self.category_x[self.n_category_input[12]]: category_x[self.n_category_input[12]],
                                    self.category_x[self.n_category_input[13]]: category_x[self.n_category_input[13]],
                                    self.category_x[self.n_category_input[14]]: category_x[self.n_category_input[14]],
                                    self.category_x[self.n_category_input[15]]: category_x[self.n_category_input[15]],
                                    self.category_x[self.n_category_input[16]]: category_x[self.n_category_input[16]],
                                    self.category_x[self.n_category_input[17]]: category_x[self.n_category_input[17]],
                                    self.category_x[self.n_category_input[18]]: category_x[self.n_category_input[18]],
                                    self.category_x[self.n_category_input[19]]: category_x[self.n_category_input[19]],
                                    self.category_x[self.n_category_input[20]]: category_x[self.n_category_input[20]],
                                    self.category_x[self.n_category_input[21]]: category_x[self.n_category_input[21]],
                                    self.category_x[self.n_category_input[22]]: category_x[self.n_category_input[22]],
                                    self.category_x[self.n_category_input[23]]: category_x[self.n_category_input[23]],
                                    self.category_x[self.n_category_input[24]]: category_x[self.n_category_input[24]],
                                    self.category_x[self.n_category_input[25]]: category_x[self.n_category_input[25]],
                                    self.category_x[self.n_category_input[26]]: category_x[self.n_category_input[26]],
                                    self.category_x[self.n_category_input[27]]: category_x[self.n_category_input[27]],
                                    self.category_x[self.n_category_input[28]]: category_x[self.n_category_input[28]],
                                    self.category_x[self.n_category_input[29]]: category_x[self.n_category_input[29]],
                                    self.category_x[self.n_category_input[30]]: category_x[self.n_category_input[30]],
                                    self.category_x[self.n_category_input[31]]: category_x[self.n_category_input[31]],
                                    self.category_x[self.n_category_input[32]]: category_x[self.n_category_input[32]],
                                    self.category_x[self.n_category_input[33]]: category_x[self.n_category_input[33]],
                                    self.category_x[self.n_category_input[34]]: category_x[self.n_category_input[34]],
                                    self.category_x[self.n_category_input[35]]: category_x[self.n_category_input[35]],
                                    self.y: y})
        auc = roc_auc_score(y_true=y, y_score=y_)
        y_ = y_ // 0.5
        count = 0
        for i in y:
            if i[0] == 0:
                count += 1
        print(count)
        accuracy = accuracy_score(y_true=y, y_pred=y_)
        precision = precision_score(y_true=y, y_pred=y_)
        recall = recall_score(y_true=y, y_pred=y_)
        return [accuracy, auc, precision, recall]
