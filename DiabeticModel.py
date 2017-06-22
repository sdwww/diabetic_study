import tensorflow as tf
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score


class DiabeticModel(object):
    def __init__(self, n_input, n_hidden_1, n_hidden_2, n_output, init_scale,
                 optimizer=tf.train.AdadeltaOptimizer(0.05)):
        self.n_input = n_input
        self.n_hidden_1 = n_hidden_1
        self.n_hidden_2 = n_hidden_2
        self.n_output = n_output
        self.init_scale = init_scale

        self.weights = self._initialize_weights()

        # model
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.y = tf.placeholder(tf.float32, [None, self.n_output])
        self.hidden_1 = tf.nn.relu(tf.add(tf.matmul(self.x, self.weights['w1']), self.weights['b1']))
        self.reconstruction = tf.nn.relu(
            tf.add(tf.matmul(self.hidden_1, self.weights['w_recon']), self.weights['b_recon']))
        self.hidden_2 = tf.nn.relu(tf.add(tf.matmul(self.hidden_1, self.weights['w2']), self.weights['b2']))
        self.y_ = tf.nn.sigmoid(tf.add(tf.matmul(self.hidden_2, self.weights['w_output']), self.weights['b_output']))

        # cost
        self.encoder_cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.encoder_optimizer = optimizer.minimize(self.encoder_cost)
        self.class_cost = tf.reduce_sum(tf.pow(tf.subtract(self.y, self.y_), 2.0))
        self.class_optimizer = optimizer.minimize(self.class_cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(
            tf.random_normal([self.n_input, self.n_hidden_1], stddev=self.init_scale, dtype=tf.float32))
        all_weights['b1'] = tf.Variable(
            tf.random_normal([self.n_hidden_1], stddev=self.init_scale, dtype=tf.float32))
        all_weights['w_recon'] = tf.Variable(
            tf.random_normal([self.n_hidden_1, self.n_input], stddev=self.init_scale, dtype=tf.float32))
        all_weights['b_recon'] = tf.Variable(
            tf.random_normal([self.n_input], stddev=self.init_scale, dtype=tf.float32))
        all_weights['w2'] = tf.Variable(
            tf.random_normal([self.n_hidden_1, self.n_hidden_2], stddev=self.init_scale, dtype=tf.float32))
        all_weights['b2'] = tf.Variable(
            tf.random_normal([self.n_hidden_2], stddev=self.init_scale, dtype=tf.float32))
        all_weights['w_output'] = tf.Variable(
            tf.random_normal([self.n_hidden_2, self.n_output], stddev=self.init_scale, dtype=tf.float32))
        all_weights['b_output'] = tf.Variable(
            tf.random_normal([self.n_output], stddev=self.init_scale, dtype=tf.float32))
        return all_weights

    def pre_train(self, x):
        cost, opt = self.sess.run((self.encoder_cost, self.encoder_optimizer), feed_dict={self.x: x})
        return cost

    def fine_tune(self, x, y):
        cost, opt = self.sess.run((self.class_cost, self.class_optimizer), feed_dict={self.x: x, self.y: y})
        return cost

    def calc_encoder_cost(self, x):
        return self.sess.run(self.encoder_cost, feed_dict={self.x: x})

    def show_predict(self, x, y):
        y_ = self.sess.run(self.y_, feed_dict={self.x: x, self.y: y})
        print(y_.shape)
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
