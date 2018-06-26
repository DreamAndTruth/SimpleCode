"""Softmax Regression For MNISTdata."""
import tensorflow as tf
import sklearn.preprocessing as prep
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


class SoftmaxRegression(object):
    """SoftmaxRegression For MNIST data."""

    def __init__(self, learning_rate, n_input, n_output):
        """Initialize all the paramaters in the class."""
        self.learning_rate = learning_rate
        self.n_input = n_input
        self.n_output = n_output
        self.images = tf.placeholder(tf.float32, [None, self.n_input])
        self.labels = tf.placeholder(tf.float32, [None, 10])
        self.weights = self._get_weights_()
        self.biases = self._get_biases_()
        self.sess = tf.Session()
        self.output = self.Model()
        # self.cost = self.cross_entropy()
        self.cost = self.MSE()
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(
            self.cost
        )

    def _get_weights_(self):
        weights = tf.Variable(
            initial_value=tf.random_normal([self.n_input, self.n_output]),
            dtype=tf.float32)
        return weights

    def _get_biases_(self):
        biases = tf.Variable(initial_value=np.zeros([self.n_output]),
                             dtype=tf.float32)
        return biases

    def Model(self):
        """Define the model."""
        affine = tf.add(
            tf.matmul(self.images, self.weights),
            self.biases
        )
        return tf.nn.softmax(affine)

    def cross_entropy(self):
        """Define cross entropy."""
        return -tf.reduce_mean(
            self.labels * tf.log(tf.clip_by_value(self.output, 1e-10, 1.0))
        )

    def MSE(self):
        """MSE loss"""
        loss = tf.reduce_mean(tf.pow(tf.subtract(self.labels, self.output),
                                     2.0))
        return loss

    def one_step_training(self, images, labels):
        """One step training."""
        self.sess.run(tf.global_variables_initializer())
        cost, _ = self.sess.run((self.cost, self.optimizer),
                                feed_dict={self.images: images,
                                           self.labels: labels})
        return cost
num_images = mnist.train.num_examples
epoch = 20
batch_size = 128
num_batch = int(num_images / 128)
step = 2


def standard_scale(X_train, X_test):
    """使用Sklearn中的类对数据进行标准化处理."""
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test


def get_random_block_from_data(data, batch_size):
    """随机生成一个batch_size数据."""
    start_index = np.random.randint(0, len(data)-batch_size)
    return data[start_index:(start_index + batch_size)]

# 未对数据进行标准化处理
X_train = mnist.train.images
X_test = mnist.test.images
X_val = mnist.validation.images

Y_train = mnist.train.labels
Y_test = mnist.test.labels
Y_val = mnist.validation.labels
model = SoftmaxRegression(learning_rate=0.01,
                          n_input=784,
                          n_output=10,
                          )
# 运行时间很慢，是在softmax还是在cross entropy中存在大量运算
# 在使用MSE作为损失函数时，运行时间依然较长
# 因此大量运算可能集中在softmax运算当中
for i in range(epoch):
    cost = 0
    for j in range(num_batch):
        X = get_random_block_from_data(X_train, batch_size)
        Y = get_random_block_from_data(Y_train, batch_size)
        cost += model.one_step_training(X, Y)
    if (i % step) == 0:
        print(i, cost / 2)

# 在类的定义中缺少对于正确率的运算以及在test集中测试情况