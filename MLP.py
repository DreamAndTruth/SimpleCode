"""MLP for MNIST"""
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import sklearn.preprocessing as prep

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()


def get_weights(num_in, num_out):
    weight = tf.Variable(
        initial_value=tf.truncated_normal([num_in, num_out], 0, 0.1, tf.float32)
    )
    return weight


def get_biases(num_bias):
    bias = tf.Variable(
        initial_value=tf.zeros([num_bias], tf.float32)
    )
    return bias


class MLP(object):

    def __init__(self, num_input, num_hidden, num_output, learning_rate):
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.x = tf.placeholder(tf.float32, shape=[None, num_input])
        self.y = tf.placeholder(tf.float32, shape=[None, num_output])
        self.keep_prob = tf.placeholder(tf.float32)
        self.learning_rate = learning_rate

    def model(self):
        """Define model construction."""
        w1 = get_weights(self.num_input, self.num_hidden)
        b1 = get_biases(self.num_hidden)
        w2 = get_weights(self.num_hidden, self.num_output)
        b2 = get_biases(self.num_output)
        hidden_logits = tf.add(tf.matmul(self.x, w1), b1)
        hidden_features = tf.nn.relu(hidden_logits)
        hidden_drop = tf.nn.dropout(hidden_features, keep_prob=self.keep_prob)
        output_logits = tf.add(tf.matmul(hidden_drop, w2), b2)
        output_features = tf.nn.relu(output_logits)
        return output_features

    def train(self, output_features):
        """Training the model"""
        global_step = tf.Variable([0], trainable=False)
        loss = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(output_features),
                                             reduction_indices=[1]))
        train_step = tf.train.AdagradOptimizer(self.learning_rate).minimize(loss, global_step=global_step)
        return train_step

    def run_train(self, training_steps, keep_prob):
        sess.run(tf.global_variables_initializer())
        for i in range(training_steps):
            X, Y = mnist.train.next_batch(100)
            loss, global_steps = sess.run(training_steps,
                                          feed_dict={self.x: X, self.keep_prob: keep_prob, self.y: Y})
            if i % 1000 is False:
                print("steps:", '%05d' % global_steps,
                      "loss:", "{:.9f}".format(loss))

    def run_test(self):
