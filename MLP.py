"""MLP for MNIST"""
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import sklearn.preprocessing as prep

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.Session()


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

    def __init__(self, num_input, num_hidden, num_output):
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.x = tf.placeholder(tf.float32, shape=[None, num_input])

    def model(self):
        w1 = get_weights(self.num_input, self.num_hidden)
        b1 = get_biases(self.hidden)
        w2 = get_weights(self.hidden, self.num_output)
        b2 = get_biases(self.num_output)
        hidden_logits = tf.add(tf.matmul(self.x, w1), b1)
        hidden_features = tf.nn.relu(hidden_logits)
        output_logits = tf.add(tf.matmul(hidden_features, w2), b2)
        output_features = tf.nn.relu(output_logits)
        return output_features


