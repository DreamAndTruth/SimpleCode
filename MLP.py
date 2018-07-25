"""MLP for MNIST"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

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


num_input = 784
num_hidden = 300
num_output = 10
learning_rate = 0.1

keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32, [None, num_input])
y = tf.placeholder(tf.float32, [None, num_output])

w1 = get_weights(num_input, num_hidden)
b1 = get_biases(num_hidden)
w2 = get_weights(num_hidden, num_output)
b2 = get_biases(num_output)

hidden_features = tf.nn.relu(tf.matmul(x, w1) + b1)
hidden_drop = tf.nn.dropout(hidden_features, keep_prob=keep_prob)
output_features = tf.nn.softmax(tf.matmul(hidden_drop, w2) + b2)

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(output_features), reduction_indices=[1]))
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(loss)

sess.run(tf.global_variables_initializer())
for i in range(10000):
    X, Y = mnist.train.next_batch(10)
    _, total_loss = sess.run([train_step, loss], feed_dict={x: X, keep_prob: 0.5, y: Y})
    if i % 1000 == 0:
        print(i, total_loss)
    # train_step返回的不是数值，只是一个更新操作，需要得到的所有数值需要传递近sess.run()当中
    # 变量名称最好不要重复
'''定义准确率'''