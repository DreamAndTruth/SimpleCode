import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# mnist = input_data.read_data_sets('../data/MNIST_data', one_hot=True)
# 可以将数据集更改为fashion_mnist
mnist = input_data.read_data_sets('../data/fashion_mnist', one_hot=True)
sess = tf.InteractiveSession()


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, w):
    """
    2-D conv ops
    :param x:input data in shape [batch, height, width, channels]
    :param w: weights in shape [filter_height, filter_width, in_channels, out_channels]
    """
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x, size):
    """
    max pooling
    :param x: input data
    :param size: filter size
    """
    return tf.nn.max_pool(x, ksize=size, strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])
# 参数-1代表该维度的数目不确定

w_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool(h_conv1, [1, 2, 2, 1])

w_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool(h_conv2, [1, 2, 2, 1])

w_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2,[-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

keep_prop = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=keep_prop)

w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv)))
# 因为是批处理操作，所以此处需要tf.reduce_mean
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

accuracy_prediction = tf.equal(tf.arg_max(y_, 1), tf.arg_max(y_conv, 1))
accuracy = tf.reduce_mean(tf.cast(accuracy_prediction, tf.float32))

tf.global_variables_initializer().run()
for i in range(1000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = sess.run(
            accuracy,
            feed_dict={x: batch[0], y_: batch[1], keep_prop: 1.0})
        print("steps %d, training accuracy %g" % (i, train_accuracy))
    sess.run(train_step,
             feed_dict={x: batch[0], y_: batch[1], keep_prop: 0.5})

test_accuracy = sess.run(
    accuracy,
    feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prop: 1.0})
print("test accuracy %g" % test_accuracy)