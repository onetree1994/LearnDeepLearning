# import minist dataset and init it to variable minist
# Note: maybe this dataset is stored in somewhere of google's server, but google's server cannot be visited from china
# mainland, so I have to download csv files and import the dataset myself.

import csv
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def load_data(path_train, path_test, use_image=True):
    train_x = []
    train_y = []
    test_x = []
    print('loading...')
    time_start = time.time()
    with open(path_train, mode='r') as path:
        reader = csv.reader(path)
        for row in reader:
            train_y.append(int(row[0]))
            if use_image:
                image = []
                for irow in range(0, 28):
                    image.append(list(map(int, row[irow*28 + 1: (irow + 1)*28 + 1])))
                train_x.append(image)
            else:
                train_x.append(list(map(int, row[1:])))
    with open(path_test, mode='r') as path:
        reader = csv.reader(path)
        for row in reader:
            if use_image:
                image = []
                for irow in range(0, 28):
                    image.append(list(map(int, row[irow * 28: (irow + 1) * 28])))
                test_x.append(image)
            else:
                test_x.append(list(map(int, row[1:])))
    time_end = time.time()
    print('elapsed time: %.2f' % (time_end - time_start), ' training samples:', len(train_x), ' testing samples:', len(test_x))
    return {'train_x': train_x, 'train_y': train_y, 'test_x': test_x}


def output_result(result, path_out):
    with open(path_out, 'w', newline='') as path:
        writer = csv.writer(path)
        writer.writerow(['ImageId', 'Label'])
        count = 1
        for element in result:
            writer.writerow([count, int(element)])
            count = count + 1


class MNIST(object):
    def __init__(self, train_path, test_path, use_image=True, use_onehot=False):
        _data = load_data(train_path, test_path, use_image)
        self.train_x = _data['train_x']
        self.train_y = []
        self.amount = len(self.train_x)
        if use_onehot:
            for _y in _data['train_y']:
                self.train_y.append([0 for label in range(10)])
                self.train_y[-1][_y] = 1
        else:
            self.train_y = _data['train_y']
        self.test_x = _data['test_x']
        self.test_y = []

    def batch_next(self, batch_amount=10):
        batch = np.random.random_integers(0, self.amount - 1, [batch_amount])
        x_batch = np.array([self.train_x[row] for row in batch])
        y_batch = np.array([self.train_y[row] for row in batch])
        return x_batch, y_batch

    def batch_all(self):
        x_all = np.array(self.train_x)
        y_all = np.array(self.train_y)
        return x_all, y_all


class PerformanceCurve(object):
    def __init__(self):
        self.performance = []
        self.cnt = 0

    def next_value(self, rate):
        self.performance.append([self.cnt, rate])
        self.cnt = self.cnt + 1

    def clear_all(self):
        self.cnt = 0
        self.performance = []

    def show(self, figure_num = 1):
        performance_show = np.array(self.performance)
        plt.figure(figure_num)
        plt.plot(performance_show[:, 0], performance_show[:, 1], '-')
        plt.show()


def weight_init(shape):
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial_value=init)


def bias_init(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(initial_value=init)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def pure_softmax(_mnist, showcurve=False):
    # design the cal graph
    # input
    x = tf.placeholder(dtype=tf.float32, shape=[None, 784])

    # softmax layer
    # w_0 = tf.Variable(weight_init([784, 500]))
    # b_0 = tf.Variable(bias_init([500]))
    # y_0 = tf.nn.relu(tf.matmul(x, w_0) + b_0)
    # w_1 = tf.Variable(weight_init([500, 500]))
    # b_1 = tf.Variable(bias_init([500]))
    # y_1 = tf.nn.relu(tf.matmul(y_0, w_1) + b_1)
    # w = tf.Variable(tf.zeros([500, 10]), dtype=tf.float32)
    # b = tf.Variable(tf.zeros([10]), dtype=tf.float32)
    # y = tf.nn.softmax(logits=tf.matmul(y_1, w) + b)

    w = tf.Variable(tf.zeros([784, 10]), dtype=tf.float32)
    b = tf.Variable(tf.zeros([10]), dtype=tf.float32)
    y = tf.nn.softmax(logits=tf.matmul(x, w) + b)

    # input label
    y_ = tf.placeholder(dtype=tf.float32, shape=[None, 10])

    # optimization function: cross entropy
    cross_entropy = -tf.reduce_sum(input_tensor=y_ * tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(learning_rate=1e-7).minimize(loss=cross_entropy)

    # cal correction rate
    correction_vector = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    correction_rate = tf.reduce_mean(tf.cast(x=correction_vector, dtype=tf.float32))

    init = tf.global_variables_initializer()
    pfr = PerformanceCurve()
    with tf.Session() as sess:
        sess.run(init)
        print("training...")
        time_start = time.time()
        for i in range(2000):
            xs, ys = _mnist.batch_next(100)
            sess.run(train_step, feed_dict={x: xs, y_: ys})
            if not i % 100:
                xs, ys = _mnist.batch_all()
                rate = sess.run(correction_rate, feed_dict={x: xs, y_: ys})
                pfr.next_value(rate)
                print(i, " :correction rate:", rate)
        time_end = time.time()
        print("elapsed time: ", time_end - time_start)
        if showcurve:
            pfr.show()


def cnn(_mnist, showcurve=False):
    # input, x:784, y:10_
    x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
    y_ = tf.placeholder(dtype=tf.float32, shape=[None, 10])
    x_image = tf.reshape(x, [-1, 28, 28, 1])    # [batch, in_height, in_width, in_channels], -1 means cal it based on x

    # conv 1, 28*28->28*28*32
    w_conv1 = weight_init([5, 5, 1, 32])        # [filter_height, filter_width, in_channels, out_channels]
    b_conv1 = bias_init([32])
    h_conv1 = tf.nn.tanh(conv2d(x_image, w_conv1) + b_conv1)

    # pool 1, 28*28*32->14*14*32
    h_pool1 = max_pool_2x2(h_conv1)

    # conv2, 14*14*32->14*14*64
    w_conv2 = weight_init([5, 5, 32, 64])
    b_conv2 = bias_init([64])
    h_conv2 = tf.nn.tanh(conv2d(h_pool1, w_conv2) + b_conv2)

    # pool2, 14*14*64->7*7*64
    h_pool2 = max_pool_2x2(h_conv2)
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

    # full connect1
    w_full1 = weight_init([7 * 7 * 64, 1024])
    b_full1 = bias_init([1024])
    h_full1 = tf.nn.tanh(tf.matmul(h_pool2_flat, w_full1) + b_full1)

    # dropout
    keep_prob = tf.placeholder(tf.float32)
    h_full_drop = tf.nn.dropout(h_full1, keep_prob)

    # output
    w_full2 = weight_init([1024, 10])
    b_full2 = bias_init([10])
    y = tf.nn.softmax(tf.matmul(h_full_drop, w_full2) + b_full2)

    # optimization function: cross entropy
    cross_entropy = -tf.reduce_sum(input_tensor=y_ * tf.log(y))
    train_step = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(loss=cross_entropy)

    # cal correction rate
    correction_vector = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    correction_rate = tf.reduce_mean(tf.cast(x=correction_vector, dtype=tf.float32))

    pfr = PerformanceCurve()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print("training...")
        time_start = time.time()
        for i in range(10000):
            xs, ys = _mnist.batch_next(100)
            sess.run(train_step, feed_dict={x: xs, y_: ys, keep_prob: 0.5})
            if not i % 10:
                # xs, ys = _mnist.batch_all()
                rate = sess.run(correction_rate, feed_dict={x: xs, y_: ys, keep_prob: 1})
                pfr.next_value(rate)
                print(i, " :batch correction rate:", rate)
                if rate > 0.95:
                    break
        time_end = time.time()
        print("elapsed time: ", time_end - time_start)
        print("evaluating...")
        xs, ys = _mnist.batch_all()
        rate = sess.run(correction_rate, feed_dict={x: xs, y_: ys, keep_prob: 1})
        print("whole correction rate:", rate)
        if showcurve:
            pfr.show()


mnist = MNIST('MNIST_data/train_data.csv', 'MNIST_data/test_data.csv', False, True)
# we use pure softmax-method to take a test
# pure_softmax(mnist, True)
cnn(mnist, True)
