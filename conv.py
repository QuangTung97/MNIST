import tensorflow as tf
import input_data
import math
# import numpy as np
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def get_test_data():
    test_X = mnist.test.images.reshape(-1, 28, 28, 1)
    test_Y = mnist.test.labels
    return {X: test_X, Y_: test_Y}


def get_next_batch():
    train_X, train_Y = mnist.train.next_batch(100)
    train_X = train_X.reshape((100, 28, 28, 1))
    train_Y = train_Y
    return train_X, train_Y


H = 1
K = 6
L = 12
M = 24
N = 200
P = 10

# Weight
W1 = tf.Variable(tf.truncated_normal((5, 5, H, K), stddev=0.1))
b1 = tf.Variable(tf.zeros([K]))

W2 = tf.Variable(tf.truncated_normal((5, 5, K, L), stddev=0.1))
b2 = tf.Variable(tf.zeros([L]))

W3 = tf.Variable(tf.truncated_normal((4, 4, L, M), stddev=0.1))
b3 = tf.Variable(tf.zeros([M]))

W4 = tf.Variable(tf.truncated_normal((7 * 7 * M, N), stddev=0.1))
b4 = tf.Variable(tf.zeros([N]))

W5 = tf.Variable(tf.truncated_normal((N, P), stddev=0.1))
b5 = tf.Variable(tf.zeros([P]))

# None mean dimension can be of any length
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
rate = tf.placeholder(tf.float32, [])

# (28, 28, K)
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=(1, 1, 1, 1), padding='SAME') + b1)

# (14, 14, L)
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=(1, 2, 2, 1), padding='SAME') + b2)

# (7, 7, M)
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=(1, 2, 2, 1), padding='SAME') + b3)

Yi = tf.reshape(Y3, shape=(-1, 7 * 7 * M))
Y4 = tf.nn.relu(tf.matmul(Yi, W4) + b4)
Y = tf.nn.softmax(tf.matmul(Y4, W5) + b5)
Y_ = tf.placeholder(tf.float32, [None, 10])

is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y + 1e-37))
train_step = tf.train.GradientDescentOptimizer(rate).minimize(cross_entropy)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

train_a = []  # accuracy
train_c = []  # cross entropy
test_a = []
test_c = []

train_data = None
test_data = get_test_data()


def check_step():
    global train_a, train_c, test_a, test_c

    # train
    a, c = sess.run([accuracy, cross_entropy], feed_dict=train_data)
    train_a.append(a)
    train_c.append(c)

    # test
    a, c = sess.run([accuracy, cross_entropy], feed_dict=test_data)
    test_a.append(a)
    test_c.append(c)


def write_stat():
    global train_a, train_c, test_a, test_c
    fp = open('stat.txt', 'w')
    for e in train_a:
        fp.write('%s ' % e)
    fp.write('\n')

    for e in train_c:
        fp.write('%s ' % e)
    fp.write('\n')

    for e in test_a:
        fp.write('%s ' % e)
    fp.write('\n')

    for e in test_c:
        fp.write('%s ' % e)
    fp.write('\n')
    fp.close()


def read_stat():
    global train_a, train_c, test_a, test_c
    fp = open('stat.txt', 'r')
    train_a = map(float, fp.readline().split())
    train_c = map(float, fp.readline().split())
    test_a = map(float, fp.readline().split())
    test_c = map(float, fp.readline().split())
    fp.close()


def draw_graph():
    global train_a, train_c, test_a, test_c
    size = len(train_a)
    plt.figure(1)
    plt.subplot(211)
    plt.xlabel("nsamples")
    plt.ylabel("cross entropy")
    plt.plot(range(1, size + 1), train_c, label='train', color='r')
    plt.plot(range(1, size + 1), test_c, label='test', color='b')
    plt.legend()

    plt.subplot(212)
    plt.xlabel("nsamples")
    plt.ylabel("accuracy")
    plt.plot(range(1, size + 1), train_a, label='train', color='r')
    plt.plot(range(1, size + 1), test_a, label='test', color='b')
    plt.legend()
    plt.show()


loop = 30000
mod = 300
for i in range(loop + 1):
    train_X, train_Y = get_next_batch()
    learning_rate = 0.0005 * (1.0 - math.exp(
        float(i - loop) * 4 / loop)
    )

    train_data = {X: train_X, Y_: train_Y,
                  rate: learning_rate}

    e = sess.run(train_step, feed_dict=train_data)
    print(i, learning_rate)

    if (i % mod == 0):
        check_step()

saver = tf.train.Saver()
saver.save(sess, "./conv.ckpt")

write_stat()
# read_stat()
# draw_graph()

print(test_a[-1])
