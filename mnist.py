import tensorflow as tf
import input_data
# import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

L1 = 200
L2 = 100
L3 = 60
L4 = 30
L5 = 10

# None mean dimension can be of any length
X = tf.placeholder(tf.float64, [None, 784])
rate = tf.placeholder(tf.float64, [])

# Weight
W1 = tf.Variable(tf.truncated_normal([784, L1], stddev=0.1, dtype=tf.float64))
W2 = tf.Variable(tf.truncated_normal([L1, L2], stddev=0.1, dtype=tf.float64))
W3 = tf.Variable(tf.truncated_normal([L2, L3], stddev=0.1, dtype=tf.float64))
W4 = tf.Variable(tf.truncated_normal([L3, L4], stddev=0.1, dtype=tf.float64))
W5 = tf.Variable(tf.truncated_normal([L4, L5], stddev=0.1, dtype=tf.float64))

b1 = tf.Variable(tf.zeros([L1], dtype=tf.float64))
b2 = tf.Variable(tf.zeros([L2], dtype=tf.float64))
b3 = tf.Variable(tf.zeros([L3], dtype=tf.float64))
b4 = tf.Variable(tf.zeros([L4], dtype=tf.float64))
b5 = tf.Variable(tf.zeros([L5], dtype=tf.float64))

Y1 = tf.nn.relu(tf.matmul(X, W1) + b1)
Y2 = tf.nn.relu(tf.matmul(Y1, W2) + b2)
Y3 = tf.nn.relu(tf.matmul(Y2, W3) + b3)
Y4 = tf.nn.relu(tf.matmul(Y3, W4) + b4)
Y = tf.nn.softmax(tf.matmul(Y4, W5) + b5)
Y_ = tf.placeholder(tf.float64, [None, 10])

is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float64))

cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))

train_step = tf.train.GradientDescentOptimizer(rate).minimize(cross_entropy)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

train_a = []  # accuracy
train_c = []  # cross entropy
test_a = []
test_c = []

train_data = None
test_data = {X: np.float64(mnist.test.images),
             Y_: np.float64(mnist.test.labels)}
print(mnist.test.images.shape)


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


loop = 10000
mod = 50
for i in range(loop):
    train_X, train_Y = mnist.train.next_batch(100)
    learning_rate = 0.005 * (1.0 - math.exp(
        float(i - loop) * 4 / loop)
    )

    train_data = {X: np.float64(train_X),
                  Y_: np.float64(train_Y),
                  rate: learning_rate}

    e = sess.run(train_step, feed_dict=train_data)
    print(i, learning_rate)

    if (i % mod == 0):
        check_step()

saver = tf.train.Saver()
saver.save(sess, "./mnist.ckpt")

draw_graph()
print(test_a[-1])
