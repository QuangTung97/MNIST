import numpy as np
import tensorflow as tf
import cv2
import sys

H = 1
K = 6
L = 12
M = 24
N = 200
P = 10

# Weight
W1 = tf.Variable(tf.zeros((5, 5, H, K)))
b1 = tf.Variable(tf.zeros([K]))

W2 = tf.Variable(tf.zeros((5, 5, K, L)))
b2 = tf.Variable(tf.zeros([L]))

W3 = tf.Variable(tf.zeros((4, 4, L, M)))
b3 = tf.Variable(tf.zeros([M]))

W4 = tf.Variable(tf.zeros((7 * 7 * M, N)))
b4 = tf.Variable(tf.zeros([N]))

W5 = tf.Variable(tf.zeros((N, P)))
b5 = tf.Variable(tf.zeros([P]))

# None mean dimension can be of any length
X = tf.placeholder(tf.float32, [None, 28, 28, 1])

# (28, 28, K)
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=(1, 1, 1, 1), padding='SAME') + b1)

# (14, 14, L)
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=(1, 2, 2, 1), padding='SAME') + b2)

# (7, 7, M)
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=(1, 2, 2, 1), padding='SAME') + b3)

Yi = tf.reshape(Y3, shape=(-1, 7 * 7 * M))
Y4 = tf.nn.relu(tf.matmul(Yi, W4) + b4)
Y = tf.nn.softmax(tf.matmul(Y4, W5) + b5)

result = tf.argmax(Y, 1)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

saver = tf.train.Saver()
saver.restore(sess, "./conv.ckpt")

impath = sys.argv[1]
if impath is None:
    exit(0)

image = cv2.imread(impath)
if image is None:
    exit(0)

image = cv2.resize(image, (28, 28))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = image.astype(np.float32, copy=True)
image = image / 255
image = image.reshape(1, 28, 28, 1)
feed_data = {X: image}
print(sess.run(result, feed_dict=feed_data)[0])
