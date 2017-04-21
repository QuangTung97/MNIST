import tensorflow as tf
import cv2
import sys
import numpy as np

L1 = 200
L2 = 100
L3 = 60
L4 = 30
L5 = 10

X = tf.placeholder(tf.float64, [1, 784])

# Weight
W1 = tf.Variable(tf.zeros([784, L1], dtype=tf.float64))
W2 = tf.Variable(tf.zeros([L1, L2], dtype=tf.float64))
W3 = tf.Variable(tf.zeros([L2, L3], dtype=tf.float64))
W4 = tf.Variable(tf.zeros([L3, L4], dtype=tf.float64))
W5 = tf.Variable(tf.zeros([L4, L5], dtype=tf.float64))

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

result = tf.argmax(Y, 1)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

saver = tf.train.Saver()
saver.restore(sess, "./mnist.ckpt")

impath = sys.argv[1]
if impath is None:
    exit(0)

image = cv2.imread(impath)
if image is None:
    exit(0)

image = cv2.resize(image, (28, 28))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(image.shape)
image = image.astype(np.float64, copy=True)
image = image / 255
image = image.reshape(1, 28 * 28)

print(sess.run(result, feed_dict={X: image})[0])
