import input_data
import numpy as np
import cv2

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

ntrain = 55000
# train image
train_X, train_Y = mnist.train.next_batch(ntrain)
X = train_X.reshape(ntrain, 28, 28, 1) * 255
Y = np.argmax(train_Y, 1)


images = open("./train/images.txt", "a")
for i in range(0, ntrain):
    cv2.imwrite("./train/%s %s.png" % (i, Y[i]), X[i])
    images.write("%s %s\n" % (i, Y[i]))
images.close()
