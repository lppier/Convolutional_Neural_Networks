import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_images
from PIL import Image


def plot_image(image):
    plt.imshow(image, cmap="gray", interpolation="nearest")
    plt.axis("off")
    plt.show()


def plot_color_image(image):
    plt.imshow(image.astype(np.uint8), interpolation="nearest")
    plt.axis("off")
    plt.show()


def plot_feature_maps(filternum):
    for i in range(filternum):
        plt.imshow(output[0, :, :, i], cmap="gray")
        plt.show()
        plt.imshow(output[1, :, :, i], cmap="gray")
        plt.show()


# Load sample images
sample_imgs = load_sample_images().images
dataset = np.array(sample_imgs, dtype=np.float32)
batch_size, height, width, channels = dataset.shape
plot_color_image(dataset[0, :, :, :])
plot_color_image(dataset[1, :, :, :])

# Create 2 filters
filters_test = np.zeros(shape=(7, 7, channels, 2))

# Create horizontal line filter [row, col, channels, idx]
filters_test[:, 3, :, 0] = 1

# Create vertical line filter [row, col, channels, idx]
filters_test[3, :, :, 1] = 1

plot_image(filters_test[:, :, :, 0])
plot_image(filters_test[:, :, :, 1])

X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
convolution = tf.nn.conv2d(X, filters_test, strides=[1, 2, 2, 1], padding="SAME")

# output [batch_size, height, width, filter_idx]
with tf.Session() as sess:
    output = sess.run(convolution, feed_dict={X: dataset})

plot_feature_maps(2)

# ----------  Shortcut : Use tf.layers.conv2d  ------------#
num_filters = 4
X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
convolution2 = tf.layers.conv2d(X, filters=num_filters, kernel_size=7, strides=[2, 2], padding="SAME")
init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    output = sess.run(convolution2, feed_dict={X: dataset})

plot_feature_maps(num_filters)
