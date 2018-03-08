import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_images


def plot_image(image):
    plt.imshow(image, cmap="gray", interpolation="nearest")
    plt.axis("off")
    plt.show()


def plot_color_image(image):
    plt.imshow(image.astype(np.uint8), interpolation="nearest")
    plt.axis("off")
    plt.show()


# Load sample images
sample_imgs = load_sample_images().images
dataset = np.array(sample_imgs, dtype=np.float32)
batch_size, height, width, channels = dataset.shape
plot_color_image(dataset[0, :, :, :])
plot_color_image(dataset[1, :, :, :])

# Create a graph with input X with a max pooling layer, kernel size 2 x 2, stride 2
X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
max_pool = tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
# max_pool = tf.nn.avg_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

with tf.Session() as sess:
    output = sess.run(max_pool, feed_dict={X: dataset})

# Note that the image is somewhat "degraded", this is caused by maxpool taking the max in the 2 x 2 pool
plot_color_image(output[0, :, :, :])
plot_color_image(output[1, :, :, :])
