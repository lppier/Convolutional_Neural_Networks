
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
    plt.imshow(image.astype(np.uint8),interpolation="nearest")
    plt.axis("off")
    plt.show()

# Load sample images
sample_imgs = load_sample_images().images
dataset = np.array(sample_imgs, dtype=np.float32)
batch_size, height, width, channels = dataset.shape
plot_color_image(dataset[0,:,:,:])
plot_color_image(dataset[1,:,:,:])



