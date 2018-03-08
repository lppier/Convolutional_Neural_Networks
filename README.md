# Exploration of Convolutional Neural Networks

**Convolutional Filters - cnn_breakdown.py**

This code demonstrates the role of filters in convolutional neural networks
Two filters are created, a horizontal line filter and a vertical line filter of 7 x 7 pixels each.
The two filters are used to do convolution with the input images and the output feature images are shown. 
The code then goes on to use the convenience function tf.layers.conv2d() that combines the filter creation
and convolution in one step. 

One can observe that the different filters "sieve out" different features in the feature-mapped images. 

**Maxpool - cnn_pool.py**

This code demonstrates how max_pool / avg_pool is used in cnn to subsample the input image. The main goal
of this is to reduce the computational load, memory usage and number of 
parameters. This limits the risk of over-fitting and helps the neural network tolerate a bit of location 
invariance (image shift).
