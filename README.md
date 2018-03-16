![Figure 1-1](https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/TensorFlowLogo.svg/200px-TensorFlowLogo.svg.png "Tensorflow")

# Exploration of Convolutional Neural Networks in Tensorflow

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


# High Accuracy CNN Minst 

**high_accuracy_cnn_minst.py**

CNN using the classic Minst dataset - this gives high accuracy of above 99%. 
A demonstration of the 
how to use the tensorflow pipeline. 

# Using Tensorboard
![Figure 1-2](https://www.tensorflow.org/images/mnist_tensorboard.png "Tensorboard")

**minst_cnn_with_tensorboard.py**

This is the previous minst file with parameters added to generate the graph for Tensorboard.
To visualize, run tensorboard : 

Run in Firefox : 

`tensorboard --logdir=/tmp/data/logs`

Firefox is recommended as the node connections don't seem to appear in the latest versions of Chrome.

# Early Stopping

**minst_cnn_with_tensorboard.py**

In this version of cnn minst, early stopping mechanism is introduced. 
The validation data is used to validate the training and the loss is measured. 

Every 500 batches of training, we check if the loss is better than the loss we have got so far. If it 
is better, make a note of it and save the model at this point of time. 

If there are 20 such iterations where there is no perceived improvement in the loss, we stop the training 
early as there is no point training any further. 
````
0 Train accuracy: 0.98 Validation accuracy: 0.979 best loss:  0.071774356  
1 Train accuracy: 0.98 Validation accuracy: 0.982 best loss:  0.05199903  
2 Train accuracy: 0.98 Validation accuracy: 0.9824 best loss:  0.043591887  
3 Train accuracy: 1.0 Validation accuracy: 0.9876 best loss:  0.04349228  
4 Train accuracy: 1.0 Validation accuracy: 0.9864 best loss:  0.041428853  
5 Train accuracy: 0.98 Validation accuracy: 0.989 best loss:  0.041428853  
6 Train accuracy: 1.0 Validation accuracy: 0.9884 best loss:  0.041428853  
7 Train accuracy: 1.0 Validation accuracy: 0.9892 best loss:  0.041428853  
8 Train accuracy: 1.0 Validation accuracy: 0.9874 best loss:  0.041428853  
9 Train accuracy: 1.0 Validation accuracy: 0.9864 best loss:  0.041428853  
10 Train accuracy: 1.0 Validation accuracy: 0.9882 best loss:  0.041428853  
11 Train accuracy: 1.0 Validation accuracy: 0.9896 best loss:  0.041428853  
Early stopping! ` 
````

We then use this model on the test data to gauge the real world performance. 
````
Final accuracy on test set: 0.9902
````

# Inception_v3

**inception_v3.py**

Inception_v3 is a pre-trained network consisting of 1001 classes that you can identify. 
Here we download the model and try to identify a dog.

Top 5 predictions with confidence level

````
African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus: 94.95%
hyena, hyaena: 3.48%
European fire salamander, Salamandra salamandra: 0.03%
bearskin, busby, shako: 0.02%
dhole, Cuon alpinus: 0.02%
````

# Transfer Learning with Inception_v3

**transfer_learning.py**

Here we leverage on Inception_v3 and apply transfer learning to train classification of flowers. 
The last layer was replaced by a dense layer which does classification for 5 flower types.
````
['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
````

````
Computing final accuracy on the test set (this will take a while)...
Test accuracy: 0.7315809
````
This supports tensorboard so you can visualize the accuracy over time. This code also has the early stopping code. 
This guy seems to have a great accuracy using transfer learning with inception, to reference in future : https://kwotsin.github.io/tech/2017/02/11/transfer-learning.html
