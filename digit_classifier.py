"""
A simple convolutional neural network to classify MNIST digits.
"""
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
data, label = mnist.train.next_batch(1)

w, h = 28, 28
data = np.ceil(data).astype('uint8') * 255
data = data.reshape(w, h)
img = Image.fromarray(data)
img.save('myNum.png')
img.show()

