"""
A simple convolutional neural network to classify MNIST digits.
"""
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def view(data, label, save_image=False):
    """ View data as a 28-by-28 pixel image, with an option to save the image
    as 'mnist_<label value>'.png.

    Reshape numpy array 'data' to a 28-by-28 array and view image.
    Args:
        data: 1-by-784 Numpy array. Element values [0,1]
        label: 1-by-10 Numpy array. One-hot.
        save_image: Boolean. Whether to save a png of data.
    """
    W, H = 28, 28
    img = np.ceil(data).astype('uint8') * 255
    img = img.reshape(W, H)
    img = Image.fromarray(img, 'L')

    img_label = np.where(label)[1][0]
    if save_image:
        img.save('./mnist_{0}.png'.format(img_label))
        
    img.show()
    return

def main():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    data, label = mnist.train.next_batch(1)

    view(data, label, True)

main()

    

