"""
A simple convolutional neural network to classify MNIST digits.
"""
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

W, H = 28, 28
WH = W*H
N_LABELS = 10
N_HIDDEN = 10

#TODO: when a batch of data is given, stack the digit images side by side
def view(data, label, save_image=False):
    """ View data as a 28-by-28 pixel image, with an option to save the image
    as 'mnist_<label value>'.png.

    Reshape numpy array 'data' to a 28-by-28 array and view image.
    Args:
        data: 1-by-784 Numpy array. Element values [0,1]
        label: 1-by-10 Numpy array. One-hot.
        save_image: Boolean. Whether to save a png of data.
    """
    img = np.ceil(data).astype('uint8') * 255
    img = img.reshape(W, H)
    img = Image.fromarray(img, 'L')

    img_label = np.where(label)[1][0]
    if save_image:
        img.save('./mnist_{0}.png'.format(img_label))
        
    img.show()
    return

def get_cnn():
    """ Build a convolutional neural network for MNIST digits
    """
    x = tf.placeholder(tf.float32, [None, WH], "data_x")
    y = tf.placeholder(tf.float32, [None, N_LABELS], "label_y")

    x_reshape = tf.reshape(x, [-1, W, H, 1])

    # TODO: initialize with Kaiming?
    f1 = tf.get_variable("f1", shape=[5, 5, 1, N_HIDDEN])

    layer1 = tf.nn.conv2d(x_reshape, f1, [1,1,1,1], "SAME")


    return x, y, layer1

def main():
    # Build neural network
    gc_data, gc_label, gc_linear = get_cnn()

    # Run data in neural network
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    data, label = mnist.train.next_batch(3)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        linear = sess.run(gc_linear, feed_dict={gc_data: data, gc_label: label})

    print linear
    #view(data, label, True)

main()

    

