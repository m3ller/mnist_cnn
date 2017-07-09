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
def view_mnist(data):
    """ View data as a 28-by-28 pixel image, with an option to save the image
    as 'mnist_<label value>'.png.

    Reshape numpy array 'data' to a 28-by-28 array and view image.
    Args:
        data: batch_size-by-784 Numpy array. Element values [0,1]
        save_image: Boolean. Whether to save a png of data.
    """
    batch_size = data.shape[0]

    img = np.ceil(data).astype('uint8') * 255
    img = img.reshape(batch_size*W, H)
    img = Image.fromarray(img, 'L')
    img.show()

def view_4D(data):
    batch_size, rows, cols, n_hidden = data.shape

    img = data.reshape((batch_size*rows, cols*n_hidden))
    img = Image.fromarray(img, 'L')
    img.show()

def get_cnn():
    """ Build a convolutional neural network for MNIST digits
    """
    x = tf.placeholder(tf.float32, [None, WH], "data_x")
    y = tf.placeholder(tf.float32, [None, N_LABELS], "label_y")

    x_reshape = tf.reshape(x, [-1, W, H, 1])

    # TODO: initialize with Kaiming?
    f1 = tf.get_variable("f1", shape=[5, 5, 1, N_HIDDEN])
    f2 = tf.get_variable("f2", shape=[5, 5, N_HIDDEN, N_HIDDEN])

    conv1 = tf.nn.conv2d(x_reshape, f1, [1,1,1,1], "SAME")
    pool1 = tf.layers.max_pooling2d(conv1, 2, 2)

    conv2 = tf.nn.conv2d(pool1, f2, [1,1,1,1], "SAME")
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2)
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * N_HIDDEN])

    full1 = tf.layers.dense(inputs=pool2_flat, units=256)
    full2 = tf.layers.dense(inputs=full1, units=10)

    tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=tf.log(full2))

    return x, y, full2

def main():
    # Build neural network
    gc_data, gc_label, gc_linear = get_cnn()

    # Run data in neural network
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    data, label = mnist.train.next_batch(3)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        linear = sess.run(gc_linear, feed_dict={gc_data: data, gc_label: label})

    view_mnist(data)
    #view_4D(linear)

main()

    

