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
N_HIDDEN = 5

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
    print data.dtype

    img = data.reshape((batch_size*rows, cols*n_hidden), order='C')
    img = Image.fromarray(img, 'I')
    img.show()

def get_cnn():
    """ Build a convolutional neural network for MNIST digits
    """
    x = tf.placeholder(tf.float32, [None, WH], "data_x")
    y = tf.placeholder(tf.float32, [None, N_LABELS], "label_y")
    bias1 = tf.get_variable("bias1", shape=[N_HIDDEN])

    x_reshape = tf.reshape(x, [-1, H, W, 1])

    # TODO: initialize with Kaiming?
    f1 = tf.get_variable("f1", shape=[5, 5, 1, N_HIDDEN])
    f2 = tf.get_variable("f2", shape=[5, 5, N_HIDDEN, N_HIDDEN])
    w1 = tf.get_variable("w1", shape=[7 * 7 * N_HIDDEN, N_LABELS])
    b1 = tf.get_variable("b1", shape=[1, N_LABELS])

    conv1 = tf.nn.conv2d(x_reshape, f1, [1,1,1,1], "SAME")
    conv1 = conv1 + bias1
    pool1 = tf.layers.max_pooling2d(conv1, 2, 2)
    pool1 = tf.nn.relu(pool1)

    conv2 = tf.nn.conv2d(pool1, f2, [1,1,1,1], "SAME")
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2)
    pool2 = tf.nn.relu(pool2)
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * N_HIDDEN])

    full1 = tf.matmul(pool2_flat, w1) + b1

    xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=full1)
    loss = tf.reduce_sum(xentropy)
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    return x, y, optimizer, conv1, loss

def main():
    # Build neural network
    gc_data, gc_label, gc_optim, gc_conv2, gc_loss = get_cnn()

    # Run data in neural network
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Training
        data = None
        for _ in xrange(1000):
            data, label = mnist.train.next_batch(10)
            _, loss, conv2 = sess.run([gc_optim, gc_loss, gc_conv2], feed_dict={gc_data: data, gc_label: label})
            print loss

    view_mnist(data)
    view_4D(conv2)
    print conv2

main()

    

