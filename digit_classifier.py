"""
A simple convolutional neural network to classify MNIST digits.
"""
import argparse
import numpy as np
import os
from PIL import Image
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

W, H = 28, 28
WH = W*H
N_LABELS = 10
N_HIDDEN = 32
TRAIN_BATCH_SIZE = 16 
TEST_BATCH_SIZE = 16

def get_args():
    """ Set flags available for this program.  Return available arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose",\
            help="Print loss onto terminal; display a subset of test images",\
            action="store_true")
    args = parser.parse_args()
    return args

def view_mnist(data, save_img_name):
    """ View data as a 28-by-28 pixel image, with an option to save the image
    as ./<save_img_name>.png.

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

    # Save image
    img_name = os.path.join("./", save_img_name + ".png")
    img.save(img_name)

def view_4D(data, save_img_name):
    """ Unroll 4D data and show it as a 2D image.

    Specifically, 'data' has dimensions of batch size, image row, image column, 
    and hidden nodes in a layer (read: filters in a layer).  view_4D(..) will
    display 'data' by unrolling it into a 2D table of images.
    
    Each element of the "table" is an image of the result of passing an input
    'image' into a hidden node. Each i-th row of images is the result of
    passing sample i into different hidden nodes.  Each j-th column of images
    is the result of passing various samples through hidden node j.
    """
    batch_size, rows, cols, n_hidden = data.shape
    data = np.swapaxes(data, 2, 3)
    img = data.reshape((batch_size*rows, cols * n_hidden), order='C')
    img = Image.fromarray(img, 'I')
    img.show()

    # Save image
    img_name = os.path.join("./", save_img_name + ".png")
    img.save(img_name)

def get_cnn():
    """ Build a convolutional neural network for MNIST digits
    """
    x = tf.placeholder(tf.float32, [None, WH], "data_x")
    y = tf.placeholder(tf.float32, [None, N_LABELS], "label_y")

    f1 = tf.get_variable("f1", shape=[5, 5, 1, N_HIDDEN])
    f2 = tf.get_variable("f2", shape=[5, 5, N_HIDDEN, N_HIDDEN])
    w = tf.get_variable("w", shape=[7 * 7 * N_HIDDEN, N_LABELS])
    b = tf.get_variable("b", shape=[1, N_LABELS])

    x_reshape = tf.reshape(x, [-1, H, W, 1])

    conv1 = tf.nn.conv2d(x_reshape, f1, [1,1,1,1], "SAME")
    pool1 = tf.layers.max_pooling2d(conv1, 2, 2)
    pool1 = tf.nn.relu(pool1)

    conv2 = tf.nn.conv2d(pool1, f2, [1,1,1,1], "SAME")
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2)
    pool2 = tf.nn.relu(pool2)

    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * N_HIDDEN])
    full = tf.matmul(pool2_flat, w) + b

    xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=full)
    loss = tf.reduce_sum(xentropy)
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    # Store on TensorBoard
    tf.summary.scalar("loss", loss)
    summary_op = tf.summary.merge_all()

    return x, y, [conv1, conv2], [pool1, pool2], full, loss, optimizer,\
        summary_op

def main():
    # Parse arguments
    args = get_args()

    # Build neural network
    (gc_data, gc_label, gc_convs, gc_pools, gc_pred, gc_loss, gc_optim,
        gc_summary_op) = get_cnn()

    # Read data
    mnist  = input_data.read_data_sets("MNIST_data/", one_hot=True)
    n_train_batches = int(np.floor(mnist.train._num_examples / TRAIN_BATCH_SIZE))
    n_test_batches = int(np.floor(mnist.test._num_examples / TEST_BATCH_SIZE))

    # Run data in neural network
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Training
        print "Training CNN.."
        train_writer = tf.summary.FileWriter("./tf_logs/", sess.graph)
        for ii in xrange(n_train_batches):
            data, label = mnist.train.next_batch(TRAIN_BATCH_SIZE)
            loss, _, summary = sess.run([gc_loss, gc_optim, gc_summary_op],\
                          feed_dict={gc_data: data, gc_label: label})

            if args.verbose and ii % 100. == 0:
                train_writer.add_summary(summary, ii)
                print "Train batch {0} loss: {1}".format(ii, loss)

        # Testing
        print "Testing CNN.."
        n_correct = 0
        for _ in xrange(n_test_batches):
            test_data, test_label = mnist.test.next_batch(TEST_BATCH_SIZE)
            convs, pools, test_pred = sess.run([gc_convs, gc_pools, gc_pred],\
                                          feed_dict={gc_data: test_data,
                                                     gc_label: test_label})

            prediction = np.argmax(test_pred, axis=1)
            answer = np.argmax(test_label, axis=1)
            n_correct += np.sum(np.equal(prediction, answer))
     
        print "Number of correct answers in Test: ", n_correct
        print "Accuracy of answers in Test:       ", \
            n_correct / float(n_test_batches * TEST_BATCH_SIZE) * 100.0
    

    if args.verbose:
        # View some of the test data
        view_mnist(test_data, "handwriting")
        view_4D(convs[0], "conv1")
        view_4D(pools[0], "pool1")
        view_4D(convs[1], "conv2")
        view_4D(pools[1], "pool2")
        print "Subset of Test's prediction: ", np.argmax(test_pred, axis=1)

if __name__ == "__main__":
    main()
