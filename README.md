# mnist_cnn
## General Overview
A simple convolutional neural network to classify MNIST digits. The code 
includes functions to view the convolutions/pooling that occurs within the
network, and uses TensorBoard to track the loss.

## Network Overview
The network takes in a batch of MNIST data.  The data goes through two layers
of convolution/max pooling.  Does a linear transformation.
Use cross entropy and minimize the loss with an Adam optimizer.

A batch of MNIST data:
![MNIST_digits](https://github.com/m3ller/mnist_cnn/blob/master/handwriting.png)

After going through one layer of convolution and then max pooling:
![conv1](https://github.com/m3ller/mnist_cnn/blob/master/conv1.png)
![pool1](https://github.com/m3ller/mnist_cnn/blob/master/pool1.png)

After going through a second layer of convolution and max pooling:
![conv2](https://github.com/m3ller/mnist_cnn/blob/master/conv2.png)
![pool2](https://github.com/m3ller/mnist_cnn/blob/master/pool2.png)
