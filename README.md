# mnist_cnn
## General Overview
A simple convolutional neural network to classify handwritten digits (MNIST) 
with greater than 98% accuracy. The code includes functions to view the
convolutions and the pooling that occurs within the network, and uses
TensorBoard to track the loss.

## Network Overview
The network takes in a batch of MNIST data.  The data goes through two layers
of convolution/max pooling.  Does a linear transformation.
Use cross entropy and minimize the loss with an Adam optimizer.

## Looking inside the Network
A batch of 16 samples of MNIST data:
![MNIST_digits](https://github.com/m3ller/mnist_cnn/blob/master/handwriting.png | width=20)

After going through one layer of convolution (with 32 hidden nodes) and
then max pooling:
![conv1](https://github.com/m3ller/mnist_cnn/blob/master/conv1.png)
![pool1](https://github.com/m3ller/mnist_cnn/blob/master/pool1.png)

After going through a second layer of convolution (with 32 hidden nodes) and
max pooling:
![conv2](https://github.com/m3ller/mnist_cnn/blob/master/conv2.png)
![pool2](https://github.com/m3ller/mnist_cnn/blob/master/pool2.png)

Network makes a prediction on what the handwritten MNIST batch data says. 
This gets displayed on terminal.
![MNIST_digits](https://github.com/m3ller/mnist_cnn/blob/master/handwriting.png)
![prediction](https://github.com/m3ller/mnist_cnn/blob/master/prediction.png)
