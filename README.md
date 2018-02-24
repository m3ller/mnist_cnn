# mnist_cnn
## General Overview
A simple convolutional neural network to classify handwritten digits (MNIST) 
with greater than 98% accuracy. The code includes functions to view the
convolutions and the pooling that occurs within the network, and uses
TensorBoard to track the loss.

## Usage
Train and test CNN
```
python digit_classifier.py
```

Train and test CNN. Peek into the CNN and look at images of a batch of the test
data. See the loss printed onto the terminal during training. 
```
python digit_classifier.py --verbose
```

TensorBoard Events are saved into `./tf_logs`.  To run TensorBoard and see the
training loss,
```
tensorboard --log_dir ./tf_logs
```

## Network Overview
The network takes in a batch of MNIST data.  The data goes through two layers
of convolution/max pooling.  Does a linear transformation.
Use cross entropy and minimize the loss with an Adam optimizer.

## Looking inside the Network
The default batch size for this network is 16, but for the sake of displaying
the array of images with a reasonable amount of space, let's demonstrate the
network with a smaller batch size of 4 samples.

A batch of 4 samples of MNIST data:   
![MNIST_digits](https://github.com/m3ller/mnist_cnn/blob/master/readme_img/handwriting.png)

After going through one layer of convolution (with 32 hidden nodes) and
then max pooling.  Note that the effects of each node is shown along the 32
columns of the image array below.
![conv1](https://github.com/m3ller/mnist_cnn/blob/master/readme_img/conv1.png)   
![pool1](https://github.com/m3ller/mnist_cnn/blob/master/readme_img/pool1.png)

After going through a second layer of convolution (with 32 hidden nodes) and
max pooling:
![conv2](https://github.com/m3ller/mnist_cnn/blob/master/readme_img/conv2.png)   
![pool2](https://github.com/m3ller/mnist_cnn/blob/master/readme_img/pool2.png)

Network makes a prediction on what the handwritten MNIST batch data says. 
This gets displayed on terminal.
![prediction](https://github.com/m3ller/mnist_cnn/blob/master/readme_img/prediction.png)
