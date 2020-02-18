# MNIST-number_recognition-using-CNN
This is yet another more powerful but time consuming code for recognizing hand-wriiten digits from MNIST dataset
This code uses tensorflow's CNN method to train and test the data, the difference between this and the FFNN is that we don't have to perform feature extraction in CNN.
In this code max pooling is used to downsample the output from each layer
Three convolutional layers each with 32,64,64 filters respectively with stride=1, the activation function is same for all. After each layer max pooling is applied with stride=2
Train accuracy= 99.87 Test Accuracy= 99.23
Caution: to test it with your data provide balanced dataset
