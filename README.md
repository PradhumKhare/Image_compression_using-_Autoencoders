# Image_compression_using-_Autoencoders
Compression of image file using Deep neural networks
We are using keras fashion mnist set for training our neural network
At first we creates an encoding model which encodes the input image using convolution layes and max pooling layer .
then we created a decoder model which decodes the encoded image using keras convolutionaltranpose layer and up sampling
Finally we merge these two model to make our autoencoder model
Autoencoder model was trained on the fashion mnist data set which acheives an accuracy of 94% .
