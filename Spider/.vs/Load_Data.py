import numpy
import tensorflow

def Taking_Dataset():

    DATA_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'

    path = tensorflow.keras.utils.get_file('mnist.npz', DATA_URL)

    return numpy.load(path)

