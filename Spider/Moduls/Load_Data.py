

def Taking_Dataset():

    DATA_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'

    path = tf.keras.utils.get_file('mnist.npz', DATA_URL)

    return np.load(path)

