

def Taking_Dataset(tfashion = false, tconv = false, other = false):
    if(tfashion == true) : 
        return tf.keras.datasets.fashion_mnist.load_data()
    if(tconv == true) :
        return datasets.cifar10.load_data()
    return tf.keras.datasets.fashion_mnist.load_data()
    
    #DATA_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'

    #path = tf.keras.utils.get_file('mnist.npz', DATA_URL)

    #return np.load(path)

