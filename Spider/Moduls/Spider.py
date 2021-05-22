#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import sys

def Taking_Dataset(tfashion = False, tconv = False, other = False):
    if(tfashion == True) : 
        return tf.keras.datasets.fashion_mnist.load_data()
    if(tconv == True) :
        return datasets.cifar10.load_data()
    return tf.keras.datasets.fashion_mnist.load_data()

def Create_Model (tFashion = False, tConv = False, His = False):
    try:
        if (((tFashion and not tConv and not His) 
             or (not tFashion and tConv and not His) 
             or (not tFashion and not tConv and His)) == True):
            if (tFashion == True):
                model = tnFashion()
            if (tConv == True):
                model = tnConv2D()
            if (His == True):
                model = creation(NULL)
    except ValueError:
        print ("Uncorrect types of net.")
    return model
# In[6]:


#function return fashion_test model

def tnFashion ():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    return model


# In[7]:


#function return conv2D_test model 

def tnConv2D ():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))
    return model


# In[8]:


# recurse function to making your own net (taking param must be realise)

def creation (model):
    if (model == NULL):
        model = model.Sequential()
        
    # taking the input (relise it)
    
    if (Flatten == True):
        model.add(layers.Flatten(input_shape=(x_lenth, y_lenth)))
    if (Dense == True):
        if (activate == True):
            model.add(layers.Dense(count, activation = 'relu'))
        else:
            model.add(layers.Dense(count))
    if (Conv2D == True):
        if (input_shape == True):
            model.add(layers.Conv2D(lenth, (x_kernel, y_kernel), 
                                    activation = 'relu', 
                                    input_shape = (x_input, y_input, canals)))
        else:
            model.add(layers.Conv2D(lenth, (x_kernel, y_kernel), 
                                    activation = 'relu'))
    if (MaxPooling2D == True):
        model.add(layers.MaxPooling2D((x_pool_kernel, y_pool_kernel)))
        
    if (next_layer == True):
        model = creation(model)
    else:
        return model


# In[9]:


def Create_result(model):
    model.summary()


def Training (model = Create_Model(tFashion = True), 
             Data = False, tconv = False, tfashion = False, other = False, 
             Train = False, Adam = False, Adagrad = False, Adadelta = False, SGD = False,
             Loss = False, MSE = False, MAE = False, SCC = False,
             Fit = False, count = 10):
    if (Data == True):
        (train_images, train_labels), (test_images, test_labels) = Taking_Dataset(tfashion, tconv, other)
    else:
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
        
    
    train_images = train_images / 255.0
    test_images = test_images / 255.0
        
    if (Train == True):
        if (Optimizer == True):
            if (Adam == True):
                name_optim = 'adam'
            if (Adagrad == True):
                name_optim = 'adagrad'
            if (Adadelta == True):
                name_optim = 'adadelta'
            if (SGD == True):
                name_optim = 'sgd'
        else:
            name_optim = 'adam'
        if (Loss == True):
            if (MSE == True):
                losses = tf.keras.losses.mean_squared_error
            if (MAE == True):
                losses = tf.keras.losses.mean_absolute_error
            if (SCC == True):
                losses = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        else:
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        
        model.compile(optimizer = name_optim,
              loss = losses,
              metrics = ['accuracy'])
    else:
        model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    if (Fit == True):
        # you must give a 'count' - number of epochs
        history = model.fit(train_images, train_labels, epochs=count,
                   validation_data = (test_images, test_labels))
    else:
        history = model.fit(train_images, train_labels, epochs=10,
                   validation_data = (test_images, test_labels))
    
    result = []
    result.append(model)
    result.append(history)
    return result

def Graf_Accur_Epoch (history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()


# In[5]:


def Accur_loss (model):
    model.evaluate(test_images,  test_labels, verbose=2)


# In[6]:


def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(predicted_label,
                                100*np.max(predictions_array),
                                true_label),
                                color=color)


# In[7]:


def plot_value_array(i, predictions_array, true_label):
  true_label = int(true_label[i])
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


# In[8]:


def Image_Statistic (i, test_labels, test_images, predictions):
        plt.figure(figsize=(6,3))
        plt.subplot(1,2,1)
        plot_image(i, predictions[i], test_labels, test_images)
        plt.subplot(1,2,2)
        plot_value_array(i, predictions[i],  test_labels)
        plt.show()


# In[9]:


def Images_Stat (i_range, test_images, test_labels, model):
    
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    predictions = probability_model.predict(test_images)
    
    for i in i_range:
        Image_Statistic(i, test_labels, test_images, predictions)


# In[10]:


def Analyse(model, history, test_images, test_labels,
           Images = False, i_range = range(0, 5), 
           AcLo = False, Graf_Ac = False):
    
    if (Images == True):
        Images_Stat (i_range, test_images, test_labels, model)
    if (AcLo == True):
        Accur_loss (model)
    if (Graf_Ac == True):
        Graf_Accur_Epoch (history)

print('Какую сеть Вы хотите открыть? Поставьте 0, если Fashion и 1, если Conv')
our_net = int(input())
model = Create_Model(tFashion = 1 - our_net, tConv = our_net)
print('Если вы хотите dataset fashion, нажмите 0, если вы хотите dataset conv, нажмите 1')
fdata = int(input())
print('Хотите ли Вы обучить модель не по умолчанию? Если хотите, поставьте 1, и 0 в обратном случае')
f_tran = int(input())
if f_tran:
    print('Выберете оптимизатор: 0 - Adam, 1 - Adagrad, 2 - Adadelta, 3 - sgv')
    f_opt = int(input())
print('Хотите ли вы изменить способ вычисления ошибки? Да - 1, Нет - 0')
f_mist = int(input())
if f_mist:
    print('Выберете способ: 0 - MSE, 1 - MAE, 2 - SCC')
    mist = int(input())
print('Если ВЫ хотите изменить кол-во эпох, нажмите 1. 0 в обратном случае')
f_ep = int(input())
if f_ep:
    print('Введите кол-во эпох')
    num_ep = int(input())
tconv = False
tfashion = False
other = False
if fdata :
    Data = True
    tconv = True
else:
    Data = False
Adam = False
Adagrad = False
Adadelta = False
SGD = False
if f_tran:
    Train = True
    if (f_opt == 0):
        Adam = True
    if (f_opt == 1):
        Adagrad = True
    if (f_opt == 2):
        Adadelta = True
    if (f_opt == 3):
        SGD = True
else:
    Train = False
    
MSE = False
MAE = False
SCC = False

if f_mist:
    Loss = True
    if (mist == 0):
        MSE = True
    if (mist == 1):
        MAE = True
    if (mist == 2):
        SCC = True
else:
    Loss = False
if f_ep:
    Fit = True
    count = num_ep
else:
    Fit = False

result = Training(model, 
             Data, tconv, tfashion, other,
             Train, Adam, Adagrad, Adadelta, SGD,
             Loss, MSE, MAE, SCC,
             Fit, count)

print('Нужен ли Вам график точность от эпохи? 1 - да, 0 - нет')
Graf_Ac = int(input())

print('Нужен ли Вам данные потери точности? 1 - да, 0 - нет')
AcLo = int(input())

print('Хотите посмотреть рез-т сети? 1 - да, 0 - нет')
Images = int(input())
i_range = []
if Images:
    print('Какие картинки хотите посмотреть? Введите номера картинок. По умолчанию первые 5. По окночанию ввода введите -1')
    tmp = int(input())
    while (tmp != -1):
        i_range.append(tmp)
        tmp = int(input())

if (Graf_Ac == 1):
    Graf_Ac = True
else:
    Graf_Ac = False
if (AcLo == 1):
    AcLo = True
else:
    AcLo = False
if (Images == 1):
    Images = True
else:
    Images = False

if (Data == True):
    (train_images, train_labels), (test_images, test_labels) = Taking_Dataset(tfashion, tconv, other)
else:
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

test_images = test_images / 255.0
        
Analyse(result[0], result[1], test_images, test_labels,
           Images, i_range, 
           AcLo, Graf_Ac)


        


