#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


from tensorflow.keras import datasets, layers, models


# In[3]:


import numpy as np
import matplotlib.pyplot as plt


# In[4]:


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


def Images_Stat (i_range, dataset, model):
    (train_images, train_labels), (test_images, test_labels) = dataset.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    predictions = probability_model.predict(test_images)
    
    for i in i_range:
        Image_Stistic(i, test_labels, test_images, predictions)


# In[10]:


def Analise(model, history, dataset,
           Images = False, i_range = range(0, 5), 
           AcLo = False, Graf_Ac = False):
    
    if (Images == True):
        Images_Stat (i_range, dataset, model)
    if (AcLo == True):
        Accur_loss (model)
    if (Graf_Ac == True):
        Graf_Accur_Epoch (history)


# In[ ]:




