#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


from tensorflow.keras import datasets, layers, models


# In[4]:


import sys


# In[5]:


import numpy as np
import matplotlib.pyplot as plt


# In[7]:


#Taking_Dataset must be raelise (it returns numpy dataset)

our_mnist = Taking_Dataset()


# In[8]:


(train_images, train_labels), (test_images, test_labels) = our_mnist.load_data()


# In[13]:


def Traning (model = Create_Model(tFashion = True), Data = False, Train = False, Fit = False):
    if (Data == True):
        our_mnist = Taking_Dataset()
    else:
        our_mnist = our_mnist = tf.keras.datasets.fashion_mnist
        
    (train_images, train_labels), (test_images, test_labels) = our_mnist.load_data()
    
    train_images = train_images / 255.0
    test_images = test_images / 255.0
        
    if (Train == True):
        # Not realise yet
        print ("Trainning")
    else:
        model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    if (Fit == True):
        # Not realise yet
        print ("Fitting")
    else:
        history = model.fit(train_images, train_labels, epochs=10,
                   validation_data = (test_images, test_labels))
    
    result = []
    result.append(model)
    result.append(history)
    return result


# In[ ]:




