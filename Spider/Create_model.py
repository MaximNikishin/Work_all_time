#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


import sys


# In[3]:


from tensorflow.keras import datasets, layers, models


# In[4]:


import numpy as np
import matplotlib.pyplot as plt


# In[5]:


# Main function, return model of net.

def Create_Model (tFashion = False, tConv = False, His = False):
    try:
        if (((tFashion and not tConv and not His) 
             or (not tFashion and tConv and not His) 
             or (not tFashion and not tConv and His)) == True):
            if (tFashion == true):
                model = tnFashion()
            if (tConv == true):
                model = tnConv2D()
            if (His == true):
                model = creation(NULL)
    except ValueError:
        print ("Uncorrect types of net.")


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
        model = model
        
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


# In[ ]:




