#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import os.path as path
import cv2
import pandas as pd
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
from tqdm import tqdm
import numpy as np
from random import shuffle
from keras.models import load_model

import imageio




TRAIN_DIR = 'data/pt'
IMG_SIZE = 100
data = ['hat', 'headphone', 'laptop','bag','handbag','wallet','watch']
img = '1_tester'


# In[2]:


def create_label(image_name):
    word_label = image_name.split('_',1) 
    if word_label[1] == 'hat.png':
        return np.array([1,0,0,0,0,0,0])
    elif word_label[1] == 'headphone.png':
        return np.array([0,1,0,0,0,0,0])
    elif word_label[1] == 'laptop.png':
        return np.array([0,0,1,0,0,0,0])
    elif word_label[1] == 'bag.png':
        return np.array([0,0,0,1,0,0,0])
    elif word_label[1] == 'handbag.png':
        return np.array([0,0,0,0,1,0,0])
    elif word_label[1] == 'wallet.png':
        return np.array([0,0,0,0,0,1,0])
    elif word_label[1] == 'watch.png':
        return np.array([0,0,0,0,0,0,1])


# In[3]:


def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        path = os.path.join(TRAIN_DIR,img)
        img_data=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img_data=cv2.resize(img_data,(IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img_data),create_label(img)])
    shuffle(training_data)
    #np.save('train_dara.npy',training_data)
    return training_data
    


# In[4]:



train_data = create_train_data()


# In[13]:


train = train_data[:5]

X_train = np.array([i[0] for i in train]).reshape(-1,100,100,1)
Y_train = np.array([i[1] for i in train])


# In[15]:


plt.imshow(X_train[1].reshape(IMG_SIZE,IMG_SIZE),cmap='gist_gray')


# In[16]:


load_naja =  load_model('modelreal.h5')


# In[17]:


predicted = load_naja.predict(X_train)


# In[20]:


predicted


# In[22]:


predicteds =np.argmax(predicted)
print(data[predicteds])


# In[ ]:




