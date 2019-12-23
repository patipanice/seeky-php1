#!/usr/bin/env python
# coding: utf-8

# In[46]:


import numpy as np
from IPython import get_ipython
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import operator
import imageio
import glob
import os.path as path
import cv2


# In[47]:


def LoR_create_Xb(X):
    N = X.shape[0]
    ones = np.ones([N, 1])
    Xb = np.hstack([ones, X])
    return Xb


# In[48]:


def create_onehot_target(label):
    K = len(np.unique(label))
    N = label.shape[0]
    onehot = np.zeros([N, K])
    for i in range(N):
        onehot[i, label[i, 0]] = 1
    return onehot


# In[49]:


def LoR_find_Yhat_mul_class(X, W):
    Xb = LoR_create_Xb(X)
    Z = np.dot(Xb, W)
    Yhat = np.exp(Z)/np.exp(Z).sum(axis=1, keepdims = True)
    return Yhat


# In[50]:







# In[ ]:




