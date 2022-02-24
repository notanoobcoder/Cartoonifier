#!/usr/bin/env python
# coding: utf-8

# In[6]:


get_ipython().system('pip install opencv-python')


# In[ ]:


#Extracting and displaying the image


# In[89]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[90]:


def read_file(filename):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()
    return img


# In[91]:


filename = "169225303_384696200134487_8943204076666535285_n-removebg-preview (1).png"
img = read_file(filename)


# In[92]:


#Create Edge Mask


# In[93]:


def edge_mask(img , line_size, blur_value):
    """
    input: Gray Scale Image
    output: Edges og Images
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.medianBlur(gray, blur_value)
    
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)
    
    return edges


# In[94]:


line_size, blur_value = 7,7
edges = edge_mask(img , line_size, blur_value)

plt.imshow(edges, cmap = "binary")
plt.show()


# In[95]:


# Reduce the Color Palette


# In[96]:


def color_quant(img, k):
    # Transform the image
    data = np.float32(img).reshape((-1, 3))
    # Determine Criteria
    criteria = (cv2.TERM_CRITERIA_EPS+ cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    # Implementing K-Means
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    
    return result


# In[97]:


img = color_quant(img, k = 15)

plt.imshow(img)
plt.show()


# In[98]:


# Reduce the noise

blurred = cv2.bilateralFilter(img, d = 10, sigmaColor = 200, sigmaSpace = 200)

plt.imshow(img)
plt.show()


# In[99]:


# Combine Edge Mask with the Quantized Image


# In[100]:


def cartoon(blurred):
    c = cv2.bitwise_and(blurred, blurred, mask = edges)
    plt.imshow(c)
    plt.show()


# In[101]:


img = read_file(filename)
cartoon(blurred)

