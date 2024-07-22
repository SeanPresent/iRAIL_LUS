#!/usr/bin/env python
# coding: utf-8

# In[7]:


import glob
import numpy as np
import cv2
import PIL
import os
from matplotlib import pyplot as plt


# In[8]:


PNG_dir2  =  glob.glob("/home/Sean/Project/iRail_US/cropped_data/Cropped"+os.sep+"/*/*")
len(PNG_dir2)


# In[ ]:


for i in PNG_dir2:
    img = cv2.imread(i)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    copy_img = img.copy()

    lower = np.array([5,5,5])
    higher = np.array([255, 255, 255])

    mask = cv2.inRange(img, lower, higher)

    cont, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cont_img = cv2.drawContours(img, cont, -1, 255, 3)
    c = max(cont, key = cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255,0),5)
    plt.imshow(img)
    plt.show()

    cropped_image = copy_img[y: y+h ,x: x+w ]
    plt.imshow(cropped_image)
    plt.show()
    cv2.imwrite(i, cropped_image)


# In[ ]:




