# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 09:18:09 2022

@author: LENOVO
"""

import numpy as np
from cv2.ximgproc import anisotropicDiffusion
from matplotlib import pyplot as plt
from skimage.transform import resize
from skimage.data import binary_blobs, \
    camera, horse, astronaut, cat
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.util import random_noise,\
    img_as_float, img_as_ubyte
import cv2
image_saint = 'D:/Documents/un/python/SC-CNN/image_example/edge_detection/grey image/saint_Paul.png'
""" Import and tranform image """
image_read = imread(image_saint)
image = rgb2gray(image_read)
image = resize(image, (400, 400))
image = random_noise(image,
                     mode='gaussian',
                     seed=0,
                     var=0.05)
image = img_as_ubyte(image)
plt.imshow(image, cmap='gray')
plt.title("noise")
plt.show()
n, m = image.shape
img = np.zeros((n, m, 3), dtype=np.uint8)
img[:, :, 0] = image
img[:, :, 1] = image
img[:, :, 2] = image

alpha = 0.075
K = 0.125
niters = 100

image_2 = anisotropicDiffusion(img, alpha, K, niters)

plt.imshow(image_2, cmap="gray")
plt.title("Perona_Malik\n alpha={}, K={}, ninters={}".format(alpha, K, niters))
plt.show()
