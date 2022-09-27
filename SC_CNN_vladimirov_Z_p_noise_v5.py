"""
Created on Fri Mar 25 18:05:43 2022

@author: LENOVO
"""
import time
from matplotlib import pyplot as plt
from skimage.transform import resize
from skimage.data import binary_blobs, \
    camera, horse, astronaut, cat
from skimage.util import random_noise, img_as_float
from skimage.color import rgb2gray
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet)
from skimage.io import imread
import Q_p
import test_functions as test
from SC_CNN_image_A import SC_CNN_image_A
# import sys
# sys.path.insert(1, 'D:/Documents/un/python/SC-CNN')


delta_t = 0.05
t = 1+delta_t
p = 3
K = 2
Z_k = Q_p.Z_N_group(p, K)






def f(x):
    return 0.5*(abs(x+1) - abs(x-1))


""" Import and tranform image """



image = imread("saint_Paul.png")
image = rgb2gray(image)
image = resize(image, (400, 400))
# image = resize(camera(),(400,400))

plt.imshow(image, cmap='gray')
plt.title("Original")
plt.show()

"""Noise """


image = random_noise(image,
                     mode='gaussian',
                     seed=0,
                     var=0.05)

plt.imshow(image, cmap='gray')
plt.title("Noise")
plt.show()

"""Normalize"""

image = (2 * image)-1

""" Restoired images with
total variation, bilateral,
and wavelet denoising filters"""


plt.show()
""" J, A, B,  and Z  operators """

# J operator

alpha = 1.15
a = -1
gamma_p = (1-p**alpha)/(1-p**(-alpha-1))

rho = (1-p**(-1))*(p**(K*alpha)-1)/(1-p**(-alpha-1))


def D(x):
    if x != 0:
        return a*gamma_p*(p**(-K))/(Q_p.norm_p(x, p)**(alpha+1))
    else:
        return a*rho


# Feedback


A = (1.25*2)*test.char_function(0, -K, p)


# Feedforward
c = 1.25


def B(x):
    if Q_p.norm_p(x, p) == 0:
        return c*8
    elif 0 < Q_p.norm_p(x, p) <= 1:
        return c*(-1)
    else:
        return 0


# Input


U = image.copy()

# Threshold


def Z(x):
    return 0

mu = 1.25
start = time.time()
results = SC_CNN_image_A(image, D, A, B, U, Z,
                         t, delta_t,
                         Z_k,
                         output=f, mu=0,
                         As="input",
                         screem_shot=True,
                         with_activation=False)
end = time.time()
print("Time ejecution ", end-start)

# max_time = 0
# times = list(results.keys())
# for t in times:
#     if results[t].max() <= 1+delta_t+0.1*delta_t and results[t].min() >= -delta_t-0.1*delta_t:
#         max_time = t
# print("maximal time for stochatic meaning= ", max_time)

"""
alpha = 1 -> 3.8
alpha = 0.5 -> 5 or more
"""
