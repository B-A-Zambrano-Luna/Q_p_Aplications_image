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

import Q_p
import test_functions as test
from SC_CNN_image_A import SC_CNN_image_A
# import sys
# sys.path.insert(1, 'D:/Documents/un/python/SC-CNN')


delta_t = 0.01
t = 3+delta_t
p = 3
K = 2
Z_k = Q_p.Z_N_group(p, K)


def f(x):
    return 0.5*(abs(x+1) - abs(x-1))

# def f(x):
#     return x


""" Import and tranform image """

gray_astro = rgb2gray(cat())
image = resize(img_as_float(camera()), (100, 100))

plt.imshow(image, cmap='gray')
plt.title("Original")
plt.show()

"""Noise """


image = random_noise(image,
                     mode='gaussian',
                     seed=0,
                     var=0.005)
"""Normalize to p^K"""

image = (2 * image)-1

plt.imshow(image, cmap='gray')
plt.title("Noise")
plt.show()

""" Restoired images with
total variation, bilateral,
and wavelet denoising filters"""

""" Total variation"""
tv_image = denoise_tv_chambolle(image, weight=0.05)
plt.imshow(tv_image, cmap="gray")
plt.title("Total variation denoised")
plt.show()
"""Bilateral"""
Bi_image = denoise_bilateral(image, sigma_color=0.05, sigma_spatial=15)
plt.imshow(Bi_image, cmap="gray")
plt.title("Bilatela denoised")
plt.show()
"""Wavelet"""
W_image = denoise_wavelet(image, rescale_sigma=True)
plt.imshow(W_image, cmap="gray")
plt.title("Wavelet denoised")
plt.show()
""" J, A, B,  and Z  operators """
# J operator
image = image

alpha = 1
a = -1
gamma_p = (1-p**alpha)/(1-p**(-alpha-1))


def D_aux(x):
    if x == 0:
        return (-1)*(gamma_p*(1-p**(-1)))\
            * ((1-p**(-K*alpha))/(1-p**(-alpha)))
    else:
        return (gamma_p)\
            / ((p**K)*Q_p.norm_p(x, p)**(alpha+1))


norm_j = 0
for i in Z_k:
    norm_j += D_aux(i)


def D(x):
    return a*D_aux(x)/norm_j


# Feedback


b = 6
# b = 1*p**K


def A(x):
    if Q_p.norm_p(x, p) == 0:
        return (-b)*(p**(K)-1)*0
    elif 0 < Q_p.norm_p(x, p) <= 1:
        return (-b)*(-1)
    else:
        return 0
# A = 2 * test.char_function(0, -K, p)

# A = (-b)*(p**K)*test.char_function(0, -K, p) \
#     + (b)*test.char_function(0, 0, p)

# Feedforward
# c = -b


def B(x):
    if Q_p.norm_p(x, p) == 0:
        return b*(p**(K)-1)*0
    elif 0 < Q_p.norm_p(x, p) <= 1:
        return b*(-1)
    else:
        return 0

# B = (b*(p**K))*test.char_function(0, -K, p) \
#     + (-1*b)*test.char_function(0, 0, p)


# Input


U = image.copy()

# Threshold


Z = 0


start = time.time()
mu = b
results = SC_CNN_image_A(image, D_aux, A, B, U, Z,
                         t, delta_t,
                         Z_k,
                         output=f, mu=mu,
                         As="input",
                         screem_shot=True,
                         with_activation=False)
end = time.time()
print("Time ejecution ", end-start)
