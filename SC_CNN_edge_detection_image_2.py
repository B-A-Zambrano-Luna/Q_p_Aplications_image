"""
Created on Fri Mar 25 18:05:43 2022

@author: LENOVO
"""
from skimage import filters
import time
from matplotlib import pyplot as plt
from skimage.transform import resize
from skimage.data import binary_blobs, \
    camera, horse, astronaut, cat
from skimage.util import random_noise, img_as_float
from skimage.color import rgb2gray
from skimage.io import imread
import Q_p
import test_functions as test
from SC_CNN_image import SC_CNN_image
# import sys
# sys.path.insert(1, 'D:/Documents/un/python/\
#                     Q_p_Aplications_image/images/\
#                         Diffusion')
root = "D:/Documents/un/python/Q_p_Aplications_image/images/Diffusion/"
image_name = root+'2.png'
""" Import and tranform image """
image = imread(image_name, as_gray=True)

delta_t = 0.05
t = 7
p = 3
K = 4
Z_k = Q_p.Z_N_group(p, K)


def f(x):
    return 0.5 * (abs(x + 1) - abs(x - 1))


""" Import and tranform image """
#image = rgb2gray(image)
image = img_as_float(resize(image, (243, 243)))
# # image = img_as_float(resize(binary_blobs(), (81, 81)))
# image = random_noise(image,
#                      mode='gaussian',
#                      seed=0,
#                      var=0.001)
# image = filters.roberts(image)
image = (image * 2 - 1)
plt.imshow(image, cmap='gray')
plt.show()

""" J, A, B,  and Z  operators """
# J operator


J = 0


# Feedback
#A = p**(K-4)*2*test.char_function(0, -K+2, p)
A = p**K*2*test.char_function(0, -K, p)
# Feedforward


def B(x):
    if Q_p.norm_p(x, p) == 0:
        return p**K*(p**(K-2)-1)
    elif 0 < Q_p.norm_p(x, p) <= p**(-K+2):
        return p**K*(-1)
    else:
        return 0


# Threshold
def Z(x):
    return -1


start = time.time()
SC_CNN_image(image, J, A, B, Z,
             t, delta_t,
             Z_k,
             U=0, X_0=0,
             As="input", split_image=True,
             screem_shot=True,
             reduce=False)
end = time.time()
print("Time ejecution ", end-start)
