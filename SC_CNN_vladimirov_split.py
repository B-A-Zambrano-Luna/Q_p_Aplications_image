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
import Q_p
import test_functions as test
from SC_CNN_image import SC_CNN_image
# import sys
# sys.path.insert(1, 'D:/Documents/un/python/SC-CNN')


delta_t = 0.05
t = 6
p = 3
K = 2
Z_k = Q_p.Z_N_group(p, K)


# def f(x):
#     return 0.5*(abs(x) - abs(x-1)+1)

def f(x):
    return x


""" Import and tranform image """

#image = resize(img_as_float(camera()), (81, 81))
image = resize(img_as_float(camera()), (150, 150))
image = random_noise(image,
                     mode='gaussian',
                     seed=0,
                     var=0.005)
#image = resize(img_as_float(camera()), (242, 242))
image = image * 2 - 1
#image = camera()
plt.imshow(image, cmap='gray')
plt.title("Original")
plt.show()

# image_g = gaussian(image, sigma=2.5)
# plt.imshow(image_g, cmap='gray')
# plt.title("Image with filter Gaussian")
# plt.show()
""" J, A, B,  and Z  operators """
# J operator

alpha = 0.9
a = -0.15
gamma_p = (1-p**alpha)/(1-p**(-alpha-1))


def D(x):
    if x == 0:
        return (p**K) * (-1)*(a*gamma_p*(1-p**(-1)))\
            * ((1-p**(-K*alpha))/(1-p**(-alpha)))
    else:
        return (p**K) * (a*gamma_p)\
            / ((p**K)*Q_p.norm_p(x, p)**(alpha+1))


# Feedback
# alpha_1 = -1


# def A(x):
#     if x == 0:
#         return (p**K) * (-1)*(a*gamma_p*(1-p**(-1)))\
#             * ((1-p**(-K*alpha_1))/(1-p**(-alpha_1)))
#     else:
#         return (p**K) * (a*gamma_p)\
#             / ((p**K)*Q_p.norm_p(x, p)**(alpha_1+1))
A = 0
# Feedforward


# def B(x):
#     if Q_p.norm_p(x, p) == 0:
#         return p**K*(p**(K-2)-1)
#     elif 0 < Q_p.norm_p(x, p) <= p**(-K+2):
#         return p**K*(-1)
#     else:
#         return 0
def B(x):
    return -5.5

# 7 no es, es muy grande
# Es mayor o igual a 5
# Input


U = image.copy()

# Threshold
Z = 0


start = time.time()
SC_CNN_image(image, D, A, B, Z,
             t, delta_t,
             Z_k,
             U, X_0=image,
             mu=-30,
             As="both", split_image=True,
             screem_shot=True,
             reduce=False,
             output=f)
end = time.time()
print("Time ejecution ", end-start)
