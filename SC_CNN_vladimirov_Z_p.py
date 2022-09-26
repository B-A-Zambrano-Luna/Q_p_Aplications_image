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
from SC_CNN_image_A import SC_CNN_image_A
# import sys
# sys.path.insert(1, 'D:/Documents/un/python/SC-CNN')


delta_t = 0.05
t = 3+delta_t
p = 3
K = 2
Z_k = Q_p.Z_N_group(p, K)


# def f(x):
#     return 0.5*(abs(x) - abs(x-1)+1)

def f(x):
    return x


""" Import and tranform image """

gray_astro = rgb2gray(cat())
image = resize(img_as_float(camera()), (200, 200))
# image = random_noise(image,
#                      mode='gaussian',
#                      seed=0,
#                      var=0.001)
# image = resize(img_as_float(camera()), (242, 242))
image = image * p**(K)
# image = camera()
plt.imshow(image, cmap='gray')
plt.title("Original")
plt.show()

# image_g = gaussian(image, sigma=2.5)
# plt.imshow(image_g, cmap='gray')
# plt.title("Image with filter Gaussian")
# plt.show()
""" J, A, B,  and Z  operators """
# J operator

alpha = 1
a = -1
gamma_p = (1-p**alpha)/(1-p**(-alpha-1))

rho = (1-p**(-1))*(p**(K*alpha)-1)/(1-p**(-alpha-1))


def D(x):
    if x != 0:
        return a*gamma_p*(p**(-K))/(Q_p.norm_p(x, p)**(alpha+1))
    else:
        return a*rho


start = time.time()
results = SC_CNN_image_A(image, D, 0, 0, 0, 0,
                         t, delta_t,
                         Z_k,
                         output=f, mu=0,
                         As="initial",
                         screem_shot=True,
                         with_activation=False)
end = time.time()
print("Time ejecution ", end-start)

max_time = 0
times = list(results.keys())
for t in times:
    if results[t].max() <= 1+delta_t+0.1*delta_t and results[t].min() >= -delta_t-0.1*delta_t:
        max_time = t
print("maximal time for stochatic meaning= ", max_time)

"""
alpha = 1 -> 3.8
alpha = 0.5 -> 5 or more
"""
