# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 17:41:39 2022

@author: LENOVO
"""
from skimage.util.shape import view_as_blocks
import numpy as np
import image2test
from SC_CNN_2 import SC_CNN
from matplotlib import pyplot as plt


def f(x):
    return 0.5 * (abs(x + 1) - abs(x - 1))


def SC_CNN_image(image, J, A, B, Z,
                 t, delta_t,
                 Z_k,
                 U=0, X_0=0,
                 output=f, mu=1,
                 As="initial", split_image=True,
                 screem_shot=True,
                 reduce=False):
    if split_image:
        p = Z_k.get_prime()
        K = Z_k.get_radio()
        size = p**(int(K/2))
        sub_image = view_as_blocks(image, (size, size))
        len_times = int(t/delta_t)
        record_aux = dict()
        record_time = dict()
        hight = sub_image.shape[0]
        wide = sub_image.shape[1]
        for i in range(0, hight):
            for j in range(0, wide):
                image = sub_image[i, j]
                image_12test = image2test.imate2test(image, Z_k,
                                                     reduction=False)
                image_12test.fit()
                U_0 = image_12test.get_test()

                # Input
                if As == "input" or As == "both":
                    U = U_0
                # Initial Datum
                if As == "initial" or As == "both":
                    X_0 = U_0
                sc_cnn = SC_CNN()
                Aij = sc_cnn.solution(J, A, B, U, Z, X_0,
                                      t, delta_t, Z_k,
                                      output=output, mu=mu)
                record_time[(i, j)] = list(Aij.keys())
                len_ij = len(Aij.keys())
                if len_ij < len_times:
                    len_times = len_ij
                record_aux[(i, j)] = Aij
        record = dict()
        scheme_image = image2test.imate2test(np.zeros([size, size]), Z_k,
                                             reduction=False)
        scheme_image.fit()
        if screem_shot:
            times_position = [int(t0/delta_t) for t0 in range(int(t)+1)
                              if int(t0/delta_t) < len_times]
        elif not screem_shot:
            times_position = [t for t in range(len_times)]
        time_output = []
        for t_position in times_position:
            time_output_aux = []
            for i in range(hight):
                for j in range(wide):
                    t = record_time[(i, j)][t_position]
                    time_output_aux.append(t)
                    sub_image[i, j] = scheme_image.\
                        inverse_transform(record_aux[(i, j)][t])
            time_output.append(min(time_output_aux))
            image_0 = [np.concatenate(sub_image[i],
                                      axis=1)
                       for i in range(hight)]
            record[t] = np.concatenate(image_0, axis=0)

        """ Image plot """
        if type(delta_t) == int:
            decimal_place = 1
        else:
            decimal_place = len(str(delta_t).split(".")[1])
        for t in time_output:
            plt.imshow(record[t], cmap='gray')
            plt.title("X at Time " + str(round(t, decimal_place)))
            plt.show()

        for t in time_output:
            plt.imshow(output(record[t]), cmap='gray')
            plt.title("Y at Time " + str(round(t, decimal_place)))
            plt.show()
    elif not split_image:
        image_12test = image2test.imate2test(image, Z_k,
                                             reduction=reduce)
        image_12test.fit()
        U_0 = image_12test.get_test()
        # Input
        if As == "input" or As == "both":
            U = U_0
        # Initial Datum
        if As == "initial" or As == "both":
            X_0 = U_0
        sc_cnn = SC_CNN()
        A1 = sc_cnn.solution(J, A, B, U, Z, X_0,
                             t, delta_t, Z_k,
                             output=output, mu=mu)
        if screem_shot:
            output_keys = list(A1.keys())
            times_output = [output_keys[int(t0/delta_t)]
                            for t0 in range(int(t)+1)
                            if int(t0/delta_t) < len(output_keys)]
        elif not screem_shot:
            times_output = list(A1.keys())

        """ Image plot """
        if type(delta_t) == int:
            decimal_place = 1
        else:
            decimal_place = len(str(delta_t).split(".")[1])
        for t in times_output:
            image_reves = image_12test.\
                inverse_transform(A1[t])
            plt.imshow(image_reves, cmap='gray')
            plt.title("X at Time " + str(round(t, decimal_place)))
            plt.show()

        for t in times_output:
            image_reves = image_12test.\
                inverse_transform(output(A1[t]))
            plt.imshow(image_reves, cmap='gray')
            plt.title("Y at Time " + str(round(t, decimal_place)))
            plt.show()
