# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 17:41:39 2022

@author: LENOVO
"""
from copy import deepcopy
import numpy as np
from matplotlib import pyplot as plt
import Q_p
import image2test


def vectorize_function(f, Z_k):
    """
    Parameters
    ----------
    f : TYPE
        DESCRIPTION.
    Z_k : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    vector = []
    for a in Z_k:
        vector.append(f(a))
    return np.array(vector)


def p_adic_Convolution_matrix(g, Z_k):
    p = Z_k.get_prime()
    k = Z_k.get_radio()
    vect_g = []
    index = np.array([i for i in Z_k], dtype=int)
    Dic_index = dict()
    for a in index:
        Dic_index[a] = g(a)
    vect_g = np.array(vect_g)
    matrix_g = np.zeros((p**k, p**k))
    num_row = 0
    for j in index:
        New_Values = (index - j) % p**k
        matrix_g[num_row] = np.array([Dic_index[a] for a in New_Values])
        num_row += 1
    return p**(-k)*matrix_g


def p_adic_Convolution(f, matrix_g, Z_k):
    vect_f = []
    index = np.array([i for i in Z_k], dtype=int)
    for a in index:
        vect_f.append(f(a))
    return np.matmul(matrix_g, vect_f)


def f(x):
    return 0.5 * (abs(x + 1) - abs(x - 1))


def index_image(n, m):
    image = []
    for i in range(n):
        for j in range(m):
            image.append((i, j))
    image = np.array(image, dtype="i,i")
    image.resize((n, m))
    return image


def SC_CNN_image_A(image, J, A, B, U, Z,
                   t, delta_t,
                   Z_k,
                   output=f, mu=1,
                   As="initial",
                   screem_shot=True,
                   with_activation=False):
    p = Z_k.get_prime()
    k = Z_k.get_radio()
    k_0 = int(k/2)
    rows, colm = image.shape

    def dx(delta_t, x, J_matrix, A_matrix,
           Bu, Z, mu, output):
        Jx = np.matmul(J_matrix, x)
        Ax = np.matmul(A_matrix, output(x))
        dx = mu*x + Jx + Ax + Bu + Z
        return dx

    """ Squards smallest"""

    if J == 0:
        J_matrix = np.zeros((p**k, p**k))
    else:
        J_matrix = p_adic_Convolution_matrix(J, Z_k)
    if A == 0:
        A_matrix = np.zeros((p**k, p**k))
    else:
        A_matrix = p_adic_Convolution_matrix(A, Z_k)
    if As == "input":
        if B == 0:
            B_matrix = np.zeros((p**k, p**k))
        else:
            B_matrix = p_adic_Convolution_matrix(B, Z_k)
    else:
        if B == 0:
            BU = np.zeros(p**k)
        else:
            B_matrix = p_adic_Convolution_matrix(B, Z_k)
            BU = p_adic_Convolution(U, B_matrix, Z_k)

    if Z == 0:
        z_vect = np.zeros(p**k)
    else:
        z_vect = vectorize_function(Z, Z_k)

    """ Euler method"""

    k_1 = int(k/2)
    i = int(p/2)
    j = int(p/2)
    dt = 0
    results = dict()
    temple_im = index_image(3, 3)
    if As == "input":
        image_input = image.copy()

    tree_sub_image_tem = image2test.\
        imate2test(temple_im, Z_k)
    tree_sub_image_tem.fit()
    while dt <= t:
        # Inner images
        step_i = 1
        step_j = 1
        image_copy = image.copy()

        while step_i < rows-1:
            step_j = 1
            while step_j < colm-1:

                sub_image = image[step_i-1:step_i+2,
                                  step_j-1:step_j+2]
                values_sub_image = tree_sub_image_tem.\
                    get_values_with(sub_image)
                if As == "input":
                    sub_image_input = image_input[step_i-1:step_i+2,
                                                  step_j-1:step_j+2]

                    values_sub_image_input = tree_sub_image_tem.\
                        get_values_with(sub_image_input)
                    U = values_sub_image_input
                    BU = np.matmul(B_matrix, U)
                new_values_image = values_sub_image\
                    + dt * dx(dt, values_sub_image, J_matrix,
                              A_matrix, BU, z_vect, mu,
                              output)
                # print(dx(dt, values_sub_image, J_matrix,
                #          A_matrix, BU, z_vect, mu,
                #          output))
                new_sub_image = tree_sub_image_tem\
                    .inverse_transform(new_values_image)
                image_copy[step_i, step_j] = new_sub_image[1, 1]

                step_j += 1
            step_i += 1
        # Boundary
        # Cornerls
        # Cornel_0
        sub_image = image[0:p**k_1,
                          0:p**k_1]

        values_sub_image = tree_sub_image_tem.\
            get_values_with(sub_image)
        if As == "input":
            sub_image_input = image_input[0:p**k_1,
                                          0:p**k_1]

            values_sub_image_input = tree_sub_image_tem.\
                get_values_with(sub_image_input)
            U = values_sub_image_input
            BU = np.matmul(B_matrix, U)
        new_values_image = values_sub_image\
            + dt * dx(dt, values_sub_image, J_matrix,
                      A_matrix, BU, z_vect, mu,
                      output)
        new_sub_image = tree_sub_image_tem\
            .inverse_transform(new_values_image)
        image_copy[0, 0] = new_sub_image[0, 0]
        # Cornel_1
        sub_image = image[0:p**k_1,
                          colm-p**k_1:colm]

        values_sub_image = tree_sub_image_tem.\
            get_values_with(sub_image)
        if As == "input":
            sub_image_input = image_input[0:p**k_1,
                                          colm-p**k_1:colm]

            values_sub_image_input = tree_sub_image_tem.\
                get_values_with(sub_image_input)
            U = values_sub_image_input
            BU = np.matmul(B_matrix, U)
        new_values_image = values_sub_image\
            + dt * dx(dt, values_sub_image, J_matrix,
                      A_matrix, BU, z_vect, mu,
                      output)
        new_sub_image = tree_sub_image_tem\
            .inverse_transform(new_values_image)
        image_copy[0, colm-1] = new_sub_image[0, 2]
        # Cornel_3
        sub_image = image[rows-p**k_1:rows,
                          0:p**k_1]

        values_sub_image = tree_sub_image_tem.\
            get_values_with(sub_image)
        if As == "input":
            sub_image_input = image_input[rows-p**k_1:rows,
                                          0:p**k_1]

            values_sub_image_input = tree_sub_image_tem.\
                get_values_with(sub_image_input)
            U = values_sub_image_input
            BU = np.matmul(B_matrix, U)
        new_values_image = values_sub_image\
            + dt * dx(dt, values_sub_image, J_matrix,
                      A_matrix, BU, z_vect, mu,
                      output)
        new_sub_image = tree_sub_image_tem\
            .inverse_transform(new_values_image)
        image_copy[rows-1, 0] = new_sub_image[2, 0]
        # Cornel_4
        sub_image = image[rows-p**k_1:rows,
                          colm-p**k_1:colm]
        # tree_sub_image = image2test.\
        #     imate2test(sub_image, Z_k)
        # tree_sub_image.fit()
        # values_sub_image = tree_sub_image.\
        #     get_values()
        values_sub_image = tree_sub_image_tem.\
            get_values_with(sub_image)
        if As == "input":
            sub_image_input = image_input[rows-p**k_1:rows,
                                          colm-p**k_1:colm]
            # tree_sub_image_input = image2test.\
            #     imate2test(sub_image_input, Z_k)
            # tree_sub_image_input.fit()
            # values_sub_image_input = tree_sub_image_input.\
            #     get_values()
            values_sub_image_input = tree_sub_image_tem.\
                get_values_with(sub_image_input)
            U = values_sub_image_input
            BU = np.matmul(B_matrix, U)
        new_values_image = values_sub_image\
            + dt * dx(dt, values_sub_image, J_matrix,
                      A_matrix, BU, z_vect, mu,
                      output)
        new_sub_image = tree_sub_image_tem\
            .inverse_transform(new_values_image)
        image_copy[rows-1, colm-1] = new_sub_image[2, 2]
        # Edges_1
        for j in range(1, colm-1):
            sub_image = image[0:p**k_1,
                              j-1:j+2]
            # tree_sub_image = image2test.\
            #     imate2test(sub_image, Z_k)
            # tree_sub_image.fit()
            # values_sub_image = tree_sub_image.\
            #     get_values()
            values_sub_image = tree_sub_image_tem.\
                get_values_with(sub_image)
            if As == "input":
                sub_image_input = image_input[0:p**k_1,
                                              j-1:j+2]
                # tree_sub_image_input = image2test.\
                #     imate2test(sub_image_input, Z_k)
                # tree_sub_image_input.fit()
                # values_sub_image_input = tree_sub_image_input.\
                #     get_values()
                values_sub_image_input = tree_sub_image_tem.\
                    get_values_with(sub_image_input)
                U = values_sub_image_input
                BU = np.matmul(B_matrix, U)
            new_values_image = values_sub_image\
                + dt * dx(dt, values_sub_image, J_matrix,
                          A_matrix, BU, z_vect, mu,
                          output)
            new_sub_image = tree_sub_image_tem\
                .inverse_transform(new_values_image)
            image_copy[0, j] = new_sub_image[0, 1]

        # Edges_2
        for i in range(1, rows-1):
            sub_image = image[i-1:i+2,
                              colm-p**k_1:colm]
            # tree_sub_image = image2test.\
            #     imate2test(sub_image, Z_k)
            # tree_sub_image.fit()
            # values_sub_image = tree_sub_image.\
            #     get_values()
            values_sub_image = tree_sub_image_tem.\
                get_values_with(sub_image)
            if As == "input":
                sub_image_input = image_input[i-1:i+2,
                                              colm-p**k_1:colm]
                # tree_sub_image_input = image2test.\
                #     imate2test(sub_image_input, Z_k)
                # tree_sub_image_input.fit()
                # values_sub_image_input = tree_sub_image_input.\
                #     get_values()
                values_sub_image_input = tree_sub_image_tem.\
                    get_values_with(sub_image_input)
                U = values_sub_image_input
                BU = np.matmul(B_matrix, U)
            new_values_image = values_sub_image\
                + dt * dx(dt, values_sub_image, J_matrix,
                          A_matrix, BU, z_vect, mu,
                          output)
            new_sub_image = tree_sub_image_tem\
                .inverse_transform(new_values_image)
            image_copy[i, colm-1] = new_sub_image[1, 2]

        # Edges_3
        for j in range(1, colm-1):
            sub_image = image[rows-p**k_1:rows,
                              j-1:j+2]
            # tree_sub_image = image2test.\
            #     imate2test(sub_image, Z_k)
            # tree_sub_image.fit()
            # values_sub_image = tree_sub_image.\
            #     get_values()
            values_sub_image = tree_sub_image_tem.\
                get_values_with(sub_image)
            if As == "input":
                sub_image_input = image_input[rows-p**k_1:rows,
                                              j-1:j+2]
                # tree_sub_image_input = image2test.\
                #     imate2test(sub_image_input, Z_k)
                # tree_sub_image_input.fit()
                # values_sub_image_input = tree_sub_image_input.\
                #     get_values()
                values_sub_image_input = tree_sub_image_tem.\
                    get_values_with(sub_image_input)
                U = values_sub_image_input
                BU = np.matmul(B_matrix, U)
            new_values_image = values_sub_image\
                + dt * dx(dt, values_sub_image, J_matrix,
                          A_matrix, BU, z_vect, mu,
                          output)
            new_sub_image = tree_sub_image_tem\
                .inverse_transform(new_values_image)
            image_copy[rows-1, j] = new_sub_image[2, 1]

        # Edges_4
        for i in range(1, rows-1):
            sub_image = image[i-1:i+2,
                              0:p**k_1]
            # tree_sub_image = image2test.\
            #     imate2test(sub_image, Z_k)
            # tree_sub_image.fit()
            # values_sub_image = tree_sub_image.\
            #     get_values()
            values_sub_image = tree_sub_image_tem.\
                get_values_with(sub_image)
            if As == "input":
                sub_image_input = image_input[i-1:i+2,
                                              0:p**k_1]
                # tree_sub_image_input = image2test.\
                #     imate2test(sub_image_input, Z_k)
                # tree_sub_image_input.fit()
                # values_sub_image_input = tree_sub_image_input.\
                #     get_values()
                values_sub_image_input = tree_sub_image_tem.\
                    get_values_with(sub_image_input)
                U = values_sub_image_input
                BU = np.matmul(B_matrix, U)
            new_values_image = values_sub_image\
                + dt * dx(dt, values_sub_image, J_matrix,
                          A_matrix, BU, z_vect, mu,
                          output)
            new_sub_image = tree_sub_image_tem\
                .inverse_transform(new_values_image)
            image_copy[i, 0] = new_sub_image[1, 0]
        results[dt] = image_copy
        image = image_copy
        # print(dt)
        dt += delta_t

    if screem_shot:
        output_keys = list(results.keys())
        times_output = [output_keys[int(t0/delta_t)]
                        for t0 in range(int(t)+1)
                        if int(t0/delta_t) < len(output_keys)]
        if type(delta_t) == int:
            decimal_place = 1
        else:
            decimal_place = len(str(delta_t).split(".")[1])
        for t in times_output:
            plt.imshow(results[t], cmap='gray')
            plt.title("X at Time " + str(round(t, decimal_place)))
            plt.show()
        if with_activation:
            for t in times_output:
                plt.imshow(output(results[t]), cmap='gray')
                plt.title("Y at Time " + str(round(t, decimal_place)))
                plt.show()
    elif not screem_shot:
        if type(delta_t) == int:
            decimal_place = 1
        else:
            decimal_place = len(str(delta_t).split(".")[1])
        for t in results:
            plt.imshow(results[t], cmap='gray')
            plt.title("X at Time " + str(round(t, decimal_place)))
            plt.show()
        if with_activation:
            for t in results:
                plt.imshow(output(results[t]), cmap='gray')
                plt.title("Y at Time " + str(round(t, decimal_place)))
                plt.show()

    return results
