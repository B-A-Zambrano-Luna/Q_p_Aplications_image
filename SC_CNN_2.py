# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 15:40:23 2022

@author: LENOVO
"""
import scipy.integrate as sint
import numpy as np


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


def nonlineality(x):
    return 0.5 * (abs(x + 1) - abs(x - 1))


class SC_CNN(object):
    """Image Processing with Cellular Neural Networks (CNN).

    Cellular Neural Networks (CNN) are a parallel computing paradigm that was
    first proposed in 1988. Cellular neural networks are similar to neural
    networks, with the difference that communication is allowed only between
    neighboring units. Image Processing is one of its applications. CNN
    processors were designed to perform image processing; specifically, the
    original application of CNN processors was to perform real-time ultra-high
    frame-rate (>10,000 frame/s) processing unachievable by digital processors.

    This python library is the implementation of CNN for the application of
    Image Processing.


    Attributes:
        n (int): Height of the image.
        m (int): Width of the image.
    """

    def __init__(self):
        """Sets the initial class attributes m (width) and n (height)."""
        self.n = 0  # height (number of rows)
        self.Z_k = []
        self.nonlineality = nonlineality
        self.mu = 1

    def f(self, t, x, J_matrix, A_matrix, Bu, Z):
        """
        Compute the righ hand side of the differential
        equation
        Parameters
        ----------
        t : TYPE
            DESCRIPTION.
        x : TYPE
            DESCRIPTION.
        J_matrix : TYPE numpy narray or 0
            DESCRIPTION. the state-controlled kernel
        A : TYPE numpy narray or 0
            DESCRIPTION.the feedback kernel
        Bu : TYPE numpy array or 0
            DESCRIPTION. The input of the CNN
        Z : TYPE numoy array or 0
            DESCRIPTION. The threshold

        Returns
        -------
        dx : TYPE
            DESCRIPTION.

        """
        mu = self.mu
        Jx = np.matmul(J_matrix, x)
        Ax = np.matmul(A_matrix, self.output(x))
        #print("f(x)", self.output(x))
        #print(x == self.output(x))
        # print("Ax", Ax)
        # print("BU", Bu)
        dx = -mu*x + Jx + Ax + Bu + Z
        # print(mu)
        # print("dx", dx)
        # print("x", x)
        return dx

    def output(self, x):
        """Piece-wise linear sigmoid function.

        Args:
            x : Input to the piece-wise linear sigmoid function.
        """
        nonlineality = self.nonlineality
        return nonlineality(x)

    def solution(self, J, A, B, U, Z,
                 X_0, t, delta_t, Z_k,
                 output=nonlineality, mu=1):
        """


        Parameters
        ----------
        J : TYPE
            DESCRIPTION.
        A : TYPE
            DESCRIPTION.
        B : TYPE
            DESCRIPTION.
        U : TYPE
            DESCRIPTION.
        Z : TYPE
            DESCRIPTION.
        X_0 : TYPE
            DESCRIPTION.
        t : TYPE float
            DESCRIPTION. maximum time of simulation
        delta_t : TYPE float
            DESCRIPTION. stape size for the simulation
        Z_k : TYPE
            DESCRIPTION.

        Returns
        -------
        ode_result : TYPE
            DESCRIPTION.

        """
        self.Z_k = Z_k
        self.nonlineality = output
        self.mu = mu
        p = Z_k.get_prime()
        k = Z_k.get_radio()
        self.n = p**k
        if J == 0:
            J_matrix = np.zeros((p**k, p**k))
        else:
            J_matrix = p_adic_Convolution_matrix(J, Z_k)
        if A == 0:
            A_matrix = np.zeros((p**k, p**k))
        else:
            A_matrix = p_adic_Convolution_matrix(A, Z_k)
        if B == 0 or U == 0:
            BU = np.zeros(p**k)
        else:
            B_matrix = p_adic_Convolution_matrix(B, Z_k)
            BU = p_adic_Convolution(U, B_matrix, Z_k)
        if Z == 0:
            z_vect = np.zeros(p**k)
        else:
            z_vect = vectorize_function(Z, Z_k)
        if X_0 == 0:
            x_0_vect = np.zeros(p**k)
        else:
            x_0_vect = vectorize_function(X_0, Z_k)

        ode_result = {0: x_0_vect}

        ode = sint.ode(self.f) \
            .set_integrator('vode') \
            .set_initial_value(x_0_vect, 0) \
            .set_f_params(J_matrix, A_matrix, BU, z_vect)
        while ode.successful() and ode.t < t:
            ode_result[ode.t+delta_t] = ode.integrate(ode.t + delta_t)
        return ode_result
