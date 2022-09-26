# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 20:02:25 2022

@author: LENOVO
"""
from test_functions import char_function, test_function
import pylab
from matplotlib import pyplot as plt
from SC_CNN_2 import SC_CNN
from Q_p_as_fractal import Christiakov_emmending
import numpy as np


def f(x):
    return 0.5 * (abs(x + 1) - abs(x - 1))


def SC_CNN_plot_fractal(J, A, B, U, Z,
                        X_0, t, delta_t, Z_k,
                        m, s, Name,
                        output=f, mu=1,
                        screen_shot=True,
                        size=(8, 8),
                        resolution=300,
                        size_points=0.5):
    """


    Parameters
    ----------
    A : TYPE: function
        DESCRIPTION:CNN Feedback with p-adic radial functions or test function.
    B : TYPE: fuction
        DESCRIPTION: CNN Feedforward with p-adic radial functions or test function.
    U : TYPE: function
        DESCRIPTION: CNN Input with p-adic values.
    Z : TYPE: CNN Threshold with p-adic values.
        DESCRIPTION.
    X_0 : TYPE: function
        DESCRIPTION: CNN Initial datum  with p-adic values.
    time : TYPE: float or Int
        DESCRIPTION: maximum time to the parameter time.
    Delta_t : TYPE: float or Int
        DESCRIPTION: The step size for the numerical approximation.
    p : TYPE: Int
        DESCRIPTION: Prime number for the p-adic CNN.
    K : TYPE: Int
        DESCRIPTION: The parameter of aproximation to the CNN respect to Q_p.
    Name : TYPE: str
        DESCRIPTION: The standar name which will be used to save the images and .txt.

    Returns: Nonte
    -------
    TYPE: None
        DESCRIPTION: Plot the State  X(x,t), the Ouput Y(x,t), the Fourier transform in the State X(x,t)
                     and the heat maps of the Kernels A(x,y), B(x,y).
                     Also saves this images with the name of the file and
                     create a .txt to save the parameters A,B,U,Z,X_0.
    """
    """Fractal emmending"""
    x_position, y_position = Christiakov_emmending(Z_k, m, s)

    """A1 is a dictionary with key as time and values a vector
    in numpy as solution values """
    sc_cnn = SC_CNN()
    A1 = sc_cnn.solution(J, A, B, U, Z, X_0, t,
                         delta_t, Z_k, output=output, mu=mu)
    """ Max and Min """
    min_0 = np.min(A1[0])
    max_0 = np.max(A1[0])
    for t in A1.keys():
        min_1 = np.min(A1[t])
        max_1 = np.max(A1[t])
        if max_0 < max_1:
            max_0 = max_1
        if min_0 > min_1:
            min_0 = min_1

    """Times outputs"""
    if screen_shot == True:
        output_keys = list(A1.keys())
        times_output = [output_keys[int(t0/delta_t)]
                        for t0 in range(int(t)+1)
                        if int(t0/delta_t) < len(output_keys)]
    else:
        times_output = screen_shot
    """ Fractal map with sympy of Funtions X(x,t) """
    for t0 in times_output:
        fig_tree = pylab.figure(figsize=size)
        # Plot X(x,y)
        im = plt.scatter(x_position, y_position,
                         c=A1[t0],
                         s=0.5,
                         vmax=max_0,
                         vmin=min_0)
        plt.xlabel("time "+str(int(t0)))
        plt.title("State X(x,t)")
        """ Plot colorbar. """
        axcolor = fig_tree.add_axes([0.95, 0.1, 0.05, 0.6])
        pylab.colorbar(im, cax=axcolor, format='%.4f')
        # pylab.colorbar(im)

        """Save image """
        #import os
        # Name=os.path.basename(__file__)[7:-3]
        plt.savefig(Name+"_X_fractal_time"+str(int(t0)) + ".png",
                    dpi=resolution, bbox_inches='tight')

    # """----------------------------------------------"""

    # """Show output Y(x,t)"""

    """ fractal map with sympy of Funtions Y(x,t) """
    for t0 in times_output:
        fig_tree = pylab.figure(figsize=size)
        # Plot X(x,y)
        im = plt.scatter(x_position, y_position,
                         c=output(A1[t0]),
                         s=size_points,
                         vmax=1,
                         vmin=-1)
        plt.xlabel("time " + str(int(t0)))
        plt.title("Output Y(x,t)")
        """ Plot colorbar. """
        axcolor = fig_tree.add_axes([0.95, 0.1, 0.05, 0.6])
        pylab.colorbar(im, cax=axcolor, format='%.4f')

        """Save image """
        plt.savefig(Name+"_Y_fractal_time"+str(int(t0)) + ".png",
                    dpi=resolution, bbox_inches='tight')
    # """----------------------------------------------"""

    """ fractal map with sympy of Funtions X_0  and Input U """
    if X_0 == 0 and U == 0:
        X_0_list = np.zeros(len(Z_k))
        U_list = np.zeros(len(Z_k))
    elif X_0 == 0 and U != 0:
        X_0_list = np.zeros(len(Z_k))
        U_list = []
        for a in Z_k:
            U_list.append(U(a))
    elif X_0 != 0 and U == 0:
        U_list = np.zeros(len(Z_k))
        X_0_list = []
        for a in Z_k:
            X_0_list.append(X_0(a))
    else:
        X_0_list = []
        U_list = []
        for a in Z_k:
            X_0_list.append(X_0(a))
            U_list.append(U(a))

    # """Show initial condition X_0"""
    fig_tree = pylab.figure(figsize=size)
    im = plt.scatter(x_position,
                     y_position,
                     c=X_0_list,
                     s=size_points)
    plt.title("Initial condition X_0")
    """ Plot colorbar. """
    axcolor = fig_tree.add_axes([0.95, 0.1, 0.05, 0.6])
    pylab.colorbar(im, cax=axcolor)

    """Save image """
    plt.savefig(Name+"_X_0_fractal.png",
                dpi=resolution, bbox_inches='tight')

    # Plot Input U
    # """Show input U"""
    fig_tree = pylab.figure(figsize=size)
    im = plt.scatter(x_position,
                     y_position,
                     c=U_list,
                     s=size_points)
    plt.title("Input U")
    """ Plot colorbar. """
    axcolor = fig_tree.add_axes([0.95, 0.1, 0.05, 0.6])
    pylab.colorbar(im, cax=axcolor)

    """Save image """
    plt.savefig(Name+"_Initial_condition_fractal.png",
                dpi=resolution, bbox_inches='tight')

    """----------------------------------------------"""
    """ Read me .txt"""
    # Name=os.path.basename(__file__)[7:-3]
    doc = open(Name+"_Readme.txt", "w+")
    from dill.source import getsource
    # J operator
    if type(J) == char_function or type(J) == test_function:
        doc.write("Feedback: "+str(J)+"\r\n")
        # doc.write("\n")
    elif J == 0:
        doc.write("Feedback: "+str(J)+"\r\n")
    else:
        doc.write("Feedback: "+getsource(J)+"\r\n")
        # doc.write("\n")
    # Feedback
    if type(A) == char_function or type(A) == test_function:
        doc.write("Feedback: "+str(A)+"\r\n")
        # doc.write("\n")
    elif A == 0:
        doc.write("Feedback: "+str(A)+"\r\n")
    else:
        doc.write("Feedback: "+getsource(A)+"\r\n")
        # doc.write("\n")
    # Feedforward
    if type(B) == char_function or type(B) == test_function:
        doc.write("Feedforward: "+str(B)+"\r\n")
    elif B == 0:
        doc.write("Feedback: "+str(B)+"\r\n")
    else:
        doc.write("Feedforward: "+getsource(B)+"\r\n")

    # Input
    if type(U) == char_function or type(U) == test_function:
        doc.write("Input: "+str(U)+"\r\n")
    elif U == 0:
        doc.write("Feedback: "+str(U)+"\r\n")
    else:
        doc.write("Input: "+getsource(U)+"\r\n")

    # Threshold
    if type(Z) == char_function or type(Z) == test_function:
        doc.write("Threshold: "+str(Z)+"\r\n")
    elif Z == 0:
        doc.write("Feedback: "+str(Z)+"\r\n")
    else:
        doc.write("Threshold: "+getsource(Z)+"\r\n")

    # Initial datum

    if type(X_0) == char_function or type(X_0) == test_function:
        doc.write("Initial datum: "+str(X_0)+"\r\n")
    elif X_0 == 0:
        doc.write("Feedback: "+str(X_0)+"\r\n")
    else:
        doc.write("Initial datum: "+getsource(X_0)+"\r\n")

    # Non-lineality

    if type(output) == char_function or type(output) == test_function:
        doc.write("Non-lineality: "+str(output)+"\r\n")
    else:
        doc.write("Non-lineality: "+getsource(output)+"\r\n")
    # Parameters

    doc.write("Parameters: \r\n")
    doc.write("time: [0,"+str(t)+"]\r\n")
    doc.write("Step for time: "+str(delta_t)+"\r\n")
    K = Z_k.get_radio()
    p = Z_k.get_prime()
    doc.write("p-adic approximation: K="+str(K)+"\r\n")
    doc.write("Prime: p="+str(p))
    doc.close()

    print("For all our color bars we are taking \
          the format format=%.4f \n \
          Also, we are taking the X scale over \
              the minimum and maximum over all time")
