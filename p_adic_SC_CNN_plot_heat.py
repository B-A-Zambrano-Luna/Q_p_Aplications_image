# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 12:56:12 2021

@author: LENOVO
"""
from test_functions import char_function, test_function
import numpy as np
import pylab
from scipy.spatial.distance import squareform
import scipy.cluster.hierarchy as sch
from matplotlib import pyplot as plt
from SC_CNN_2 import SC_CNN,\
    p_adic_Convolution_matrix
from Q_p import norm_p


def f(x):
    return 0.5 * (abs(x + 1) - abs(x - 1))


def SC_CNN_plot_heat(J, A, B, U, Z,
                     X_0, t,
                     delta_t, Z_k,
                     Name,
                     output=f, mu=1, all_tree=True,
                     size=(8, 8),
                     resolution=300):
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

    """Parameters"""

    """ Auxiliar functions """
    def X(D, j):
        """D is a dict where D.keys is the time and for fixed t
          D[t] is the a array with the values of p^{-k}, p^{-k+1},ldots, p^{k}
          return a list of X's which represen the function X(p^j,t)
        """
        ans = []
        for t0 in D.keys():
            ans.append(D[t0][j])
        ans = np.array(ans)
        return ans

    def color_matrix(D, Z_k):
        """ Reorder values depending on Z_k"""
        C = []
        for b in range(len(Z_k)):
            C.append(X(D, b))
        return np.array(C)

    """A1 is a dictionary with key as time and values a vector
    in numpy as solution values """
    sc_cnn = SC_CNN()
    A1 = sc_cnn.solution(J, A, B, U, Z, X_0,
                         t, delta_t, Z_k,
                         output=output, mu=mu)
    Y = np.array([b for b in Z_k])
    C_X = color_matrix(A1, Z_k)
    C_Y = []
    for b in range(len(Z_k)):
        C_Y.append(output(X(A1, b)))

    C_Y = np.array(C_Y)

    """Show state X(x,t)"""
    p = Z_k.get_prime()
    K = Z_k.get_radio()
    """ Generate distance matrix. """

    Number = len(Z_k)
    D = np.zeros([Number, Number])
    if A == 0:
        C_A = np.zeros([Number, Number])
    else:
        C_A = p_adic_Convolution_matrix(A, Z_k)
    if B == 0:
        C_B = np.zeros([Number, Number])
    else:
        C_B = p_adic_Convolution_matrix(B, Z_k)
    C_X_0 = np.zeros([Number, Number])
    C_U = np.zeros([Number, Number])
    if J == 0:
        C_J = np.zeros([Number, Number])
    else:
        C_J = p_adic_Convolution_matrix(J, Z_k)
    if X_0 != 0 and U != 0:
        for i in range(Number):
            for j in range(Number):
                D[i, j] = norm_p(Y[i]-Y[j], p)
                C_X_0[i, j] = X_0(Y[j])
                C_U[i, j] = U(Y[j])
    elif X_0 != 0:
        for i in range(Number):
            for j in range(Number):
                D[i, j] = norm_p(Y[i]-Y[j], p)
                C_X_0[i, j] = X_0(Y[j])
    elif U != 0:
        for i in range(Number):
            for j in range(Number):
                D[i, j] = norm_p(Y[i]-Y[j], p)
                C_U[i, j] = U(Y[j])
    else:
        for i in range(Number):
            for j in range(Number):
                D[i, j] = norm_p(Y[i]-Y[j], p)

    condensedD = squareform(D)

    """ Tree levels"""
    with_labels = True
    if K > 3:
        with_labels = False

    # labels
    if with_labels:
        if all_tree:
            level_max = K
        else:
            level_max = 0
        x_levels = []
        x_sticks = []
        for i in range(0, K):
            x_levels.append(p**(-i))
            x_sticks.append("Level " + str(i))
        x_levels.append(0)
        x_sticks.append("Level "+str(K))
    else:
        x_levels = []
        x_sticks = []
    list_Z_k = [a for a in Z_k]
    """x_axis"""
    Xticks = [int(a/delta_t) for a in range(0, t+1)]
    Xticklabels = [t for t in range(t+1)]

    """ y_axis"""
    if with_labels:
        Yticks = range(Number)
        Yticklabels = list_Z_k
        # position heat maps A and B
        first_map_position = [0.04, 0.1, 0.2, 0.6]
        second_map_position = [0.3, 0.74, 0.6, 0.2]
        # position heat maps X_0 and U
        position_tree_input = [0.3, 0.3, 0.6, 0.25]
    else:
        # labels
        Yticks = []
        Yticklabels = []
        # position heat maps A and B
        first_map_position = [0.1, 0.1, 0.2, 0.6]
        second_map_position = [0.3, 0.702, 0.6, 0.2]
        # position heat maps X_0 and U
        position_tree_input = [0.3, 0.25, 0.6, 0.25]
    """ Compute and plot first dendrogram. """

    fig_tree = pylab.figure(figsize=size)

    ax1 = fig_tree.add_axes(first_map_position)
    Y1 = sch.linkage(condensedD, method='centroid')
    Z1 = sch.dendrogram(Y1, orientation='left')
    ax1.set_xticks(x_levels)
    ax1.set_xticklabels(x_sticks, rotation='vertical')
    ax1.set_yticks([])

    """ Heat map with sympy of Funtions X(x,t) """
    axmatrix = fig_tree.add_axes([0.3, 0.1, 0.6, 0.6])
    # Plot X(x,y)
    im = axmatrix.matshow(C_X, aspect='auto', origin='lower')
    axmatrix.set_yticks([])
    plt.xlabel("time t")
    plt.title("State X(x,t)")
    """ Plot colorbar. """
    axcolor = fig_tree.add_axes([0.95, 0.1, 0.05, 0.6])
    pylab.colorbar(im, cax=axcolor, format='%.4f')

    """         Plot axes  """
    axmatrix.set_xticks(Xticks)
    axmatrix.set_xticklabels(Xticklabels, minor=False)
    axmatrix.xaxis.set_label_position('bottom')
    axmatrix.xaxis.tick_bottom()

    pylab.xticks(rotation=-90, fontsize=8)

    axmatrix.set_yticks(Yticks)
    axmatrix.set_yticklabels(Yticklabels, minor=False)
    axmatrix.yaxis.set_label_position('right')

    """Save image """
    #import os
    # Name=os.path.basename(__file__)[7:-3]
    plt.savefig(Name+"_X.png", dpi=resolution, bbox_inches='tight')

    """----------------------------------------------"""

    """Show output Y(x,t)"""

    """ Generate distance matrix. """

    """ Compute and plot first dendrogram. """

    fig_tree = pylab.figure(figsize=size)

    ax1 = fig_tree.add_axes(first_map_position)
    Y1 = sch.linkage(condensedD, method='centroid')
    Z1 = sch.dendrogram(Y1, orientation='left')
    ax1.set_xticks(x_levels)
    ax1.set_xticklabels(x_sticks, rotation='vertical')
    ax1.set_yticks([])

    """ Heat map with sympy of Funtions Y(x,t) """
    axmatrix = fig_tree.add_axes([0.3, 0.1, 0.6, 0.6])
    # Plot X(x,y)
    im = axmatrix.matshow(C_Y, aspect='auto', origin='lower')
    axmatrix.set_yticks([])
    plt.xlabel("time t")
    plt.title("Output Y(x,t)")
    """ Plot colorbar. """
    axcolor = fig_tree.add_axes([0.95, 0.1, 0.05, 0.6])
    pylab.colorbar(im, cax=axcolor, format='%.4f')

    """         Plot axes  """
    axmatrix.set_xticks(Xticks)
    axmatrix.set_xticklabels(Xticklabels, minor=False)
    axmatrix.xaxis.set_label_position('bottom')
    axmatrix.xaxis.tick_bottom()

    pylab.xticks(rotation=-90, fontsize=8)

    axmatrix.set_yticks(Yticks)
    axmatrix.set_yticklabels(Yticklabels, minor=False)
    axmatrix.yaxis.set_label_position('right')

    """Save image """
    #import os
    # Name=os.path.basename(__file__)[7:-3]
    plt.savefig(Name+"_Y.png", dpi=resolution, bbox_inches='tight')

    """----------------------------------------------"""
    """ Heat maps of   J, A, and B"""

    """ Heat map of J"""

    # Compute and plot first dendrogram.
    fig = pylab.figure(figsize=size)
    ax1 = fig.add_axes(first_map_position)
    Y1 = sch.linkage(condensedD, method='centroid')
    Z1 = sch.dendrogram(Y1, orientation='left')
    ax1.set_xticks(x_levels)
    ax1.set_xticklabels(x_sticks, rotation='vertical')
    ax1.set_yticks([])

    # Compute and plot second dendrogram.
    ax2 = fig.add_axes(second_map_position)
    Y2 = sch.linkage(condensedD, method='single')
    Z2 = sch.dendrogram(Y2)
    ax2.set_yticks(x_levels)
    ax2.set_yticklabels(x_sticks, rotation='horizontal')
    ax2.set_xticklabels(Yticklabels, minor=False)
    plt.title("Diffution J(x,y)")

    # Plot distance matrix.
    axmatrix = fig.add_axes([0.3, 0.1, 0.6, 0.6])
    idx1 = Z1['leaves']
    idx2 = Z2['leaves']
    im = axmatrix.matshow(C_J, aspect='auto', origin='lower')
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])

    # Plot colorbar.
    axcolor = fig.add_axes([0.98, 0.1, 0.05, 0.6])
    pylab.colorbar(im, cax=axcolor, format='%.4f')
    # fig.show()

    # Plot axes
    axmatrix.set_xticks(Yticks)
    axmatrix.set_xticklabels(Yticklabels, minor=False)
    axmatrix.xaxis.set_label_position('bottom')
    axmatrix.xaxis.tick_bottom()

    pylab.xticks(rotation=-90, fontsize=8)

    # axmatrix.set_yticks(range(40))
    axmatrix.set_yticks(Yticks)
    axmatrix.set_yticklabels(Yticklabels, minor=False)
    axmatrix.yaxis.set_label_position('right')

    """Save image"""
    #import os
    # Name=os.path.basename(__file__)[7:-3]
    plt.savefig(Name+"_heat_J.png", dpi=resolution, bbox_inches='tight')

    """ Heat map of A"""

    # Compute and plot first dendrogram.
    fig = pylab.figure(figsize=size)
    ax1 = fig.add_axes(first_map_position)
    Y1 = sch.linkage(condensedD, method='centroid')
    Z1 = sch.dendrogram(Y1, orientation='left')
    ax1.set_xticks(x_levels)
    ax1.set_xticklabels(x_sticks, rotation='vertical')
    ax1.set_yticks([])

    # Compute and plot second dendrogram.
    ax2 = fig.add_axes(second_map_position)
    Y2 = sch.linkage(condensedD, method='single')
    Z2 = sch.dendrogram(Y2)
    ax2.set_yticks(x_levels)
    ax2.set_yticklabels(x_sticks, rotation='horizontal')
    ax2.set_xticklabels(Yticklabels, minor=False)
    plt.title("Feedback A(x,y)")

    # Plot distance matrix.
    axmatrix = fig.add_axes([0.3, 0.1, 0.6, 0.6])
    idx1 = Z1['leaves']
    idx2 = Z2['leaves']
    im = axmatrix.matshow(C_A, aspect='auto', origin='lower')
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])

    # Plot colorbar.
    axcolor = fig.add_axes([0.98, 0.1, 0.05, 0.6])
    pylab.colorbar(im, cax=axcolor, format='%.4f')
    # fig.show()

    # Plot axes
    axmatrix.set_xticks(Yticks)
    axmatrix.set_xticklabels(Yticklabels, minor=False)
    axmatrix.xaxis.set_label_position('bottom')
    axmatrix.xaxis.tick_bottom()

    pylab.xticks(rotation=-90, fontsize=8)

    # axmatrix.set_yticks(range(40))
    axmatrix.set_yticks(Yticks)
    axmatrix.set_yticklabels(Yticklabels, minor=False)
    axmatrix.yaxis.set_label_position('right')

    """Save image"""
    #import os
    # Name=os.path.basename(__file__)[7:-3]
    plt.savefig(Name+"_heat_A.png", dpi=resolution, bbox_inches='tight')

    """ Heat map of B"""

    # Compute and plot first dendrogram.
    fig = pylab.figure(figsize=size)
    ax1 = fig.add_axes(first_map_position)
    Y1 = sch.linkage(condensedD, method='centroid')
    Z1 = sch.dendrogram(Y1, orientation='left')
    ax1.set_xticks(x_levels)
    ax1.set_xticklabels(x_sticks, rotation='vertical')
    ax1.set_yticks([])

    # Compute and plot second dendrogram.
    ax2 = fig.add_axes(second_map_position)
    Y2 = sch.linkage(condensedD, method='single')
    Z2 = sch.dendrogram(Y2)
    ax2.set_yticks(x_levels)
    ax2.set_yticklabels(x_sticks, rotation='horizontal')
    ax2.set_xticklabels(Yticklabels, minor=False)
    plt.title("Feedforward B(x,y)")

    # Plot distance matrix.
    axmatrix = fig.add_axes([0.3, 0.1, 0.6, 0.6])
    idx1 = Z1['leaves']
    idx2 = Z2['leaves']
    im = axmatrix.matshow(C_B, aspect='auto', origin='lower')
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])

    # Plot colorbar.
    axcolor = fig.add_axes([0.98, 0.1, 0.05, 0.6])
    pylab.colorbar(im, cax=axcolor, format='%.4f')
    # fig.show()

    # Plot axes
    axmatrix.set_xticks(Yticks)
    axmatrix.set_xticklabels(Yticklabels, minor=False)
    axmatrix.xaxis.set_label_position('bottom')
    axmatrix.xaxis.tick_bottom()

    pylab.xticks(rotation=-90, fontsize=8)

    # axmatrix.set_yticks(range(40))
    axmatrix.set_yticks(Yticks)
    axmatrix.set_yticklabels(Yticklabels, minor=False)
    axmatrix.yaxis.set_label_position('right')

    """Save image"""
    #import os
    # Name=os.path.basename(__file__)[7:-3]
    plt.savefig(Name+"_heat_B.png", dpi=resolution, bbox_inches='tight')

    """----------------------------------------------"""
    """  Heat maps of the input and initial datum"""
    fig = pylab.figure(figsize=size)

    # Compute and plot second dendrogram.
    ax2 = fig.add_axes(position_tree_input)
    Y2 = sch.linkage(condensedD, method='single')
    Z2 = sch.dendrogram(Y2)
    ax2.set_yticks(x_levels)
    ax2.set_yticklabels(x_sticks, rotation='horizontal')
    ax2.set_xticklabels(Yticklabels, minor=False)
    plt.title("Initial datum X_0(x)")

    # Function X_0
    axmatrix = fig.add_axes([0.3, 0.1, 0.6, 0.15])
    idx2 = Z2['leaves']
    im = axmatrix.matshow(C_X_0, aspect='auto', origin='lower')
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])

    # Plot colorbar.
    axcolor = fig.add_axes([0.98, 0.1, 0.05, 0.45])
    pylab.colorbar(im, cax=axcolor, format='%.4f')

    # Plot axes
    axmatrix.set_xticks(Yticks)
    axmatrix.set_xticklabels(Yticklabels, minor=False)
    axmatrix.xaxis.set_label_position('bottom')
    axmatrix.xaxis.tick_bottom()

    pylab.xticks(rotation=-90, fontsize=8)

    """Save image"""
    plt.savefig(Name+"_heat_initial_datum.png",
                dpi=resolution, bbox_inches='tight')

    """----------------------------------------------"""
    """ Heat map of Input"""

    fig = pylab.figure(figsize=size)

    # Compute and plot second dendrogram.
    ax2 = fig.add_axes(position_tree_input)
    Y2 = sch.linkage(condensedD, method='single')
    Z2 = sch.dendrogram(Y2)
    ax2.set_yticks(x_levels)
    ax2.set_yticklabels(x_sticks, rotation='horizontal')
    ax2.set_xticklabels(Yticklabels, minor=False)
    plt.title("Input U(x)")

    # Function C_U
    axmatrix = fig.add_axes([0.3, 0.1, 0.6, 0.15])
    idx2 = Z2['leaves']
    im = axmatrix.matshow(C_U, aspect='auto', origin='lower')
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])

    # Plot colorbar.
    axcolor = fig.add_axes([0.98, 0.1, 0.05, 0.45])
    pylab.colorbar(im, cax=axcolor, format='%.4f')

    # Plot axes
    axmatrix.set_xticks(Yticks)
    axmatrix.set_xticklabels(Yticklabels, minor=False)
    axmatrix.xaxis.set_label_position('bottom')
    axmatrix.xaxis.tick_bottom()

    pylab.xticks(rotation=-90, fontsize=8)

    """Save image"""
    plt.savefig(Name+"_heat_input.png", dpi=resolution, bbox_inches='tight')

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
            the format format=%.4f")
