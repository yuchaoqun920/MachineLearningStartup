#!/usr/bin/python
# -*- coding:utf-8 -*-

from time import time as timenow
import math
import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.optimize import leastsq
import scipy.optimize as opt
import scipy
import matplotlib.pyplot as plt

def warmUpExercise():
    print np.eye(5)

def plotData(x, y):
    plt.plot(x, y, 'go', markersize=8)
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.show()

# theta' * X - y
def computeCost(X, y, theta):
    m = len(y)
    y_hat = np.dot(X, theta)
    cost = np.dot((y_hat-y).T, (y_hat-y))/(2.0*m)
    return cost[0, 0]

def gradientDescent(X, y, theta, alpha, iterations):
    m = len(y)
    j_history = np.zeros((iterations, 1))

    for iter in range(iterations):
        error = np.dot(X, theta)-y
        delta = np.dot(X.T, error)*alpha/m
        theta = theta - delta

        j_history[iter] = computeCost(X, y, theta)

    # plt.plot(j_history, 'b-', linewidth=4)
    # plt.show()
    # print j_history

    return theta

if __name__ == "__main__":

    # warmUpExercise()
    # raw_input()

    # pandas读入
    # data = pd.read_csv(path)

    # numpy读入
    data = np.loadtxt('ex1data1.txt', delimiter=',', skiprows=0)
    # print data

    x, y = np.split(data, (1,), axis=1)

    # X = data[:, 0]
    # y = data[:, 1]
    m = len(y)
    # print x.shape, y.shape
    # plotData(X, y)
    #
    # raw_input()

    X = np.hstack((np.ones((m,1)), data[:, 0].reshape((m,1))))
    # print X
    theta = np.zeros((2, 1))
    cost1 = computeCost(X, y, theta)
    print cost1

    iterations = 1500
    alpha = 0.01

    theta = gradientDescent(X, y, theta, alpha, iterations)
    print theta
    y_hat = np.dot(X, theta)

    plt.plot(X[:, 1], y, 'go', markersize=8)
    plt.plot(X[:, 1], y_hat, 'b-', linewidth=4)
    plt.xlim(4, 25)
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.show()


    # 画3D图
    N, M = 100, 100     # 横纵各采样多少个值
    x1_min, x1_max = -10, 10   # 第0列的范围
    x2_min, x2_max = -1, 4     # 第1列的范围
    t1 = np.linspace(x1_min, x1_max, N)
    t2 = np.linspace(x2_min, x2_max, M)
    x1, x2 = np.meshgrid(t1, t2)                    # 生成网格采样点
    x_test = np.stack((x1.flat, x2.flat), axis=1)   # 测试点

    print x_test.shape

    j_vals = np.zeros((len(x_test), 1))
    for i in range(len(x_test)):
        j_vals[i] = computeCost(X, y, x_test[i, :])

    print j_vals.shape

    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(x1, x2, j_vals.reshape(x1.shape), rstride=1, cstride=1, cmap=cm.Paired, linewidth=0)
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x1, x2, j_vals.reshape(x1.shape), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
