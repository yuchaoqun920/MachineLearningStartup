#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np


import matplotlib as mpl
import matplotlib.pyplot as plt
print 'Plotting data with + indicating (y = 1) examples and o '\
'indicating (y = 0) examples.\n'

def plotData(X, y):
    pos = y==1
    neg = y==0
#     print len(y[pos])
    x1 = X[:, 0].reshape(-1,1)
    x2 = X[:, 1].reshape(-1,1)
    plt.plot(x1[pos], x2[pos], 'k+', linewidth=4, markersize=7, label='Admitted')
    plt.plot(x1[neg], x2[neg], 'yo', linewidth=4, markersize=6, label='Not admitted')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.title('Final Results')
    # plt.legend(loc='upper left')
    plt.legend(loc='upper right')
    plt.show()

# 预测结果画图
def plotDecisionBoundary(X, y, theta):
    M, N = 1000, 1000
    x1_min, x1_max = 30, 100
    x2_min, x2_max = 30, 100
    t1 = np.linspace(x1_min, x1_max, M)
    t2 = np.linspace(x2_min, x2_max, N)
    x1, x2 = np.meshgrid(t1, t2)
    x_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
    # print x_test.shape
    # cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])
    # cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    cm_light = mpl.colors.ListedColormap(['#FF8080', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['r', 'b'])
    (m, n) = x_test.shape
    xt_test = np.hstack((np.ones((m, 1)), x_test))
    yhat_test = predict(xt_test, theta)
    yhat_test = yhat_test >= 0.5
    # print yhat_test
    plt.pcolormesh(x1, x2, yhat_test.reshape(x1.shape), cmap=cm_light)     # 预测值的显示
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', s=50, cmap=cm_dark)    # 样本的显示
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.title('Final Results')
#     plt.legend(loc='upper right')
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.grid()
    plt.savefig('lr.png')
    plt.show()

def sigmoid(Z):
    expz = np.exp(-Z)
    sigmd = 1/(1+expz)
    return sigmd

def costFunction(X, y, theta):
    m = len(y)
    hx = sigmoid(np.dot(X, theta))
    print hx.shape
    j = -1*np.dot(y.T, np.log(hx)) - np.dot((1-y.T), np.log(1-hx))
    j=j/m
#     grad = np.dot(X.T, (hx-y))/m
    return j[0][0]

def costFunction2(X, y, theta):
    m = len(y)
#     print np.dot(X, theta)
    hx = sigmoid(np.dot(X, theta))
#     print hx
#     print hx.shape
    j = -1*np.dot(y.T, np.log(hx)) - np.dot((1-y.T), np.log(1-hx))
    j=j/m
    # grad = X'.(hx-y)/m
    grad = np.dot(X.T, (hx-y))/m
    return j[0][0], grad

# 直接使用梯度下降，收敛非常慢，需要更高级的优化算法
# 迭代次数10,000,000，一千万次，勉强收敛到一个比较好的值
def gradientDescent(X, y, theta, alpha, costfunc, iterations):
    m = len(y)
    j_history = np.zeros((iterations, 1))

    for iter in range(iterations):
        j_history[iter], delta = costfunc(X, y, theta)
        theta = theta - alpha*delta

    # plt.plot(j_history, 'b-', linewidth=4)
    # plt.show()
    print j_history

    return theta

# 预测
def predict(X, theta):
    m = len(X)
    p = sigmoid(np.dot(X, theta))
    return p

if __name__ == '__main__':
    data = np.loadtxt('ex2data1.txt', delimiter=',')
    print data.shape

    X, y = np.split(data, (2, ), axis=1)
    # print y

    # Plotting
    plotData(X, y)

    (m, n) = X.shape
    print m, n
    Xt = np.hstack((np.ones((m, 1)), X))
    initial_theta = np.zeros((n + 1, 1))
    cost = costFunction(Xt, y, initial_theta)
    print cost
    theta = initial_theta

    # gradient descent
    theta = gradientDescent(Xt, y, theta, 0.001, costFunction2, 10000000)
    print theta

    prob = sigmoid(np.dot(np.array([1, 45, 85]), theta))
    print 'For a student with scores 45 and 85, we predict an admission ', 'probability of %f\n' % prob

    # 计算训练准确率
    p = predict(Xt, theta)
    p_r = p >= 0.5
    # print p_r
    accuracy = np.mean(p_r == y)
    print 'accuracy: %0.2f%%\n' % (accuracy * 100)

    plotDecisionBoundary(X, y, theta)
