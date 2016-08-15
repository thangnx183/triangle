import readfile as rf
import numpy as np
import math

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def grad(X, Y, theta):
    m,n = X.shape

    result = X.T * (sigmoid(X * theta) - Y)

    return result / m

def gradient_descent(X, Y, alpha = 0.03, iterator = 1000):
    m,n = X.shape

    theta = np.ones((n,1))
    theta = np.matrix(theta)

    for i in range(iterator):
        theta = theta - alpha * grad(X, Y, theta)

    return theta

def percent_test(Xt, Yt, theta):
    h = sigmoid(Xt * theta)

    for i in range(len(h)):
        if h[i] >= 0.5:
            h[i] = 1
        else:
            h[i] = 0

    count = 0

    for i in range(len(h)):
        if h[i] == Yt[i]:
            count = count + 1

    return float(count) / len(h)

X, Y, X_test, Y_test = rf.getdata()

theta = gradient_descent(X,Y)

print percent_test(X_test, Y_test, theta)

Xtest, lisdir = rf.get_final_test()

h = np.matrix(np.zeros((len(lisdir), 1)))

h = sigmoid(Xtest * theta)

for i in range(len(h)):
    if h[i] >= 0.5:
        h[i] = 1
    else:
        h[i] = 0

rf.result(h, lisdir)
