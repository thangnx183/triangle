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


#kNN
#print X.shape
#f1 = open('X.txt','w')
#print X
#f1.write(str(X))
'''
m,n = X.shape
title = [i for i in range(n)]
X = np.insert(X,0,title,0)
np.savetxt('T.txt',X,fmt = '%i',delimiter = ',')

Y = np.insert(Y,0,[5],0)
np.savetxt('Y.txt',Y,fmt = '%i')
'''
#Y = np.array(Y)
from sklearn import preprocessing, cross_validation, neighbors

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size = 0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train,Y_train)

percent = clf.score(X_test, Y_test)
print percent

Xtest, lisdir = rf.get_final_test()

h = clf.predict(Xtest)

print h[2]

rf.result(h, lisdir)

#kNN
'''
theta = gradient_descent(X,Y)

print percent_test(X_test, Y_test, theta)

Xtest, lisdir = rf.get_final_test()

#print Xtest.shape

h = np.matrix(np.zeros((len(lisdir), 1)))

h = sigmoid(Xtest * theta)
#print h.shape


for i in range(len(h)):
    if h[i] >= 0.5:
        h[i] = 1
    else:
        h[i] = 0

rf.result(h, lisdir)
'''
