import os
import cv2
import numpy as np
#import os


# return path of all image in folder
def get_filepaths(directory):
    file_paths = []

    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)

    return file_paths

#convert image into binary matrix
def convert_image(path):
    img = cv2.imread(path,0)

    row, column = img.shape

    for x in range(row):
        for y in range(column):
            if img[x][y] == 255:
                img[x][y] = 1
            else:
                img[x][y] = 0
    return img

#convert data to training set
def image_to_matrix(lis_dir):
    lis = np.array([])
    lis_test = np.array([])

    #for i in range(len(lis_dir)/100):
    for i in range(7500):
        #if i < len(lis_dir)*3/4/100:
        if i < 7500*3/4:
            matrix = convert_image(lis_dir[i])
            matrix = matrix.reshape((1,400))
#kNN            matrix = np.insert(matrix,0,[1])
            lis = np.append(lis, matrix)
        else:
            matrix = convert_image(lis_dir[i])
            matrix = matrix.reshape((1,400))
#kNN            matrix = np.insert(matrix,0,[1])
            lis_test = np.append(lis_test, matrix)
    x = 7500*3/4
    y = 7500 - 7500*3/4
    return lis,  x, lis_test, y



def getdata():
    #lis = os.listdir()
    lis = get_filepaths('/home/thangnx/code/triangletest/triangle_competition/train/triangle')
    X1, len1, X1_test, len_test1 = image_to_matrix(lis)
    Y1 = np.ones((len1, 1))
    Y1_test = np.ones((len_test1, 1))

    lis = get_filepaths('/home/thangnx/code/triangletest/triangle_competition/train/non-triangle')
    X2, len2, X2_test, len_test2 = image_to_matrix(lis)
    Y2 = np.zeros((len2, 1))
    Y2_test = np.zeros((len_test2, 1))

    X = np.append(X1, X2)
    X = X.reshape((len1 + len2, 400)) #knn
    #X = np.matrix(X, dtype = float)

    X_test = np.append(X1_test, X2_test)
    X_test = X_test.reshape((len_test1 + len_test2, 400)) #kNN
    X_test = np.matrix(X_test, dtype = float)

    Y = np.append(Y1, Y2)
    #Y = Y.reshape((len1 + len2, 1))
    #Y = np.matrix(Y.reshape((len1+len2,1)), dtype = float)

    Y_test = np.append(Y1_test, Y2_test)
    Y_test = np.matrix(Y_test.reshape((len_test1 + len_test2, 1)), dtype =float)

    return X, Y, X_test, Y_test

def get_final_test():
    lisdir = os.listdir('/home/thangnx/code/triangletest/triangle_competition/test')
    x = len(lisdir)
    lis = np.array([])

    for each in lisdir:
        matrix = convert_image('/home/thangnx/code/triangletest/triangle_competition/test/'+each)
        matrix = matrix.reshape((1,400))
#knn        matrix = np.insert(matrix,0,[1])
        lis = np.append(lis, matrix)
    #print len(lis)
#knn    Xtest = np.matrix(lis.reshape((x, 401)))
    Xtest = lis.reshape((x,400)) #knn
    return Xtest, lisdir

def result(h, lisdir):
    f = open('result_knn1.txt','w')

    for i in range(len(h)):
        f.write(str(h[i])+ str(' ') + str(lisdir[i]) +str('\n') )

    return
