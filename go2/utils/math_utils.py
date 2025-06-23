# math_utils.py

import numpy as np

def getPseudoInverse(A):
    return np.linalg.pinv(A)

def getZerosMatrix(r, c):
    return np.zeros((r, c))

def getIdentityMatrix(i):
    return np.eye(i)

def printSize(matrix):
    print(matrix.shape)

def getRowSize(matrix):
    return matrix.shape[0]
def getColumnSize(matrix):
    return matrix.shape[1]

    