# math_utils.py

import numpy as np

def getPseudoInverse(A):
    return np.linalg.pinv(A)

def getZerosMatrix(r, c):
    return np.zeros((r, c))

def getIdentityMatrix(i):
    return np.eye(i)