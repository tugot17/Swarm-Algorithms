import numpy as np

A = 20
B = 0.2
C = 2 * np.pi

def ackley(x):
    """
    x: vector of input values
    """
    d = len(x) # dimension of input vector x

    sum_sq_term = -A * np.exp(-B * np.sqrt(np.sum(x*x) / d))

    cos_term = -np.exp(np.sum(np.cos(C*x) / d))

    return A + np.exp(1) + sum_sq_term + cos_term