import numpy as np

def kl_divergence(p, q):
    return np.sum(p * np.log(p / q))
