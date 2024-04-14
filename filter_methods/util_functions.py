import numpy as np

def kl_divergence(p, q):
    return np.sum(p * np.log(p / q))

def create_joint_distribution(p, q):
    return np.outer(p, q)
