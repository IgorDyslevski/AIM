import numpy as np


def mae(predicted: np.array, true: np.array):
    return np.sum(np.abs(predicted - true)) / predicted.shape[0]

def mse(predicted: np.array, true: np.array):
    return np.sum(np.power(predicted - true, 2)) / predicted.shape[0]

def bcross(predicted: np.array, true: np.array):
    return -np.sum(true * np.log2(predicted)) / predicted.shape[0]
