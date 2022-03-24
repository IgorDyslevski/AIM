import numpy as np


def sigmoid(x: np.array) -> np.array:
    return 1 / (1 + np.exp(-x))

def relu(x: np.array) -> np.array:
    return np.maximum(0, x)

def softmax(x: np.array) -> np.array:
    return x / np.sum(x)

def tanh(x: np.array) -> np.array:
    return np.tanh(x)
