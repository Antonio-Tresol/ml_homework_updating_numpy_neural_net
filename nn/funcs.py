import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_grad(s):
    return s * (1.0 - s)


def relu(x):
    return x * (x > 0)


def relu_grad(x):
    return 1.0 * (x > 0)


# with numerical stability
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def logloss(x, y):
    probs = softmax(x)
    return probs, -y * np.log(probs)


def logloss_grad(probs, y):
    probs[:, y] -= 1.0
    return probs


def batch_hits(x, y):
    return np.sum(np.argmax(x, axis=1) == y)

def mean_absolute_error(x, y):
    return np.abs(x - y)

def tanh(x):
    return tanhFunc(x)

def tanh_grad(x):
    return 1 - np.power(x, 2)

def tanhFunc(x):
    max_exp_argument = np.log(np.finfo(np.float64).max)  
    min_exp_argument = np.log(np.finfo(np.float64).tiny) 
    
    clipped_x = np.clip(x, min_exp_argument, max_exp_argument)
    
    numerator = np.exp(clipped_x) - np.exp(-clipped_x)   # e^x - e^-x
                                                         # -----------
    denominator = np.exp(clipped_x) + np.exp(-clipped_x) # e^x + e^-x

    tanh_value = numerator / denominator

    return tanh_value


def tanh_np(x):
    return np.tanh(x)

def tanh_grad_np(x):
    return 1 - np.power(x, 2)