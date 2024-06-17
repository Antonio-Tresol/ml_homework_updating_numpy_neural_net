import numpy as np
<<<<<<< Updated upstream

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))
=======
import math

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
>>>>>>> Stashed changes

def sigmoid_grad(s):
    return s * (1.0 - s)

def relu(x):
    return x * (x > 0)

<<<<<<< Updated upstream
def  relu_grad(x):
=======
def relu_grad(x):
>>>>>>> Stashed changes
    return 1.0 * (x > 0)

#with numerical stability
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def logloss(x, y):
    probs = softmax(x)
    return probs, -y * np.log(probs)

def logloss_grad(probs, y):
    probs[:,y] -= 1.0
    return probs

def batch_hits(x, y):
    return np.sum(np.argmax(x, axis=1) == y)
<<<<<<< Updated upstream
=======

def mean_absolute_error(x, y):
    return np.abs(x - y)

def tanh(x):
    return tanhFunc(x)
    #return np.tanh(x)

def tanh_grad(x):
    return 1 - tanhFunc(x) ** 2
    #return 1 - np.tanh(x) ** 2

def tanhFunc(x):
    #print(f"tanhFunc: {x}")
    max_exp_argument = np.log(np.finfo(np.float64).max)  # Maximum exponent argument for 64-bit float
    min_exp_argument = np.log(np.finfo(np.float64).tiny) # Minimum exponent argument for 64-bit float
    
    clipped_x = np.clip(x, min_exp_argument, max_exp_argument)
    
    tanh_value = np.where(clipped_x >= 0,
                          1 - 2 / (1 + np.exp(-2 * clipped_x)),
                          2 / (1 + np.exp(2 * clipped_x)) - 1)

    #print(f"numerator: {numerator}")
    #print(f"denominator: {denominator}")
    
    # Handle division by zero carefully
    # If denominator is very close to zero, return a very large or small value as approximation
    return tanh_value

    '''
    return ((np.exp(x) - np.exp(-x)) # e^x - e^-x
        /                                # ---------- 
        (np.exp(x) + np.exp(-x)))    # e^x + e^-x
    '''
>>>>>>> Stashed changes
