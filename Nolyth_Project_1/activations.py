import numpy as np


def sigmoid(x):
    #sigmoid formula 
    result =1/(1+np.exp(-x))
    return result 

def relu(x):
    #
    # relu formula        

    result =np.maximum(0,x)
    return result

def tanh(x):
    ### tanh formula 
    result= np.tanh(x)
    return result


def softmax(x):
    # softmax formula
    value = x- np.max(x,axis=1 , keepdims=True)
    exp_value = np.exp(value)
    result =exp_value/np.sum(exp_value,axis=1,keepdims=True)
    return result

def sigmoid_derivative(x):
    # Derivative of sigmoid
    s = sigmoid(x)
    result = s*(1 - s)
    return result

def relu_derivative(x):
    ##Derivative of relu
    return (x > 0).astype(int)

def tanh_derivative(x):
    """
    derivative of tanh.
    """
    result= 1 - np.tanh(x)**2
    return result
if __name__ == "__main__":
    x = np.array([-2, -1, 0, 1, 2])
    print("Input:",x)
    print("Sigmoid:",sigmoid(x))
    print("ReLU:", relu(x))
    print("Tanh:",tanh(x))
    print("Softmax:",softmax(x.reshape(1, -1)))
