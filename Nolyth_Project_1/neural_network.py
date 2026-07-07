import numpy as np

from activations import sigmoid, relu, relu_derivative


def initialize_parameters(input_size, hidden_size, output_size, random_seed=42):
###    Create the starting weights and biases.

    np.random.seed(random_seed)
    w1 =np.random.randn(input_size,hidden_size)*0.01
    b1 =np.zeros((1, hidden_size))
    w2 =np.random.randn(hidden_size,output_size)*0.01
    b2 = np.zeros((1, output_size))
    parameters = {"W1": w1,"b1":b1,"W2":w2,"b2":b2}

    return parameters


def forward_pass(X, parameters):
    w1 =parameters["W1"]
    b1 = parameters["b1"]
    w2 =parameters["W2"]
    b2= parameters["b2"]

    z1=X @ w1 + b1
    a1 =relu(z1)
    z2 =a1 @ w2 + b2
    a2=sigmoid(z2)

    cache = {"Z1":z1,"A1":a1,"Z2":z2,"A2": a2}
    return a2,cache


def backward_pass(X, y, parameters, cache):
    """
    Calculate gradients descent"""

    m =X.shape[0]
    w2 =parameters["W2"]
    z1 =cache["Z1"]
    a1 =cache["A1"]
    a2 =cache["A2"]

    dZ2 =a2-y
    dW2=a1.T@dZ2/m
    db2 = np.mean(dZ2,axis=0, keepdims=True)

    dA1 =dZ2 @ w2.T
    dZ1=dA1 *relu_derivative(z1)
    dW1= X.T@ dZ1 / m
    db1 =np.mean(dZ1,axis=0,keepdims=True)
    gradients = {"dW1": dW1,"db1":db1,"dW2": dW2,"db2": db2,
    }
    return gradients


def update_parameters(parameters, gradients, learning_rate):
    """
    Update weights and biases.

    Simple meaning:
    The model has made a mistake.
    Gradients tell us the direction of improvement.
    Learning rate controls how big the update step should be.
    """

    w1 =parameters["W1"]
    b1 =parameters["b1"]
    w2= parameters["W2"]
    b2 = parameters["b2"]

    dW1 =gradients["dW1"]
    db1 =gradients["db1"]
    dW2= gradients["dW2"]
    db2 =gradients["db2"]

    parameters["W1"] =w1-learning_rate*dW1
    parameters["b1"] =b1 -learning_rate *db1
    parameters["W2"]= w2-learning_rate*dW2
    parameters["b2"]=b2- learning_rate*db2

    return parameters


def predict(X, parameters):

    probabilities, _ =forward_pass(X,parameters)
    predictions =(probabilities >= 0.5).astype(int)
    return predictions
