import numpy as np


def binary_cross_entropy(y_true, y_pred):
    # bineary cross entropy formula
    y_pred = np.clip(y_pred, 1e-8,1 - 1e-8)
    # this is formula for binary cross entropy
    loss = -np.mean(y_true*np.log(y_pred)+(1 - y_true) *np.log(1-y_pred))
    return loss


def mean_squared_error(y_true, y_pred):
    """
    mean squared error formula
    """
    result = np.mean((y_true-y_pred)**2)
    return result

if __name__ == "__main__":
    y_true = np.array([[1], [0], [1], [0]])
    y_pred = np.array([[0.9], [0.2], [0.8], [0.1]])
    print("binary_cross_entropy:",binary_cross_entropy(y_true, y_pred))
    print("mean squared error:",mean_squared_error(y_true, y_pred))
