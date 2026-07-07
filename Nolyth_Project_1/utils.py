import numpy as np


def generate_dataset(n_samples=1000, radius=0.5, random_seed=42):
    np.random.seed(random_seed)
    X = np.random.uniform(-1, 1, (n_samples, 2)) 
    x1=X[:,0]
    x2=X[:,1]
    distance= x1**2 +x2**2
    y = (distance < radius).astype(int).reshape(-1,1)
    return X, y

def train_test_split(X, y, test_size=0.2, random_seed=42):
    np.random.seed(random_seed)
    a = X.shape[0]
    indices =np.random.permutation(a)
    test_count = int(a*test_size)
    test_data= indices[:test_count]
    train_data = indices[test_count:]
    X_train =X[train_data]
    y_train =y[train_data]
    X_test = X[test_data]
    y_test =y[test_data]
    return X_train, X_test, y_train, y_test

def accuracy_score(y_true, y_pred):
    ###    Calculate accuracy.
    accuracy = np.mean(y_true == y_pred)
    return accuracy
