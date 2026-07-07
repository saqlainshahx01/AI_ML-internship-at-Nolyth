from utils import generate_dataset, train_test_split, accuracy_score
from losses import binary_cross_entropy
from neural_network import (
    initialize_parameters,
    forward_pass,
    backward_pass,
    update_parameters,
    predict,
)


def train_model():
    n_samples =1000
    learning_rate = 0.1
    hidden_size= 10
    epochs = 5000
    input_size =2
    output_size =1
    
    X,y =generate_dataset(n_samples=n_samples)
    print("X shape:",X.shape)
    print("y shape:",y.shape)

    X_train, X_test,y_train,y_test= train_test_split(X,y,test_size=0.2)  
    print("\ntraining samples:",X_train.shape[0])
    print("Testing samples :",X_test.shape[0])

    parameters= initialize_parameters(input_size,hidden_size,output_size)

    print("... Start training ...")


    for epoch in range(epochs):
        y_pred,cache = forward_pass(X_train,parameters)
        loss = binary_cross_entropy(y_train, y_pred)
        grads = backward_pass(X_train, y_train, parameters, cache)
        parameters =update_parameters(parameters, grads, learning_rate)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}|Loss: {loss:.2f}")

    train_pre=predict(X_train,parameters)
    test_preds =predict(X_test,parameters)
    train_acc=accuracy_score(y_train, train_pre)
    test_acc = accuracy_score(y_test, test_preds)
    
    print(f"Train Accuracy : {train_acc:.3f}")
    print(f"Test Accuracy  : {test_acc:.3f}")


if __name__ == "__main__":
    train_model()
