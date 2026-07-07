# Project Report: Tiny Neural Network Using NumPy

## 1. What is a Neural Network?

Write your answer here.

Hint:
A neural network is a model that learns patterns from data. It takes input, performs calculations using weights and biases, and gives an output.

Your answer:

```text
A neural network is a model that learns from data. It takes some input, does math with weights and biases, and gives an output as a prediction. In this project, it takes two numbers (x1, x2) and tells whether the point is inside or outside a circle.
```

## 2. What is a Feature?

Write your answer here.

Hint:
In this project, x1 and x2 are features.

Your answer:

```text
A feature is one piece of input information the model uses to make a decision. In this project, x1 and x2 are the features. They are the only two numbers the model looks at to decide the answer.
```

## 3. What is a Label?

Write your answer here.

Hint:
The label is the correct answer.

In this project:

```text
0 means outside the circle
1 means inside the circle
```

Your answer:

```text
A label is the correct answer for an input. The model tries to match its prediction to this label.
0 means outside the circle
1 means inside the circle
```

## 4. What is an Activation Function?

Write your answer here.

Hint:
An activation function changes the values inside the network and helps the model learn better patterns.

Your answer:

```text
An activation function changes the numbers inside the network so it can learn more complex patterns, not just straight lines. I used ReLU in the hidden layer and sigmoid in the output layer to turn the final number into something between 0 and 1.
```

## 5. What is Loss?

Write your answer here.

Hint:
Loss tells us how wrong the model is.

Your answer:

```text
Loss is a number that tells us how wrong the model's prediction is. If loss is high, the model is making bad guesses. If loss is low, the model is close to the correct answer. I used binary cross-entropy since there are only two classes (0 and 1).
```

## 6. What is Gradient Descent?

Write your answer here.

Hint:
Gradient descent is the process of changing weights and biases step by step so the model becomes better.

Your answer:

```text
Gradient descent is how the model improves itself. It looks at the gradients (the direction of the error) and slowly changes the weights and biases so the loss keeps getting smaller after every epoch.
```

## 7. What is Backpropagation?

Write your answer here.

Hint:
Backpropagation calculates how weights and biases should change to reduce the loss.

Your answer:

```text
Backpropagation is how the model figures out which weights and biases caused the error. It goes backward from the output to the input and calculates how much each weight should change to reduce the loss.
```

## 8. What Was the Most Difficult Part?

Write your answer here.

Example:

```text
The most difficult part was understanding the shape of arrays.
```

Your answer:

```text
The most difficult part was keeping track of the shapes of the matrices, especially in backward_pass(), where I had to make sure dW1, db1, dW2, and db2 all matched the correct layer.
```

## 9. How Did You Solve the Difficulty?

Write your answer here.

Example:

```text
I solved it by printing the shapes of X, W1, A1, W2, and A2.
```

Your answer:

I solved it by printing the .shape of X, W1, Z1, A1, W2, Z2, and A2 while writing the code. Seeing the actual shapes made it much easier to catch mistakes.

## 10. What Did You Learn From This Project?

Write your answer here.

Try to write at least 5 lines.

Your answer:

```
I learned how data actually moves through a neural network step by step, not just in theory.I learned that loss is just a number showing how wrong the model is, and training means trying to make that number smaller.I also learned that settings like learning rate and hidden layer size really change how well the model trains — a small learning rate barely moved the loss, but a bigger one worked much better.

```
