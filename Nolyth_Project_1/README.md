# Tiny Neural Network Using Python and NumPy

##  Project Overview

This project implements a simple neural network from scratch using only Python and NumPy.

No machine learning frameworks such as TensorFlow, PyTorch, or Keras are used.

The model classifies 2D points:

- 0 → Outside the circle  
- 1 → Inside the circle  

---

## Objective

The goal of this project is to understand:

- Forward propagation  
- Backpropagation  
- Gradient descent  
- Loss calculation  
- Accuracy evaluation  

---

##  Project Structure

Tiny-Neural-Network/
│
├── main.py
├── neural_network.py
├── activations.py
├── losses.py
├── utils.py
├── requirements.txt
├── README.md
└── report.md

---

##  File Explanation

### main.py
- Runs the program  
- Handles training loop  
- Prints results  

### utils.py
- Generates dataset  
- Splits data  
- Calculates accuracy  

### neural_network.py
- Initializes parameters  
- Forward pass  
- Backward pass  
- Updates weights  

### activations.py
- ReLU  
- Sigmoid  

### losses.py
- Binary Cross Entropy  

---

##  How to Run

### Step 1: Install dependencies

pip install -r requirements.txt

### Step 2: Run the project

python main.py

---

## Training Flow

1. Generate dataset  
2. Split into train/test  
3. Initialize parameters  
4. Train model:
   - Forward pass  
   - Compute loss  
   - Backpropagation  
   - Update parameters  
5. Calculate accuracy  

---

## Neural Network Architecture

- Input Layer: 2 neurons  
- Hidden Layer: 10 neurons (ReLU)  
- Output Layer: 1 neuron (Sigmoid)  

## Current Output

Epoch 0   | Loss: 0.69  
Epoch 100 | Loss: 0.68  
Epoch 500 | Loss: 0.66  
Epoch 900 | Loss: 0.66  

Train Accuracy : 0.626  
Test Accuracy  : 0.600  

---

## Issue

- Loss is not decreasing properly  
- Accuracy is low (~60%)  

---

## 🛠 Improvements

- Reduce learning rate (0.01)  
- Increase epochs (10000)  
- Adjust hidden neurons (4–8)  
- Check backpropagation implementation  

---

## Expected Output

Epoch 0    | Loss: 0.69  
Epoch 1000 | Loss: 0.45  
Epoch 3000 | Loss: 0.25  

Train Accuracy : 90%+  
Test Accuracy  : 85%+  

---

## Concepts Used

- Forward Propagation  
- Backpropagation  
- Gradient Descent  
- ReLU Activation  
- Sigmoid Activation  
- Binary Cross Entropy  

---

##  Libraries NOT Used

- TensorFlow  
- PyTorch  
- Keras  
- scikit-learn  
- pandas  

---

##  What I Learned

- How neural networks work step-by-step  
- How data flows through layers  
- How loss affects learning  
- How weights are updated  

---

## Author

Name: Saqlain Shah  
Batch: AI-03  
GitHub: https://github.com/saqlainshahx01  

---

## Conclusion

This project demonstrates a basic neural network, but performance can be improved with better tuning and debugging.
