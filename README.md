# **Comparing Single Hidden Layer, Two Hidden Layers (based on [Chapter 11](https://github.com/rasbt/machine-learning-book/blob/main/ch11/ch11.ipynb)) and Fully Connected Keras Implementation for MNIST Classification**

This repository contains the implementation and comparison of artificial neural networks (ANNs) for MNIST digit classification. The models include a single-hidden-layer ANN, a two-hidden-layer ANN, and a fully connected Keras implementation. The work is based on [Chapter 11](https://github.com/rasbt/machine-learning-book/blob/main/ch11/ch11.ipynb) of *Machine Learning with PyTorch and Scikit-Learn* by Raschka et al. (2022).

---

## **Submitted by:**
*Noa Magrisso & Shaked Tayouri*

---

## **Overview**
  This project explores the impact of architectural choices, activation functions, and loss functions on training and performance. The main focus is on comparing:
  - A **single-hidden-layer ANN** - baseline, a basic ANN as described in [Chapter 11](https://github.com/rasbt/machine-learning-book/blob/main/ch11/ch11.ipynb).
  - A **two-hidden-layer ANN** - an extended version to explore deeper architectures and hierarchical feature learning.
  - A **fully connected Keras ANN** - for benchmarking with pre-optimized deep learning libraries.

The initial experiments followed a **Mean Squared Error (MSE) loss with Sigmoid activation** (Notebook 1), later improved by switching to **Cross-Entropy loss, ReLU activation in hidden layers, and SoftMax in the output layer** (Notebook 2).

---

## **Implemented Models**
### **Custom Models**
- **One-Hidden-Layer ANN**: Implements a simple ANN following the original notebook.
- **Two-Hidden-Layer ANN**: Extends the previous model to explore the effect of additional depth.

### **Keras Model**
- Implements a **fully connected ANN** in Keras to compare performance with the custom models.
- Initially followed the same **Sigmoid + MSE** structure before switching to **ReLU + SoftMax with Cross-Entropy**.

---

## **Training Setup**
### **Notebook 1: Initial Version (Sigmoid + MSE)**
- **Loss Function**: MSE
- **Activation Functions**:
  - Sigmoid in all layers (hidden + output).
- **Findings**:
  - The two-hidden-layer ANN provided a **slight improvement** in accuracy.
  - The **Keras ANN underperformed** with MSE loss.
  - Adding SoftMax to the **Keras output layer** and switch it to Cross-Entropy improved performance.

### **Notebook 2: Improved Version (ReLU + Cross-Entropy)**
- **Loss Function**: Cross-Entropy (replacing MSE).
- **Activation Functions**:
  - **ReLU in hidden layers** for better convergence.
  - **SoftMax in the output layer** for class probability distribution.
- **Findings**:
  - All models showed improved stability and learning efficiency.
  - The **Keras model matched the custom models in performance** after the updates.
  - **Training convergence was significantly improved.**

---

## **Evaluation Metrics**
The models were evaluated using the following metrics:

- **Accuracy**: Percentage of correct predictions.
- **Macro AUC**: Measures the model’s ability to distinguish between classes equally.
- **Loss Function**:
  - **Initial Version** (Notebook 1): **Mean Squared Error (MSE)**.
  - **Improved Version** (Notebook 2): **Cross-Entropy** (better for multi-class classification).
- **Comparison of Activation Functions**:
  - **Sigmoid vs. ReLU**: Evaluating their effect on performance and training speed.
  - **SoftMax in Output Layer**: Ensuring better probability distribution.

---

## **Dataset & Experimental Setup**
- **Dataset**: MNIST (28×28 grayscale images, 10 classes: digits 0-9).
- **Data Split**:
  - 70% Training
  - 30% Testing
  - 10% Validation (from training set).
- **Training Parameters**:
  - **Batch Size**: 100
  - **Epochs**: 50
  - **Learning Rate**: 0.1

### **Hyperparameter Experiments**
- Validation set size, learning rate, and epochs were tested.
- These variations **did not yield significant changes**, so the main results are presented.

---

## **Key Takeaways**
- **MSE is not ideal for multi-class classification** since it does not optimize class probabilities directly.
- **Switching to Cross-Entropy + SoftMax** led to better classification performance.
- **ReLU activation** improved training speed and gradient flow, addressing issues with Sigmoid in deep networks.
- **Deeper networks did not always increase accuracy**, but they improved feature learning.
- **The Keras ANN showed competitive performance** after updates.

---

## **References**

1. **Machine Learning with PyTorch and Scikit-Learn**:
   - Raschka, S., Liu, Y., & Mirjalili, V. (2022). "Implementing a Multi-layer Artificial Neural Network from Scratch."
   - [Chapter 11 on GitHub](https://github.com/rasbt/machine-learning-book/blob/main/ch11/ch11.ipynb)

2. **MNIST Dataset**:
   - Yann LeCun and Corinna Cortes. "The MNIST Database of Handwritten Digits."
   - [Website Link](http://yann.lecun.com/exdb/mnist/)

3. **Keras Documentation**:
   - TensorFlow/Keras: "Building Fully Connected Neural Networks."
   - [Keras Documentation](https://keras.io/guides/sequential_model/)
