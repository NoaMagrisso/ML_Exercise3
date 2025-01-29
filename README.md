# **Comparing Single Hidden Layer, Two Hidden Layers (based on [Chapter 11](https://github.com/rasbt/machine-learning-book/blob/main/ch11/ch11.ipynb)) and Fully Connected Keras Implementation for MNIST Classification**

This repository contains the implementation and comparison of artificial neural networks (ANNs) with one hidden layer, two hidden layers, and a fully connected implementation in Keras for the classification of handwritten digits from the MNIST dataset. The work is based on [Chapter 11](https://github.com/rasbt/machine-learning-book/blob/main/ch11/ch11.ipynb) of the book *Machine Learning with PyTorch and Scikit-Learn* by Raschka et al. (2022).

---

Submitted by:

  - Noa Magrisso
  - Shaked Tayouri

---

## **Features**

- **Custom Implementations**:
  - **One-Hidden-Layer ANN**: A basic ANN as described in Chapter 11.
  - **Two-Hidden-Layer ANN**: An extended version to explore deeper architectures and hierarchical feature learning.

- **Fully Connected Keras Model**:
  - A fully connected ANN implemented in Keras to compare performance with the custom models.

- **Metrics for Evaluation**:
- - **MSE (Mean Squared Error)**: Measures the average squared difference between predicted probabilities and true labels.
  - **Accuracy**: Measures the correctness of the predictions.
  - **Macro AUC**: Assesses the model's ability to distinguish between all classes with equal weight given to each.
  
- **Data**:
  - The MNIST dataset is used for training, validation, and testing.
  - Data split: 70% training, 30% testing.
  - Validation split is initialized 10% from training, but other splits were checked.

---

## **You have to know**

**Different Validation Sizes and Parameters**:
 - This repository includes multiple notebooks with variations in validation sizes and other parameters to test their impact on model performance.
 - These configurations allow for comprehensive experimentation and insights into how the models perform under different setups.

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
