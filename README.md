# MNIST-Handwritten-Digit-Classification
This project is a Deep Learning application focused on Image Classification using the classic MNIST Dataset. It demonstrates a complete workflow from data preprocessing and normalization to building, training, and evaluating a Neural Network using TensorFlow/Keras.

This project is a **Deep Learning** application focused on **Image Classification** using the classic **MNIST Dataset**. It demonstrates a complete workflow from data preprocessing and normalization to building, training, and evaluating a Neural Network using **TensorFlow/Keras**.

---

## MNIST Handwritten Digit Classification

This repository contains a Jupyter Notebook that implements a **Multi-Layer Perceptron (MLP)** to recognize handwritten digits (0-9). The model is trained on the MNIST dataset, which contains 70,000 grayscale images of  pixels.

###  Model Architecture

The neural network is built using the Keras `Sequential` API and consists of:

* **Flatten Layer:** Converts the  image matrix into a 1D vector of 784 pixels.
* **Dense Layer (128 units):** Fully connected layer with **ReLU** activation.
* **Dense Layer (32 units):** Fully connected layer with **ReLU** activation.
* **Output Layer (10 units):** Uses **Softmax** activation to provide probability scores for each digit class.

###  Technical Stack

* **Language:** Python
* **Deep Learning:** TensorFlow, Keras
* **Data Analysis:** NumPy
* **Visualization:** Matplotlib
* **Metrics:** Scikit-learn (`accuracy_score`)

###  Key Features

* **Data Normalization:** Scales pixel values from [0, 255] to [0, 1] to accelerate model convergence.
* **Optimization:** Uses the **Adam** optimizer and **Sparse Categorical Crossentropy** loss function.
* **Validation:** Implements a 20% validation split during training to monitor for overfitting.
* **Performance Tracking:** Includes visual plots for **Loss** and **Accuracy** across 25 epochs.
* **Prediction:** A functional prediction pipeline to test the model on individual test images.

###  Results

The model achieves an impressive accuracy of **~97.9%** on the test dataset after 25 epochs of training.

---

###  Getting Started

1. Clone this repository.
2. Install dependencies:
```bash
pip install tensorflow matplotlib scikit-learn

```


3. Run the `mnist-classification.ipynb` notebook to see the results.
