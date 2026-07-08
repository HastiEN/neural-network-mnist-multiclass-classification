# MNIST Multi-Class Classification Using a NumPy MLP

**Course:** Computational Intelligence  
**Project Type:** Applied multi-class image classification  
**Dataset:** MNIST handwritten digits  
**Implementation:** Multi-Layer Perceptron implemented from scratch using NumPy

## Objective

The goal of this project is to implement a neural network from scratch for classifying handwritten digit images from the MNIST dataset.

The model is a one-hidden-layer Multi-Layer Perceptron (MLP). All major neural-network components are implemented manually with NumPy, including forward propagation, backpropagation, loss calculation, and weight updates. No ready-made deep learning framework such as TensorFlow or PyTorch is used.

## Problem Type

This project is a **multi-class classification** problem.

The model receives an image of a handwritten digit and predicts one class from ten possible classes:

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9
```

## Dataset

The MNIST dataset contains grayscale images of handwritten digits.

Main dataset information:

| Item | Value |
|---|---:|
| Training samples | 60,000 |
| Test samples | 10,000 |
| Image size | 28 x 28 pixels |
| Input features after flattening | 784 |
| Number of classes | 10 |

## Data Preprocessing

The following preprocessing steps are applied before training:

1. Each 28 x 28 image is flattened into a 784-dimensional vector.
2. Pixel values are normalized from `[0, 255]` to `[0, 1]`.
3. Integer labels are converted to one-hot encoded vectors.

These steps make the data suitable for training a neural network and help improve numerical stability.

## Model Architecture

The implemented neural network architecture is:

```text
784 -> 128 -> 10
```

This means:

- 784 input neurons for the flattened image pixels
- 128 neurons in the hidden layer
- 10 output neurons for the digit classes

Main model components:

| Component | Description |
|---|---|
| Input layer | Flattened 784-dimensional MNIST image |
| Hidden layer | Fully connected layer with 128 neurons |
| Hidden activation | ReLU |
| Output layer | Fully connected layer with 10 neurons |
| Output activation | Sigmoid |
| Loss function | Mean Squared Error (MSE) |
| Optimizer | Stochastic Gradient Descent (SGD) |

## Training Process

The model is trained using mini-batch gradient descent.

Training settings:

| Parameter | Value |
|---|---:|
| Epochs | 10 |
| Batch size | 128 |
| Base hidden layer size | 128 |
| Optimizer | SGD |
| Loss | MSE |

**Note:** The original notebook and report contain a small learning-rate inconsistency. The report mentions `lr = 0.1`, while the base model run in the notebook uses `lr = 0.5`. For final submission, the learning rate should be kept consistent between the code and the report.

## Important Results

### Base MLP Result

The one-hidden-layer MLP successfully learns meaningful patterns from the MNIST dataset.

Important observations:

- Training loss decreases over the epochs.
- Training accuracy increases as learning continues.
- Test accuracy also increases, showing that the model generalizes to unseen data.
- The final test accuracy is reported at around **97%** for the base experiment.
- The small difference between training and test accuracy suggests that the model does not show strong overfitting.

### Hidden Neuron Sensitivity

The project also analyzes hidden-layer neuron sensitivity by reshaping selected first-layer weights into 28 x 28 images.

Main interpretation:

- Some neurons become sensitive to vertical or diagonal stroke patterns.
- Some neurons respond to curved structures.
- These patterns are useful for recognizing handwritten digits.
- The hidden layer acts like a feature extractor that detects low-level visual structures such as edges, lines, and curves.

### Hidden Layer Size Comparison

The model is trained with different hidden-layer sizes: 64, 128, and 256 neurons.

Reported comparison:

| Hidden neurons | Best test accuracy | Mean epoch time |
|---:|---:|---:|
| 64 | 93.68% | 0.75 s |
| 128 | 94.10% | 1.80 s |
| 256 | 94.30% | 2.48 s |

Main conclusion from the comparison:

- Increasing the hidden-layer size improves test accuracy.
- The improvement from 128 to 256 neurons is small.
- Training time increases noticeably when the hidden layer becomes larger.
- A hidden layer with **128 neurons** gives a good balance between accuracy and computational cost.

## Discussion

The results show that a simple MLP can classify MNIST digits with good performance, even without using deep learning frameworks.

The project is useful because it shows how the main parts of a neural network work internally:

- Linear layers
- Activation functions
- Loss calculation
- Backpropagation
- Gradient-based parameter updates
- Accuracy evaluation

Although the model performs well, it is still a simple educational implementation. For image classification tasks, convolutional neural networks usually perform better because they preserve spatial information in images.

## Limitations

Main limitations of this project:

- The model uses an MLP instead of a CNN.
- The image is flattened, so spatial structure is not directly preserved.
- The implementation uses Sigmoid with MSE, while Softmax with cross-entropy is usually better for multi-class classification.
- Training is slower than optimized deep learning frameworks.
- The learning-rate value should be checked and made consistent before final submission.

## Conclusion

This project implements a multi-class MNIST digit classifier from scratch using NumPy. The model demonstrates the full training pipeline of a neural network, including preprocessing, forward propagation, backpropagation, and weight updates.

The base model achieves strong performance, with final test accuracy reported around **97%**. The hidden-layer size experiment shows that increasing model capacity can improve accuracy, but also increases training time. Based on the reported results, **128 hidden neurons** provide the best practical balance between performance and computational cost.
