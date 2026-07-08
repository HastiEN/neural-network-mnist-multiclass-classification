# MNIST Classification Using a NumPy MLP

**Course:** Computational Intelligence  
**University:** Amirkabir University of Technology  
**Faculty:** Biomedical Engineering  
**Project Type:** Applied multi-class image classification  
**Implementation:** Multi-Layer Perceptron from scratch using NumPy

## Objective

In this project, a Multi-Layer Perceptron (MLP) neural network was implemented from scratch to classify handwritten digits from the MNIST dataset.

All main steps, including data preprocessing, forward propagation, backpropagation, and weight updates, were implemented manually using NumPy. No ready-made deep learning framework such as TensorFlow or PyTorch was used. This makes the project useful for understanding the internal mechanics of neural networks.

## MNIST Dataset

The MNIST dataset contains grayscale images of handwritten digits from 0 to 9. Each image has a size of 28 x 28 pixels.

Dataset specifications:

- Number of training samples: 60,000
- Number of test samples: 10,000
- Image size: 28 x 28 pixels
- Number of classes: 10

Because there are ten possible output classes, this project is a **multi-class classification** problem.

## Data Preprocessing

Before training the neural network, the data was prepared using the following steps:

- The 28 x 28 images were flattened into 784-dimensional vectors.
- Pixel values were normalized to the range `[0, 1]` by dividing them by 255.
- Labels were converted to one-hot encoded vectors so they matched the network output format.

These preprocessing steps improve numerical stability and help the model converge faster during training.

## Neural Network Architecture

The implemented architecture is:

```text
784 -> 128 -> 10
```

The input layer receives 784 features, one for each pixel in the flattened image. The hidden layer contains 128 neurons, and the output layer contains 10 neurons, one for each digit class.

The network includes:

- A fully connected input-to-hidden linear layer
- ReLU activation in the hidden layer
- A fully connected hidden-to-output linear layer
- Sigmoid activation in the output layer
- Mean Squared Error (MSE) loss
- Stochastic Gradient Descent (SGD) for weight updates

## Reason for Activation Function Choices

ReLU was used in the hidden layer because it reduces saturation problems and helps gradients flow more effectively during training.

Sigmoid was used in the output layer to produce values in the range `[0, 1]`, making the output compatible with one-hot encoded labels.

For real-world multi-class classification, Softmax with cross-entropy is usually a more standard choice. However, the original assignment logic was preserved in this project.

## Training Settings

The training parameters were selected as follows:

- Hidden layer size: 128 neurons
- Batch size: 128
- Number of epochs: 10
- Optimizer: SGD
- Loss function: MSE

**Important note:** The original report mentions `lr = 0.1`, while the base model run in the notebook uses `lr = 0.5`. The final submitted result should keep the learning rate consistent between the notebook and the report.

## Question 1: Base Model Training

The training results show the Mean Squared Error (MSE), training accuracy, and test accuracy over 10 epochs.

The loss decreases gradually during training, which indicates that backpropagation and the weight update process are working correctly. The faster decrease during the first epochs shows that the network quickly learns the main patterns in the data.

The training loss curve shows a stable downward trend without strong oscillations. This suggests that the training process is numerically stable and that the selected learning rate is suitable for the experiment.

In the accuracy plot, both training and test accuracy improve as the number of epochs increases. The final test accuracy is reported at around 97%, showing that the model can generalize well to unseen MNIST samples. The small gap between training and test accuracy suggests that the model does not suffer from serious overfitting.

Overall, the results show that a one-hidden-layer MLP can learn meaningful patterns from MNIST and perform well in handwritten digit classification.

## Question 2: Hidden Neuron Sensitivity Analysis

To analyze the sensitivity of hidden-layer neurons, the input weights of selected neurons from the first-layer weight matrix `W1` were extracted and reshaped into 28 x 28 images.

These visualizations show which image regions and patterns each neuron is more sensitive to.

The results show that different neurons respond to different visual features. Some neurons highlight vertical or diagonal stroke-like patterns, which are common in digits such as 1 and 7. Other neurons respond more strongly to curved structures that appear in digits such as 2, 3, and 8.

Some neurons show mixed edge and light-dark region patterns, suggesting sensitivity to pen-stroke direction and local image structure. These observations indicate that the hidden layer extracts low-level features such as edges, lines, and curves, then transforms them into representations that are more useful for the output layer.

## Question 3: Effect of Hidden Layer Size

In this section, the effect of changing the number of hidden-layer neurons was studied. The network was trained with three different hidden sizes:

- 64 neurons
- 128 neurons
- 256 neurons

The models were compared using test accuracy, convergence behavior, and training time.

The results show that increasing the hidden-layer size improves test accuracy. For `H = 64`, the final test accuracy is about 93.68%. For `H = 128`, it increases to about 94.10%, and for `H = 256`, it reaches about 94.30%.

This means that increasing model capacity helps the network learn more complex patterns. However, the improvement becomes smaller as the hidden layer grows.

The test accuracy comparison plot shows that all three models converge quickly during the first epochs. Larger hidden layers converge slightly better, but after about epoch 6, the improvement becomes small.

From a computational cost perspective, increasing the hidden-layer size increases training time. The average epoch time is about:

- `H = 64`: 0.75 seconds
- `H = 128`: 1.80 seconds
- `H = 256`: 2.48 seconds

Although `H = 256` gives the highest accuracy, the improvement over `H = 128` is limited compared with the additional training cost.

In conclusion, choosing the hidden-layer size is important for balancing final accuracy and computational cost. In this experiment, 128 hidden neurons provide the best balance between accuracy, convergence speed, and training time.

## Limitations and Notes

This implementation is educational and intentionally uses only NumPy. It is useful for understanding the internal mechanics of neural networks, but it is not optimized like deep learning frameworks such as TensorFlow or PyTorch.

Main limitations:

- The model uses MSE with Sigmoid instead of Softmax with cross-entropy.
- Training is slower than framework-based implementations.
- The model is a simple MLP and does not use convolutional layers, which are usually better for image data.
- The learning-rate value should be checked for consistency between the notebook and the final report.

## Conclusion

This project demonstrates how a neural network can be implemented from scratch for multi-class handwritten digit classification. The MLP successfully learns patterns from MNIST images and achieves good classification performance.

The project also shows the effect of hidden-layer size on accuracy and training time. A hidden layer with 128 neurons provides a good balance between model performance and computational cost.

## Report Figures

The images are stored in the `figures/` folder next to this Markdown file. Keep `report.md` and the `figures/` folder together so the figures open correctly.

### Figure 1

![Figure 1: Training output and numerical results from the base MLP experiment.](figures/figure_01.png)

*Training output and numerical results from the base MLP experiment.*

### Figure 2

![Figure 2: Training loss curve for the base model.](figures/figure_02.png)

*Training loss curve for the base model.*

### Figure 3

![Figure 3: Training and test accuracy over epochs.](figures/figure_03.png)

*Training and test accuracy over epochs.*

### Figure 4

![Figure 4: Example MNIST predictions from the trained model.](figures/figure_04.png)

*Example MNIST predictions from the trained model.*

### Figure 5

![Figure 5: Parameter statistics or summary generated from the trained model.](figures/figure_05.png)

*Parameter statistics or summary generated from the trained model.*

### Figure 6

![Figure 6: Hidden neuron sensitivity visualization.](figures/figure_06.png)

*Hidden neuron sensitivity visualization.*

### Figure 7

![Figure 7: Additional hidden neuron sensitivity result.](figures/figure_07.png)

*Additional hidden neuron sensitivity result.*

### Figure 8

![Figure 8: Training comparison for different hidden-layer sizes.](figures/figure_08.png)

*Training comparison for different hidden-layer sizes.*

### Figure 9

![Figure 9: Test accuracy comparison for hidden-layer configurations.](figures/figure_09.png)

*Test accuracy comparison for hidden-layer configurations.*

### Figure 10

![Figure 10: Training time and best accuracy comparison.](figures/figure_10.png)

*Training time and best accuracy comparison.*

### Figure 11

![Figure 11: Final project result figure from the original report.](figures/figure_11.png)

*Final project result figure from the original report.*

