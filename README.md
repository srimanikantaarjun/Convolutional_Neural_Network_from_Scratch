# Convolutional Neural Network from Scratch

## In this repository, I have built a Convolutional Neural Network from scratch using just Python and Numpy


### How the model gets trained ?

* Initialize Weights and Biases in ForwardPropagation method of Convolutional layer
* Do the Forward Propagation of `ConvOp`, `MaxPool`, and `Softmax` layers in order
* Calculate loss at `Softmax` layer
* Do the Backward Propagation of `Softmax`, `MaxPool`, and `ConvOp` layers in order
* Update Weights and Biases
* Do the Forward Propagation
* . . .

### Flow

Output of **`back_prop`** method of **`Softmax`** layer -> Input of **`back_prop`** method of **`MaxPool`** layer\
Output of **`back_prop`** method of **`MaxPool`** layer -> Input of **`back_prop`** method of **`ConvOp`** layer\
Output of **`back_prop`** method of **`Convop`** layer -> Weights and Biases get updated in **`forward_prop`** method of **`ConvOp`** layer

### Convolutional layer

- Defined parameters like **Number of Filters** `num_filters` and **Filter Size** `filter_size`
- Used basic python **Arithmetic Operators**(`+, *, /, //`) to build a **Convolutional Filter** `conv_filter`
- Built functions for **Forward propagation** and **Backpropagation** `forward_prop` and `back_prop`

### MaxPooling layer

- Defined the pooling size `filter_size` to reduce the image size
- Used basic python **Arithmetic Operators**(`+, *, /, //`) to build a **Convolutional Filter** `conv_filter`
- Built functions for **Forward propagation** and **Backpropagation** `forward_prop` and `back_prop`
- Trained the **Backpropagation** using the output of **Backpropagation** from **`Softmax`** layer

### Softmax layer

- Initialized the `weights` and `biases` using `np.random.randn` and `np.zeros` respectively
- **Forward Propagation** `forward_prop`, **flattened** the image into a **1-D array** and multiplying it with corresponding weights and biases
- Transforming the output of forward propagation into **prediction probabilities**
- The weights and biases are **updated** during the `back_prop` layer

### Results
|Epoch | CNN from Scratch using only Python and NumPy | CNN using keras |
| :--- | :------: | :-----: |
|Epoch 1 Loss| 0.444 | 0.344  |
|Epoch 1 Accuracy| 87% | 90.16% |
|Epoch 2 Loss| 0.455 | 0.1932 |
|Epoch 2 Accuracy| 85% | 94% |
|Epoch 3 Loss| 0.397 | 0.1427 |
|Epoch 3 Accuracy| 88% | 95.94% |
|Epoch 4 Loss| 0.397 | 0.1126 |
|Epoch 4 Accuracy| 87% | 96.76 |
|Epoch 5 Loss| 0.450 | 0.0947 |
|Epoch 5 Accuracy| 86% | 97.30 |
|Test Loss | 11.0 | 0.0909 |
|Test Accuracy | 85% | 97.25% |
