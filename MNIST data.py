import numpy as np
import ConvOp
import MaxPool
import Activations
import os
import sys
os.environ["TF_CP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf

sys.stdout = open("MNIST data output.text", "w")

path = "/Users/srimanikantaarjunkarimalammanavar/Desktop/Projects/Neural Networks from Scratch/CNN_from_Scratch/mnist.npz"
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data(path = path)
train_images = X_train[:1500]
train_labels = y_train[:1500]
test_images = X_test[:1500]
test_labels = y_test[:1500]
print("Imported MNIST data and split it into train and test to try our CNN model on it . . .")

num_filters = 8
filter_size = 3
filter_size_for_maxpool = 2
input_node = 13 * 13 * 8
softmax_node = 10
conv = ConvOp.ConvOp(num_filters = num_filters, filter_size = filter_size)
maxpool = MaxPool.Max_Pool(filter_size = filter_size_for_maxpool)
softmax = Activations.Softmax(input_node = input_node, softmax_node = softmax_node)
print(f"Defined the filter_size as {filter_size} and num_filters as {num_filters} for Convolutional layer,"
      f"Defined the filter_size as {filter_size_for_maxpool} for MaxPool layer,"
      f"Defined the input_node as {input_node} and softmax_node as {softmax_node} for Softmax layer")


def cnn_forward_prop(image, label):
    out_p = conv.forward_prop((image/255) - 0.5)
    out_p = maxpool.forward_prop(out_p)
    out_p = softmax.forward_prop(out_p)
    cross_entropy_loss = -np.log(out_p[label])
    accuracy_eval = 1 if np.argmax(out_p) == label else 0
    return out_p, cross_entropy_loss, accuracy_eval


def training_cnn(image, label, learning_rate = 0.005):
    out, loss, acc = cnn_forward_prop(image, label)
    gradient = np.zeros(10)
    gradient[label] = -1 / out[label]
    grad_back = softmax.back_prop(dL_dout = gradient, learning_rate = learning_rate)
    grad_back = maxpool.back_prop(dL_dout = grad_back)
    grad_back = conv.back_prop(dL_dout = grad_back, learning_rate = learning_rate)
    return loss, acc

epochs = 5
print("Training the CNN . . .")
for epoch in range(epochs):
    print("Epoch %d --->" % (epoch + 1))

    shuffle_data = np.random.permutation(len(train_images))
    train_images = train_images[shuffle_data]
    train_labels = train_labels[shuffle_data]

    loss = 0
    num_correct = 0
    for i, (im, label) in enumerate(zip(train_images, train_labels)):
        if i % 100 == 0:
            print("%d steps out of 100 steps: Average loss %.3f and Accuracy: %d%%" % (i + 1, loss / 100, num_correct))
            loss = 0
            num_correct = 0
        l1, accu = training_cnn(image = im, label = label)
        loss += l1
        num_correct += accu

print("Testing the CNN . . .")
loss = 0
num_correct = 0
for im, label in zip(test_images, test_labels):
    _, l1, accu = cnn_forward_prop(image = im, label = label)
    loss += 11
    num_correct += accu

num_tests = len(test_images)
print("Test loss :", loss/num_tests)
print("Test accuracy :", num_correct/num_tests)

sys.stdout.close()
