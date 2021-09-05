import os
import numpy as np
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow.keras.optimizers
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.activations import softmax
import matplotlib.pyplot as plt

sys.stdout = open("CNN_using_Keras output.text", "w")

path = "/Users/srimanikantaarjunkarimalammanavar/Desktop/Projects/Neural Networks from Scratch/CNN_from_Scratch/mnist.npz"
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data(path = path)
print("Imported MNIST data and split it into train and test to try our CNN model on it . . .")
print("X train shape :", X_train.shape)
print("Y train shape :", y_train.shape)
print("X test shape :", X_test.shape)
print("Y test shape :", y_test.shape)

image_index = 34        # (0, 60000)
plt.imshow(X_train[image_index], cmap = "gray")
plt.title(f"Image at index : {image_index}")
plt.show()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
print("X train shape :", X_train.shape)
print("X test shape :", X_test.shape)

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

X_train /= 255 - 0.5
X_test /= 255 - 0.5

num_filters = 8
filter_size = 3
filter_size_for_maxpool = 2
model = Sequential(layers = [Conv2D(filters = num_filters, kernel_size = filter_size, strides = (1, 1), input_shape = (28, 28, 1)),
                             MaxPooling2D(pool_size = filter_size_for_maxpool),
                             Flatten(),
                             Dense(units = 10, activation = softmax)],
                   name = "CNN_Model_using_keras")
print("Built a basic CNN model using keras . . .")

model.compile(optimizer = tensorflow.keras.optimizers.Adam(),
              loss = "categorical_crossentropy",
              metrics = ["accuracy"])
print("Compiled our CNN model using Adam optimizer and Categorical CrossEntropy as loss function . . .")
print("Training our model . . .")
model.fit(x = X_train, y = to_categorical(y_train), epochs = 5, verbose = 1,
          validation_data = (X_test, to_categorical(y_test)))
print("Evaluating our model . . .")
score = model.evaluate(x = X_test, y = to_categorical(y_test))
print("Test loss :", score[0])
print("Test accuracy :", score[1])

predictions = model.predict(x = X_test[:5])
print("Making predictions with our model . . .", np.argmax(predictions, axis = 1))

print("Comparing our predictions to our ground truth", y_test[:5])

sys.stdout.close()
