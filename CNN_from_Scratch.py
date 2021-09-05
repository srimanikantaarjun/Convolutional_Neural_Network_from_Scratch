import cv2
import matplotlib.pyplot as plt
import MaxPool
import ConvOp
import Activations
import sys

# Using sys.stdout to print the output of this script to a text file
sys.stdout = open("CNN_from_Scratch_output.text", "w")

# Reading a sample image to test our implementation of Convolutional Neural Network
sample_image = cv2.imread("WhatsApp_Image_2021-09-04_at_21.59.00_225x225.jpeg", cv2.IMREAD_GRAYSCALE)/255
plt.imshow(X = sample_image, cmap = "gray")
plt.show()
print("Sample Image Shape :", sample_image.shape)

# Applying our Convolutional layer to our sample image
cnn = ConvOp.ConvOp(num_filters = 18, filter_size = 7)
out = cnn.forward_prop(image = sample_image)
print("Sample Image Shape after applying a Convolutional layer :", out.shape)
plt.imshow(out[:, :, 17], cmap = "gray")
plt.title("Plotting the 17th image of size (219, 219)")
plt.show()

# Applying our MaxPool layer to the output of our Convolutional layer
cnn2 = MaxPool.Max_Pool(filter_size = 4)
out2 = cnn2.forward_prop(image = out)
print("Sample Image Shape after applying a MaxPool layer :", out2.shape)
plt.title("Plotting the 17th image of size (54, 54)")
plt.imshow(out2[:, :, 17], cmap = "gray")
plt.show()

# Applying our Softmax layer to the output of MaxPool layer
cnn3 = Activations.Softmax(input_node = 54*54*18, softmax_node = 10)
out3 = cnn3.forward_prop(image = out2)
print("Softmax prediction outputs :", out3)

sys.stdout.close()
