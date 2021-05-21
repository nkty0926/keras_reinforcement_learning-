'''
// Main File:        intro_keras.py
// Semester:         CS 540 Fall 2020
// Authors:          Tae Yong Namkoong
// CS Login:         namkoong
// NetID:            kiatvithayak
// References:       TA's & Peer Mentor's Office Hours
                     https://www.tensorflow.org/guide/keras/sequential_model
                     https://keras.io/api/layers/
                     https://stackoverflow.com/questions/57318863/knowing-what-to-put-into-a-keras-dense-layer
                     https://stackoverflow.com/questions/56622231/how-to-train-keras-models-consecutively
'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import matplotlib.pyplot as plt
import numpy as np

def get_dataset(training=True):
    """
    Input: an optional boolean argument (default value is True for training dataset)
    Return: two NumPy arrays for the train_images and train_labels
    """
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    if training:
        return train_images, train_labels
    else:
        return test_images, test_labels

def print_stats(train_images, train_labels):
    '''
    This function will print several statistics about the data
    Input: the dataset and labels produced by the previous function; does not return anything
    '''
    # Total # of images in given dataset
    num_dataset = len(train_images)

    # Image Dimension (height, width = 28 x 28 grayscale)
    width = len(train_images[0])
    height = len(train_images[0][0])

    # create list of dict for class_names
    class_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']
    num_occurrence = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in train_labels:
        num_occurrence[i] += 1

    # print corresponding stats about data
    print(num_dataset)
    print(str(width) + 'x' + str(height))
    for i in range(len(class_names)):
        print(str(i) + ". " + str(class_names[i]) + " - " + str(num_occurrence[i]))

def build_model():
    '''
    takes no arguments and returns an untrained neural network model
    '''
    # Create sequential obj to hold layers
    model = keras.models.Sequential()
    # A Flatten layer to convert the 2D pixel array to a 1D array of numbers; Specify the input shape here
    # based on your results from print_stats() above.
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    # A Dense layer with 128 nodes and a ReLU activation
    model.add(keras.layers.Dense(128, activation='relu'))
    # A Dense layer with 64 nodes and a ReLU activation.
    model.add(keras.layers.Dense(64,activation='relu'))
    # A Dense layer with 10 nodes.
    model.add(keras.layers.Dense(10))

    opt = keras.optimizers.SGD(learning_rate=0.001)
    # Compile model using the params: SGD w/ learning rate of 0.001, loss function with sparse
    # categorical class-entropy and metrics = accuracy
    model.compile(opt, loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    return model


def train_model(model, train_images, train_labels, T):
    '''takes the model produced by the previous function and the dataset and labels produced by the first function
    and trains the data for T epochs; does not return anything
    '''
    # Use model.fit function with the training images and labels, with the number of epochs from the parameters
    model.fit(train_images, train_labels, epochs=T)

def evaluate_model(model, test_images, test_labels, show_loss=True):
    ''' takes the trained model produced by the previous function and the test image/labels, and prints the evaluation
    statistics as described below (displaying the loss metric value if and only if the optional parameter
    has not been set to False); does not return anything
    '''
    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose='0')
    if show_loss:
        # Format the loss with four decimal places.
        print("Loss: {:.4f}".format(test_loss))
    # Format the accuracy output with two decimal place as %age
    print("Accuracy: {:.2f}%".format(test_accuracy * 100.0))

def predict_label(model, test_images, index):
    '''
    takes the trained model and test images, and prints the top 3 most likely labels for the image at the given index,
    along with their probabilities; does not return anything
    '''
    predicted = model.predict(test_images)
    prediction = predicted[index]
    sorted_labels = np.argsort(prediction)

    class_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']
    # display the top three most likely class labels for the image at the given index with respective probalities
    print("{}: {:.2f}%".format(class_names[sorted_labels[-1]], prediction[sorted_labels[-1]] * 100.0))
    print("{}: {:.2f}%".format(class_names[sorted_labels[-2]], prediction[sorted_labels[-2]] * 100.0))
    print("{}: {:.2f}%".format(class_names[sorted_labels[-3]], prediction[sorted_labels[-3]] * 100.0))