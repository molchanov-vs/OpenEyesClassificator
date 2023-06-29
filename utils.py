import json
import random
import numpy as np
from tqdm import tqdm
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report


def analyze_data(file_path):
    data = read_json(file_path)
    labels, counts = np.unique(np.array(list(data.values())), return_counts=True)
    print('Labels:', *labels)
    print('Counts:', *counts)


def read_json(file_path):

    with open(file_path, "r") as json_file:
        data = json.load(json_file)
    
    return data


def load_image(path):
    return np.array(Image.open(path)).reshape(-1, 24, 24, 1).astype('float32') / 255


def load_data(path, type_):
    X = []
    Y = []

    for i in tqdm(glob(f'{path}/{type_}/opened/*')):
        temp = np.array(Image.open(i))
        X.append(temp)
        Y.append(1)
        
    for i in tqdm(glob(f'{path}/{type_}/closed/*')):
        temp = np.array(Image.open(i))
        X.append(temp)
        Y.append(0)

    X = np.array(X)
    Y = np.array(Y)

    X = X.reshape(-1, 24, 24, 1)
    # X = X.astype('float32')
    # Y = Y.astype('float32')
    X = X / 255

    return X, Y


def show_examples(train_X, train_Y):
    print()
    print('Пример размеченных изображений:')
    class_names = ['Closed', 'Opened']
    plt.figure(figsize=(12, 10))
    for i in range(15):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        index = random.randint(0, len(train_X))
        plt.imshow(train_X[index], cmap=plt.cm.gray)
        plt.xlabel(class_names[int(train_Y[index])])
    plt.show()


def show_examples_of_predicted_images(predicted_labels, test_X):
    values = predicted_labels
    sorted_values = sorted(values)
    sorted_positions = [index for index, _ in sorted(enumerate(values), key=lambda x: x[1])]

    bad_pos = []
    for i, el in zip(sorted_positions, sorted_values):
        if el > 0.4 and el < 0.6:
            bad_pos.append(i)

    
    plt.figure(figsize=(8, 6))
    for i, pos in enumerate(sorted_positions[:3] + bad_pos[4:7] + sorted_positions[-3:]):
        plt.subplot(3,3,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(test_X[pos], cmap=plt.cm.gray)
        plt.xlabel(values[pos][0])
    plt.show()

    # for i, el in enumerate(sorted(predicted_labels)):
    # if el > 0.4 and el < 0.6:
    #     print(el)


def split_train(train_X, train_Y, test_size=0.2):
        return train_test_split(train_X, train_Y, test_size=test_size, random_state=42)


def show_plots(trained_model):

    accuracy = trained_model.history['accuracy']
    val_accuracy = trained_model.history['val_accuracy']
    loss = trained_model.history['loss']
    val_loss = trained_model.history['val_loss']
    epochs = range(len(accuracy))

    plt.figure(figsize=(10, 4))  # Adjust the figure size as per your preference

    # Plotting the first subplot (Training and validation accuracy)
    plt.subplot(1, 2, 1)
    plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.grid(True)
    plt.legend()

    # Plotting the second subplot (Training and validation loss)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.tight_layout()  # Adjusts the spacing between subplots
    plt.grid(True)
    plt.show()


def computer_eer(fpr, tpr):
    fnr = 1 - tpr
    return fnr[np.nanargmin(np.absolute((fnr - fpr)))]


def report(test_Y, predicted_classes):

    # Compute the AUC
    fpr, tpr, _ = roc_curve(test_Y, predicted_classes)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    target_names = ['Opened', 'Closed']
    predicted_classes = np.round(predicted_classes)
    print(classification_report(test_Y, predicted_classes, target_names=target_names))
    print(12*'#')
    print(f"EER: {computer_eer(fpr, tpr):.2}")
    print(12*'#')






