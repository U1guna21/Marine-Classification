import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Set the path to the dataset folder
data_dir = 'Fish_Dataset'

# Set the parameters for image preprocessing
img_size = 299
batch_size = 32

# Load the data
data = []
labels = []
for folder in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, folder)
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg'):
                img_path = os.path.join(folder_path, filename)
                img = plt.imread(img_path)
                img = preprocess_input(img)
                data.append(img)
                labels.append(folder)

# Convert the data and labels to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Check if the dataset is empty
if len(labels) == 0:
    raise ValueError("The dataset is empty.")

# Split the data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

# Create the image generator for the training set
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
train_datagen.fit(train_data)

# Create the image generator for the testing set
test_datagen = ImageDataGenerator()
test_datagen.fit(test_data)

# Load the InceptionV3 model
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add the custom head for classification
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(16, activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 20
history = model.fit(train_datagen.flow(train_data, train_labels, batch_size=batch_size),
                    steps_per_epoch=len(train_data)//batch_size,
                    epochs=epochs,
                    validation_data=test_datagen.flow(test_data, test_labels, batch_size=batch_size),
                    validation_steps=len(test_data)//batch_size)

# Save the model
model.save('fish_species_classification_model.h5')

# Plot the training and testing accuracy and loss curves

