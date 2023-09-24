import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import re  

left_iris_dir = r"C:\Users\yamini m r\Desktop\multi-modal-biometric-detection\projectpy\eye_image\left_eye"
right_iris_dir = r"C:\Users\yamini m r\Desktop\multi-modal-biometric-detection\projectpy\eye_image\right_eye"

image_height = 64
image_width = 64
num_channels = 3 
num_classes = 10 

X_left = []
y_left = []
X_right = []
y_right = []

def preprocess_label(label):
    # Remove special characters and spaces, and convert to lowercase
    return re.sub(r'[^a-zA-Z0-9]', '_', label.lower())

def load_images_from_directory(directory):
    X = []
    y = []
    for person_dir in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, person_dir)):
            for filename in os.listdir(os.path.join(directory, person_dir)):
                if filename.endswith('.jpg'):
                    img = cv2.imread(os.path.join(directory, person_dir, filename))
                    img = cv2.resize(img, (image_width, image_height))  # Resize images to a consistent size
                    X.append(img)
                   
                    label = preprocess_label(person_dir)
                    y.append(label)
    return np.array(X), np.array(y)


X_left, y_left = load_images_from_directory(left_iris_dir)


X_right, y_right = load_images_from_directory(right_iris_dir)


label_encoder = LabelEncoder()
y_left_encoded = label_encoder.fit_transform(y_left)
y_right_encoded = label_encoder.fit_transform(y_right)

y_left_onehot = to_categorical(y_left_encoded, num_classes)
y_right_onehot = to_categorical(y_right_encoded, num_classes)

# Perform normalization (you can use MinMaxScaler or other methods)
scaler = MinMaxScaler()
X_left_normalized = scaler.fit_transform(X_left.reshape(-1, image_height * image_width * num_channels))
X_left_normalized = X_left_normalized.reshape(-1, image_height, image_width, num_channels)

X_right_normalized = scaler.fit_transform(X_right.reshape(-1, image_height * image_width * num_channels))
X_right_normalized = X_right_normalized.reshape(-1, image_height, image_width, num_channels)

# Split the left iris dataset into training, validation, and testing sets
X_left_train, X_left_temp, y_left_train, y_left_temp = train_test_split(X_left_normalized, y_left_onehot, test_size=0.4, random_state=42)
X_left_val, X_left_test, y_left_val, y_left_test = train_test_split(X_left_temp, y_left_temp, test_size=0.5, random_state=42)

# Split the right iris dataset into training, validation, and testing sets
X_right_train, X_right_temp, y_right_train, y_right_temp = train_test_split(X_right_normalized, y_right_onehot, test_size=0.4, random_state=42)
X_right_val, X_right_test, y_right_val, y_right_test = train_test_split(X_right_temp, y_right_temp, test_size=0.5, random_state=42)

# Data augmentation 
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.1,
)

# Create and compile the model for the left iris
model_left = Sequential()
model_left.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, num_channels)))
model_left.add(MaxPooling2D((2, 2)))
model_left.add(Conv2D(64, (3, 3), activation='relu'))
model_left.add(MaxPooling2D((2, 2)))
model_left.add(Conv2D(128, (3, 3), activation='relu'))
model_left.add(MaxPooling2D((2, 2)))
model_left.add(Flatten())
model_left.add(Dense(256, activation='relu'))
model_left.add(Dropout(0.5))
model_left.add(Dense(num_classes, activation='softmax'))

model_left.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model for the left iris with data augmentation
model_left.fit(datagen.flow(X_left_train, y_left_train, batch_size=32), epochs=10, validation_data=(X_left_val, y_left_val))

# Evaluate the model for the left iris
test_loss_left, test_acc_left = model_left.evaluate(X_left_test, y_left_test)
print(f'Left Iris Test accuracy: {test_acc_left}')

model_left.save('left_iris_recognition_model.keras')

# Define the CNN model for the right iris (similar to left iris)
model_right = Sequential()
model_right.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, num_channels)))
model_right.add(MaxPooling2D((2, 2)))
model_right.add(Conv2D(64, (3, 3), activation='relu'))
model_right.add(MaxPooling2D((2, 2)))
model_right.add(Conv2D(128, (3, 3), activation='relu'))
model_right.add(MaxPooling2D((2, 2)))
model_right.add(Flatten())
model_right.add(Dense(256, activation='relu'))
model_right.add(Dropout(0.5))
model_right.add(Dense(num_classes, activation='softmax'))

model_right.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model for the right iris with data augmentation
model_right.fit(datagen.flow(X_right_train, y_right_train, batch_size=32), epochs=10, validation_data=(X_right_val, y_right_val))

# Evaluate the model for the right iris
test_loss_right, test_acc_right = model_right.evaluate(X_right_test, y_right_test)
print(f'Right Iris Test accuracy: {test_acc_right}')

model_right.save('right_iris_recognition_model.keras')