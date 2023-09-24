import os 
import cv2 as cv
import numpy as np
import time

people = ['Adriana Lima', 'Alex Lawther', 'Alexandra Daddario', 'Barack Obama', 'Barbara Palvin','Yamini']

DIR = r'C:\Users\yamini m r\Desktop\multi-modal-biometric-detection\projectpy\people'
face_cascade = cv.CascadeClassifier(r'haar_face1.xml')

features = []
labels = []

def create_train():
    total_datasets = 0
    start_time = time.time()

    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            img_array = cv.imread(img_path)
            total_datasets += 1

            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            faces_rect = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces_rect:
                face = gray[y:y+h, x:x+w]
                features.append(face)
                labels.append(label)

    end_time = time.time()
    time_taken = end_time - start_time
    return time_taken, total_datasets

def test_face_recognizer(face_recognizer, features, labels):
    DIR1=r'C:\Users\yamini m r\Desktop\multi-modal-biometric-detection\projectpy\test-of-people'
    total_correct = 0
    total_images = 0

    for person in people:
        path = os.path.join(DIR1, person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            faces_rect = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces_rect:
                face = gray[y:y+h, x:x+w]
                label, _ = face_recognizer.predict(face)

                if label == people.index(person):
                    total_correct += 1
                total_images += 1
    
    accuracy = total_correct / total_images
    return accuracy

# Training
time_taken, total_datasets = create_train()
print('Training done')
print(f"Time taken for training  datasets: {time_taken} seconds")

features = np.array(features, dtype='object')
labels = np.array(labels)
face_recognizer = cv.face_LBPHFaceRecognizer.create()

face_recognizer.train(features, labels)

face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)

# Testing
face_recognizer = cv.face_LBPHFaceRecognizer.create()
face_recognizer.read('face_trained.yml')
features = np.load('features.npy', allow_pickle=True)
labels = np.load('labels.npy')

accuracy = test_face_recognizer(face_recognizer, features, labels)
print("Testing done")
print("Accuracy:", accuracy)
