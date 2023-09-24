import cv2 as cv
import numpy as np
from keras.models import load_model
import os


face_cascade = cv.CascadeClassifier(r'haar_face1.xml')
eye_cascade = cv.CascadeClassifier(r'haar_eye.xml')

people = ['Adriana Lima', 'Alex Lawther', 'Alexandra Daddario', 'Barack Obama', 'Barbara Palvin','Yamini']
left_iris_recognizer = load_model('left_iris_recognition_model.keras')
right_iris_recognizer = load_model('right_iris_recognition_model.keras')



features = np.load('features.npy', allow_pickle=True)
labels = np.load('labels.npy')
face_recognizer = cv.face_LBPHFaceRecognizer.create()
face_recognizer.read('face_trained.yml')



img = cv.imread(r"C:\Users\yamini m r\Desktop\multi-modal-biometric-detection\projectpy\people\Barbara Palvin\barbara palvin240_875.jpg")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.6, minNeighbors=5)

for (x, y, w, h) in faces:
    face_region = gray[y:y+h, x:x+w]
    predicted_face_label, _ = face_recognizer.predict(face_region)

    if predicted_face_label >= 0 and predicted_face_label < len(people):
        recognized_person = people[predicted_face_label]
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  
        #cv.putText(img, recognized_person, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 10, (0, 255, 0), 2)
        #print("recognized person:", recognized_person)
    else:
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 0,255), 2)
        cv.putText(img,"Unknown", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        print("person is  not recognized as anyone")
        
   
    eyes = eye_cascade.detectMultiScale(face_region)
    
    

    for (ex, ey, ew, eh) in eyes:
        iris_region = face_region[ey:ey+eh, ex:ex+ew]
        
        iris_region=cv.resize(iris_region,(64,64))
        iris_region=iris_region.astype(np.float32)/255.0
        iris_region=np.expand_dims(iris_region,axis=0)
        iris_region=np.expand_dims(iris_region,axis=-1)
        iris_region=np.repeat(iris_region,3,axis=-1)
        
        
        left_iris_label = np.argmax(left_iris_recognizer.predict(iris_region))
        cv.rectangle(face_region, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

       
        right_iris_label = np.argmax(right_iris_recognizer.predict(iris_region))
        cv.rectangle(face_region, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
       
    cv.putText(img, recognized_person, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    print("recognized person :", recognized_person)
        
    # Display the left and right iris labels
    print("Left Iris Label:", left_iris_label)
    print("Right Iris Label:", right_iris_label)

cv.imshow('img', cv.resize(img,(800,600)))
cv.waitKey(0)
cv.destroyAllWindows()