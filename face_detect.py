import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("mask_detector.h5")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (128, 128))
        face_img = face_img / 255.0
        face_img = np.expand_dims(face_img, axis=0)
        
        prediction = model.predict(face_img)
        label = "Mask" if prediction > 0.5 else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        
        cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    cv2.imshow('Mask Detector', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
