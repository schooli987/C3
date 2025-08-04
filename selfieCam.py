import cv2
import time
from datetime import datetime
from playsound import playsound
import random

# Load 
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start webcam
cap = cv2.VideoCapture(0)

face_detected = False
start_time = None
countdown_duration = 3  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    clean_frame = frame.copy() 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    if len(faces) > 0:
        # Draw bounding box 
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if not face_detected:
            print("Face detected. Starting 3-second timer...")
            face_detected = True
            start_time = time.time()
        else:
            elapsed = time.time() - start_time
            remaining = int(countdown_duration - elapsed + 1)
            if remaining > 0:
               
                cv2.putText(frame, f"{remaining}", (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            if elapsed >= countdown_duration:
                playsound("camera_click.mp3")
                filename = f"{(random.randint(1000,9999))}"+".jpg"
                cv2.imwrite(filename, clean_frame)
                break
    else:
        face_detected = False
        start_time = None

    cv2.imshow("Face Detection Selfie Cam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exited manually.")
        break

cap.release()
cv2.destroyAllWindows()
