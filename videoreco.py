from deepface import DeepFace
import cv2

cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()

    if not ret:
        break
    try:
        res=DeepFace.analyze(frame,actions=['age','gender','emotion'],enforce_detection=False)
        print(res)
    finally:
        print("No face detected")
    if cv2.waitKey(1):
        break
cap.release()
cv2.destroyAllWindows()