import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def get_gesture_name(hand_landmarks, handedness):
    fingers = []
    tips = [4, 8, 12, 16, 20]
    
    if handedness == "Right":  # For right hand
        if hand_landmarks.landmark[tips[0]].x < hand_landmarks.landmark[tips[0] - 1].x:
            fingers.append(1)
        else:
            fingers.append(0)
    else:  # For left hand
        if hand_landmarks.landmark[tips[0]].x > hand_landmarks.landmark[tips[0] - 1].x:
            fingers.append(1)
        else:
            fingers.append(0)

    for tip in tips[1:]: 
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    total_fingers = sum(fingers)
    if total_fingers == 0:
        return "Fist"
    elif total_fingers == 5:
        return "Open Hand"
    elif total_fingers == 1:
        return "Pointing"
    else:
        return f"{total_fingers} Fingers Up"

cap = cv2.VideoCapture(1)

with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks and result.multi_handedness:
            for hand_landmarks, handedness_info in zip(result.multi_hand_landmarks, result.multi_handedness):
                handedness = handedness_info.classification[0].label
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture_name = get_gesture_name(hand_landmarks, handedness)
                h, w, _ = frame.shape
                cx, cy = int(hand_landmarks.landmark[0].x * w), int(hand_landmarks.landmark[0].y * h)
                cv2.putText(frame, f"{handedness}: {gesture_name}", (cx, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Hand Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
