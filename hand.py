import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def recognize_gesture(landmarks):
    thumb_is_open = landmarks[4].x < landmarks[3].x < landmarks[2].x
    index_is_open = landmarks[8].y < landmarks[6].y
    middle_is_open = landmarks[12].y < landmarks[10].y
    ring_is_open = landmarks[16].y < landmarks[14].y
    pinky_is_open = landmarks[20].y < landmarks[18].y
    
    if thumb_is_open and index_is_open and middle_is_open and ring_is_open and pinky_is_open:
        return "Open Hand"
    elif not thumb_is_open and index_is_open and not middle_is_open and not ring_is_open and not pinky_is_open:
        return "Pointing"
    elif not thumb_is_open and index_is_open and middle_is_open and not ring_is_open and not pinky_is_open:
        return "Victory"
    elif not thumb_is_open and not index_is_open and not middle_is_open and not ring_is_open and not pinky_is_open:
        return "Fist"
    elif not thumb_is_open and not index_is_open and not middle_is_open and ring_is_open and pinky_is_open:
        return "Rock Sign"
    elif thumb_is_open and not index_is_open and not middle_is_open and not ring_is_open and not pinky_is_open:
        return "Thumbs Up"
    elif not thumb_is_open and not index_is_open and not middle_is_open and not ring_is_open and not pinky_is_open and landmarks[4].y > landmarks[3].y > landmarks[2].y:
        return "Thumbs Down"
    elif thumb_is_open and index_is_open and not middle_is_open and not ring_is_open and pinky_is_open:
        return "OK Sign"
    elif not thumb_is_open and index_is_open and middle_is_open and ring_is_open and not pinky_is_open:
        return "Three Fingers Up"
    elif not thumb_is_open and index_is_open and middle_is_open and ring_is_open and pinky_is_open:
        return "Four Fingers Up"
    elif thumb_is_open and not index_is_open and not middle_is_open and not ring_is_open and pinky_is_open:
        return "Call Me"
    elif not thumb_is_open and index_is_open and middle_is_open and not ring_is_open and not pinky_is_open:
        return "Peace Sign"
    elif thumb_is_open and index_is_open and middle_is_open and ring_is_open and pinky_is_open and all(
        [abs(landmarks[8].x - landmarks[4].x) > 0.1, abs(landmarks[12].x - landmarks[4].x) > 0.1,
        abs(landmarks[16].x - landmarks[4].x) > 0.1, abs(landmarks[20].x - landmarks[4].x) > 0.1]):
        return "Five Fingers Spread"
    elif not thumb_is_open and not index_is_open and not middle_is_open and not ring_is_open and pinky_is_open:
        return "Palm Closed with Thumb Up"
    elif not thumb_is_open and not index_is_open and not middle_is_open and not ring_is_open and not pinky_is_open and landmarks[4].y > landmarks[3].y:
        return "Palm Closed with Thumb Down"
    elif thumb_is_open and not index_is_open and not middle_is_open and not ring_is_open and pinky_is_open:
        return "Shaka Sign"
    elif thumb_is_open and index_is_open and not middle_is_open and not ring_is_open and not pinky_is_open:
        return "Finger Gun"
    
    else:
        return "Unknown Gesture"

def main():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        
        frame = cv2.flip(frame, 1)
        
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        

        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
            
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
            
                gesture = recognize_gesture(hand_landmarks.landmark)
                cv2.putText(frame, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        
        cv2.imshow('Hand Gesture Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()