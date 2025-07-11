import cv2
import mediapipe as mp
from pynput.keyboard import Key, Controller

# Initialize MediaPipe Hand Detection
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Initialize Keyboard Controller
keyboard = Controller()

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(3, 720)
cap.set(4, 480)

# Finger tip landmarks
finger_tips = [4, 8, 12, 16, 20]


def count_raised_fingers(landmarks):
    fingers_up = 0
    for tip_id in finger_tips[1:]:  # skip thumb
        if landmarks[tip_id].y < landmarks[tip_id - 2].y:
            fingers_up += 1
    return fingers_up


def show_feedback(image, message, color):
    cv2.putText(image, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


def main():
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Error: Cannot access webcam.")
            continue

        # Flip image and convert to RGB
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hands
        rgb_frame.flags.writeable = False
        results = hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        # Convert back to BGR for OpenCV
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                raised = count_raised_fingers(hand_landmarks.landmark)

                if raised >= 3:
                    keyboard.press(Key.right)
                    keyboard.release(Key.left)
                    show_feedback(frame, "Accelerate →", (0, 255, 0))
                else:
                    keyboard.press(Key.left)
                    keyboard.release(Key.right)
                    show_feedback(frame, "Brake ←", (0, 255, 255))
        else:
            keyboard.release(Key.left)
            keyboard.release(Key.right)
            show_feedback(frame, "No Hand Detected", (0, 0, 255))

        cv2.imshow("Gesture-Controlled Hill Climb Racing", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
