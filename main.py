import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Disable PyAutoGUI's fail-safe feature
pyautogui.FAILSAFE = False

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Get screen size
screen_width, screen_height = pyautogui.size()

# Initialize previous finger tip position
prev_x, prev_y = pyautogui.position()

# Smoothing factor (adjustable)
smoothing = 0.5

# Mouse speed factor (adjustable)
speed_factor = 1.5

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Variables for drag and scroll
is_dragging = False
scroll_mode = False
prev_scroll_y = 0

# Gesture state variables
left_click_frames = 0
right_click_frames = 0
drag_frames = 0
scroll_frames = 0

# Gesture threshold (number of consecutive frames to trigger action)
gesture_threshold = 5

# Gesture status for GUI
gesture_status = {
    "Left Click": False,
    "Right Click": False,
    "Drag": False,
    "Scroll": False
}


def get_finger_slope(point1, point2):
    return (point2.y - point1.y) / (point2.x - point1.x)


def distance(point1, point2):
    return np.linalg.norm(np.array([point1.x, point1.y]) - np.array([point2.x, point2.y]))


def draw_gesture_status(frame, gesture_status):
    height, width, _ = frame.shape
    for i, (gesture, status) in enumerate(gesture_status.items()):
        color = (0, 255, 0) if status else (0, 0, 255)
        cv2.putText(frame, f"{gesture}: {'Active' if status else 'Inactive'}",
                    (10, height - 30 - i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get finger landmarks
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]

            # Mouse movement (using index finger)
            x, y = int(index_tip.x *
                       screen_width), int(index_tip.y * screen_height)
            smooth_x = int(prev_x + smoothing * (x - prev_x))
            smooth_y = int(prev_y + smoothing * (y - prev_y))
            dx, dy = (smooth_x - prev_x) * \
                speed_factor, (smooth_y - prev_y) * speed_factor

            try:
                pyautogui.moveRel(dx, dy)
            except pyautogui.FailSafeException:
                print("Mouse movement limited to prevent going off-screen")

            prev_x, prev_y = smooth_x, smooth_y

            # Left click (index finger and thumb pinch)
            if distance(index_tip, thumb_tip) < 0.05:
                left_click_frames += 1
                if left_click_frames == gesture_threshold:
                    pyautogui.click()
                    gesture_status["Left Click"] = True
            else:
                left_click_frames = 0
                gesture_status["Left Click"] = False

            # Right click (middle finger and thumb pinch)
            if distance(middle_tip, thumb_tip) < 0.05:
                right_click_frames += 1
                if right_click_frames == gesture_threshold:
                    pyautogui.rightClick()
                    gesture_status["Right Click"] = True
            else:
                right_click_frames = 0
                gesture_status["Right Click"] = False

            # Drag mode (index and middle finger pinch)
            if distance(index_tip, middle_tip) < 0.05:
                drag_frames += 1
                if drag_frames == gesture_threshold:
                    if not is_dragging:
                        pyautogui.mouseDown()
                        is_dragging = True
                    gesture_status["Drag"] = True
            else:
                drag_frames = 0
                if is_dragging:
                    pyautogui.mouseUp()
                    is_dragging = False
                gesture_status["Drag"] = False

            # Two-finger scroll (index and middle finger extended, others closed)
            if (index_tip.y < index_pip.y and middle_tip.y < index_pip.y and
                    ring_tip.y > index_pip.y and thumb_tip.y > index_pip.y):
                scroll_frames += 1
                if scroll_frames == gesture_threshold:
                    scroll_mode = True
                    prev_scroll_y = (index_tip.y + middle_tip.y) / 2
                    gesture_status["Scroll"] = True
            elif scroll_mode:
                current_scroll_y = (index_tip.y + middle_tip.y) / 2
                # Adjust multiplier for scroll speed
                scroll_amount = (current_scroll_y - prev_scroll_y) * 1000
                pyautogui.scroll(-int(scroll_amount))
                prev_scroll_y = current_scroll_y
                if not (index_tip.y < index_pip.y and middle_tip.y < index_pip.y and
                        ring_tip.y > index_pip.y and thumb_tip.y > index_pip.y):
                    scroll_mode = False
                    scroll_frames = 0
                    gesture_status["Scroll"] = False
            else:
                scroll_frames = 0
                gesture_status["Scroll"] = False

    # Draw gesture status on the frame
    draw_gesture_status(frame, gesture_status)

    # Display the frame
    cv2.imshow('Air Mouse', frame)

    # Quit on 'q' key press
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
