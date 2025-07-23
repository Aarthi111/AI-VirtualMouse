import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Get screen size
screen_width, screen_height = pyautogui.size()

# Function to recognize gestures
def recognize_gesture(hand_landmarks):
    # Get the coordinates of the index finger tip
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_x = int(index_finger_tip.x * screen_width)
    index_y = int(index_finger_tip.y * screen_height)

    # Get the coordinates of the thumb tip
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_x = int(thumb_tip.x * screen_width)
    thumb_y = int(thumb_tip.y * screen_height)

    # Calculate distance between index finger and thumb
    distance = np.sqrt((index_x - thumb_x) ** 2 + (index_y - thumb_y) ** 2)

    # Define gesture thresholds
    if distance < 50:  # Close together for click
        return "click"
    elif index_y < screen_height / 2:  # Move up
        return "move_up"
    elif index_y > screen_height / 2:  # Move down
        return "move_down"
    else:
        return "move"

# Main loop
while True:
    success, image = cap.read()
    if not success:
        break

    # Flip the image horizontally for a mirror effect
    image = cv2.flip(image, 1)

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and find hands
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the image
            mp.solutions.drawing_utils.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Recognize gesture
            gesture = recognize_gesture(hand_landmarks)

            # Get the index finger tip position
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            cursor_x = int(index_finger_tip.x * screen_width)
            cursor_y = int(index_finger_tip.y * screen_height)

            # Move the cursor
            pyautogui.moveTo(cursor_x, cursor_y)

            # Perform click action if gesture is recognized
            if gesture == "click":
                pyautogui.click()

    # Display the image
    cv2.imshow("Virtual Mouse", image)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
