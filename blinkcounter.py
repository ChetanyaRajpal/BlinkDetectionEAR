import mediapipe as mp
import cv2
import numpy as np

mp_drawing = mp.solutions.drawing_utils  # Drawing helpers
mp_holistic = mp.solutions.holistic     # Mediapipe Solutions

def calculate_EAR(landmarks, eye_indices):
    # Calculate the distances between vertical landmarks
    vertical_1 = np.linalg.norm(
        np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y]) -
        np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y])
    )
    vertical_2 = np.linalg.norm(
        np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y]) -
        np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y])
    )
    
    # Calculate the distance between horizontal landmarks
    horizontal = np.linalg.norm(
        np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y]) -
        np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y])
    )
    
    # Calculate the EAR
    EAR = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return EAR

# EAR threshold for blink detection
EAR_THRESHOLD = 0.30

cap = cv2.VideoCapture(0)
blink_count = 0
eye_closed = False  # Tracks whether the eyes are currently closed

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Unable to access camera")
            break

        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make Detections
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.face_landmarks:
            # Get face landmarks
            landmarks = results.face_landmarks.landmark

            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                  mp_drawing.DrawingSpec(color = (80, 110, 10), thickness=1, circle_radius = 1),
                                  mp_drawing.DrawingSpec(color = (80, 256, 121), thickness=1, circle_radius = 1))
            # Left and right eye indices
            left_eye_indices = [33, 160, 158, 133, 153, 144]
            right_eye_indices = [362, 385, 387, 263, 373, 380]

            # Calculate EAR for both eyes
            left_EAR = calculate_EAR(landmarks, left_eye_indices)
            right_EAR = calculate_EAR(landmarks, right_eye_indices)

            # Average EAR
            avg_EAR = (left_EAR + right_EAR) / 2.0

            # Detect blink
            if avg_EAR < EAR_THRESHOLD:  # Eyes are closed
                eye_closed = True
            else:  # Eyes are open
                if eye_closed:  # If eyes were previously closed, it's a blink
                    blink_count += 1
                    print(f"Blink detected! Total count: {blink_count}")
                eye_closed = False

            # Display EAR and Blink Count on Screen
            cv2.putText(image, f'EAR: {avg_EAR:.2f}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f'Blinks: {blink_count}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show the image
        cv2.imshow('Blink Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
