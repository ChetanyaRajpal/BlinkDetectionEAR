import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Holistic
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# EAR threshold
EAR_THRESHOLD = 0.25

def calculate_EAR(eye_landmarks):
    """Calculate Eye Aspect Ratio (EAR) using eye landmarks."""
    # Vertical distances
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    # Horizontal distance
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    # EAR formula
    EAR = (A + B) / (2.0 * C)
    return EAR

def detect_sclera(eye_region):
    """Detect the amount of white region (sclera) in the cropped eye area."""
    gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    _, sclera_mask = cv2.threshold(gray_eye, 200, 255, cv2.THRESH_BINARY)
    white_pixel_percentage = (np.sum(sclera_mask == 255) / sclera_mask.size) * 100
    return white_pixel_percentage > 1  # Adjust threshold as needed

# Start capturing video
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame with Mediapipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.face_landmarks:
            h, w, _ = image.shape
            landmarks = results.face_landmarks.landmark

            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                  mp_drawing.DrawingSpec(color = (80, 110, 10), thickness=1, circle_radius = 1),
                                  mp_drawing.DrawingSpec(color = (80, 256, 121), thickness=1, circle_radius = 1))
            
            # Get left and right eye landmarks
            left_eye_points = [463, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374,
                                   380, 381, 382, 362]
            right_eye_points = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145,
                                    144, 163, 7]           
            left_eye_points = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in left_eye_points])
            right_eye_points = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in right_eye_points])

            # EAR calculation
            left_EAR = calculate_EAR(left_eye_points)
            right_EAR = calculate_EAR(right_eye_points)

            # Average EAR for both eyes
            avg_EAR = (left_EAR + right_EAR) / 2.0

            # Crop eye regions for sclera detection
            left_eye_box = (int(left_eye_points[:, 0].min()), int(left_eye_points[:, 1].min()),
                            int(left_eye_points[:, 0].max()), int(left_eye_points[:, 1].max()))
            right_eye_box = (int(right_eye_points[:, 0].min()), int(right_eye_points[:, 1].min()),
                             int(right_eye_points[:, 0].max()), int(right_eye_points[:, 1].max()))

            left_eye_region = image[left_eye_box[1]:left_eye_box[3], left_eye_box[0]:left_eye_box[2]]
            right_eye_region = image[right_eye_box[1]:right_eye_box[3], right_eye_box[0]:right_eye_box[2]]

            # Detect sclera
            left_sclera_detected = detect_sclera(left_eye_region) if left_eye_region.size > 0 else False
            right_sclera_detected = detect_sclera(right_eye_region) if right_eye_region.size > 0 else False

            # Decision Logic
            if avg_EAR < EAR_THRESHOLD and not (left_sclera_detected or right_sclera_detected):
                eye_state = "Closed"
            else:
                eye_state = "Open"

            # Display EAR and state
            cv2.putText(image, f"EAR: {avg_EAR:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"Eye State: {eye_state}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow('Blink Detection', image)

        # Exit on pressing 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
