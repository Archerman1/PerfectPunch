import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

ret, frame = cap.read()
frame_height, frame_width = frame.shape[:2]

def is_protecting_head(nose, wrist, threshold=0.1):
    return (wrist.y < nose.y) and (abs(wrist.x - nose.x) < threshold)

def is_protecting_body(shoulder, elbow, threshold=0.1):
    return (elbow.y > shoulder.y) and (abs(elbow.x - shoulder.x) < threshold)

def draw_box(frame, text, position, is_highlighted):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    color = (0, 255, 0) if is_highlighted else (255, 255, 255)
    bg_color = (0, 200, 0) if is_highlighted else (200, 200, 200)

    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    cv2.rectangle(frame, position, (position[0] + text_size[0] + 10, position[1] + text_size[1] + 10), bg_color, -1)
    cv2.putText(frame, text, (position[0] + 5, position[1] + text_size[1] + 5), font, font_scale, color, thickness)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb_frame)

    protecting_head = False
    protecting_body = False

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        left_mouth = results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_LEFT]
        right_mouth = results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT]
        left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]

        protecting_head = is_protecting_head(left_mouth, left_wrist) or is_protecting_head(right_mouth, right_wrist)
        protecting_body = is_protecting_body(left_shoulder, left_elbow) or is_protecting_body(right_shoulder, right_elbow)

    draw_box(frame, "HEAD", (10, 30), protecting_head)
    draw_box(frame, "BODY", (10, 80), protecting_body)

    cv2.imshow("Pose Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()