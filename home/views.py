from django.shortcuts import render
import cv2
from django.http import StreamingHttpResponse
import mediapipe as mp
import numpy as np
import random

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
cap = cv2.VideoCapture(0)

def generate_circle(left, right, up, down):
    x = random.randint(left, right)
    y = random.randint(up, down)
    radius = 20
    temp = random.randint(0,2)
    action = "uppercut"
    if temp == 0:
        color = (255,0,0)
    elif temp == 1:
        color = (0,255,0)
        action = "jab"
    else:
        color = (0,0,255)
        action = "hook"
    print(x, y, radius, color)
    return (x, y, radius, color, action)



def gen_frames():
    # Initialize MediaPipe Pose detection
    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        while True:
            success, frame = cap.read()
            if not success:
                break

            # Convert the BGR frame to RGB for processing
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            # Convert the image back to BGR for OpenCV drawing
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Extract necessary landmarks
                nose = landmarks[mp_pose.PoseLandmark.NOSE]
                right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

                # Calculate boundaries for the region based on landmark positions
                mid_hip = (left_hip.y + right_hip.y) / 2
                lower_bound = int((mid_hip + right_shoulder.y) / 2 * image.shape[0])
                higher_bound = int(right_eye.y * image.shape[0])
                nose_x = int(nose.x * image.shape[1])
                rhip_nose_distance = abs(right_hip.x - nose.x) * image.shape[1]
                lhip_nose_distance = abs(left_hip.x - nose.x) * image.shape[1]
                hip_nose_distance = round((rhip_nose_distance + lhip_nose_distance) / 2)
                left_bound = max(0, int(nose_x - 3 * hip_nose_distance))
                right_bound = min(image.shape[1], int(nose_x + 3 * hip_nose_distance))

                # Draw a rectangle in the calculated region
                cv2.rectangle(image, (left_bound, higher_bound), (right_bound, lower_bound), (0, 255, 0), 2)
                # Generate and draw a circle within the rectangle
                circle = generate_circle(left_bound, right_bound, higher_bound, lower_bound)
                cv2.circle(image, (circle[0], circle[1]), circle[2], circle[3], -1)

            # Encode the processed frame as JPEG
            _, buffer = cv2.imencode('.jpg', image)
            frame_bytes = buffer.tobytes()

            # Yield the frame in the format required by Django StreamingHttpResponse
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')



# Create your views here.
def index(request):
    return render(request, 'home/index.html')

def video_feed(request):
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def training(request):
    return render(request, 'home/training.html')