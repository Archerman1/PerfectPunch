from django.shortcuts import render
import cv2
from django.http import StreamingHttpResponse
import mediapipe as mp
import numpy as np
import random
from google import genai
from google.genai import types
import time
import pandas as pd


#Google Gemini declarations
API_KEY= "AIzaSyBRkSPbu_9L2anGOae1yoybstpgNveWXgY"
client = genai.Client(api_key=API_KEY)
sys_instruct = """You are a boxing form analysis chatbot. You will receive punch type, 
                                    reaction time of punch and an image of the punch. Based on this input provide the following info in a concise easy to read paragraph:
                                    1. Stance evaluation
                                    2. Punch technique analysis
                                    3. Detailed breakdown of deviations in stance and punch
                                    4. Comparison to professional boxing standards
                                    5. Suggestions for improvement
                                    6. Additional tips for training and practice"""

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
screen_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
screen_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
ret, frame = cap.read()
frame_height, frame_width = frame.shape[:2]
left_bound = 0
right_bound = screen_width
lower_bound = screen_height
higher_bound = 0
circles = []
max_circles = 1
startTime = time.time()
hits = 0

obstacle_direction = "left"
obstacle = None
last_obstacle_time = time.time()
min_interval = 6
max_interval = 10
hit = False

wrist_paths = []
elbow_paths = []
current_wrist_path = []
current_elbow_path = []
plots = []
handsUsed = []
punchType = []
handUsed = "left"
actions = []


reaction_times = []
jab_hits = 0
jab_attempts = 0
uppercut_hits = 0
uppercut_attempts = 0
hook_hits = 0
hook_attempts = 0
total_time = 0
head_def_time = 0
body_def_time = 0
total_left_obstacles = 0
total_right_obstacles = 0
left_hit = 0
right_hit = 0
stances = []
send_image = None


def create_obstacle_right(y):
    global total_right_obstacles
    total_right_obstacles = total_right_obstacles + 1
    return {
        'x': screen_width,
        'y': y,
        'width': 30,
        'height': 30,
        'speed': 4,
        'side': 'right'
    }

def create_obstacle_left(y):
    global total_left_obstacles
    total_left_obstacles = total_left_obstacles + 1
    return {
        'x': 0,
        'y': y,
        'width': 30,
        'height': 30,
        'speed': 4,
        'side': 'left'
    }

def move_obstacle_right(obs):
    obs['x'] -= obs['speed']

def move_obstacle_left(obs):
    obs['x'] += obs['speed']

def draw_obstacle(frame, obs):
    cv2.rectangle(frame, (obs['x'], obs['y']), (obs['x'] + obs['width'], obs['y'] + obs['height']), (0, 0, 255), -1)

def is_off_screen(obs):
    return obs['x'] + obs['width'] < 0 or obs['x'] > screen_width

def check_collision(obs, eye_y, mouth_y, left_eye_x, right_eye_x):
    eye_mouth_distance = abs(eye_y - mouth_y)
    return (eye_y - eye_mouth_distance) < obs['y'] < (eye_y + eye_mouth_distance) and right_eye_x < obs['x'] < left_eye_x

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

def generate_circle(left, right, up, down):
    global hook_attempts
    global jab_attempts
    global uppercut_attempts
    x = random.randint(left, right)
    y = random.randint(up, down)
    radius = 25
    temp = random.randint(0,2)
    action = "hook"
    if temp == 0:
        color = (255,0,0)
        hook_attempts = hook_attempts + 1
    elif temp == 1:
        color = (0,255,0)
        action = "jab"
        jab_attempts = jab_attempts + 1
    else:
        color = (0,0,255)
        action = "uppercut"
        uppercut_attempts = uppercut_attempts + 1
    return (x, y, radius, color, action)

def gen_frames():

    global total_time, hits, body_def_time, head_def_time, left_hit, right_hit, \
        jab_attempts, hook_attempts, uppercut_attempts, circles, actions, startTime, \
        obstacle, last_obstacle_time, current_wrist_path, current_elbow_path, handUsed, \
        punchType, handsUsed, wrist_paths, elbow_paths, stances, reaction_times, obstacle_direction, send_image
    # Initialize MediaPipe Pose detection
    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        while total_time < 660:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            total_time = total_time + 1
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Process the frame with MediaPipe Pose
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                nose = landmarks[mp_pose.PoseLandmark.NOSE]
                right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE]
                left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE]
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                left_mouth = results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_LEFT]
                right_mouth = results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT]
                left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
                right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
                left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                left_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
                right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]

                mid_hip = (left_hip.y + right_hip.y) / 2
                lower_bound = int((mid_hip + right_shoulder.y) / 2 * image.shape[0])

                higher_bound = int(right_eye.y * image.shape[0])
                nose_x = int(nose.x * image.shape[1])
                rhip_nose_distance = abs(right_hip.x - nose.x) * image.shape[1]
                lhip_nose_distance = abs(left_hip.x - nose.x) * image.shape[1]
                hip_nose_distance = round((rhip_nose_distance + lhip_nose_distance) / 2)
                left_bound = max(0, int(nose_x - 3 * hip_nose_distance))
                right_bound = min(image.shape[1], int(nose_x + 3 * hip_nose_distance))

                eye_y = int(right_eye.y * screen_height)
                mouth_y = int(right_mouth.y * screen_height)
                right_eye_x = int(right_eye.x * screen_width)
                left_eye_x = int(left_eye.x * screen_width)

                left_wrist_pos = (int(left_wrist.x * frame_width), int(left_wrist.y * frame_height))
                right_wrist_pos = (int(right_wrist.x * frame_width), int(right_wrist.y * frame_height))
                left_elbow_pos = (int(left_elbow.x * frame_width), int(left_elbow.y * frame_height))
                right_elbow_pos = (int(right_elbow.x * frame_width), int(right_elbow.y * frame_height))

                current_wrist_path.append({'left': left_wrist_pos, 'right': right_wrist_pos})
                current_elbow_path.append({'left': left_elbow_pos, 'right': right_elbow_pos})

                while len(circles) < max_circles:
                    circles.append(generate_circle(left_bound, right_bound, higher_bound, lower_bound))
                    actions.append(circles[0][4])
                    startTime = time.time()

                for circle in circles:
                    cv2.circle(frame, (circle[0], circle[1]), circle[2], circle[3], -1)

                protecting_head = False
                protecting_body = False

                current_time = time.time()
                if obstacle is None and current_time - last_obstacle_time > random.uniform(min_interval, max_interval):
                    if random.randint(0, 1) == 0:
                        obstacle = create_obstacle_left(eye_y)
                        obstacle_direction = "left"
                    else:
                        obstacle = create_obstacle_right(eye_y)
                        obstacle_direction = "right"
                    last_obstacle_time = current_time

                if obstacle:
                    if (obstacle_direction == "left"):
                        move_obstacle_left(obstacle)
                    else:
                        move_obstacle_right(obstacle)
                    draw_obstacle(frame, obstacle)

                    if check_collision(obstacle, eye_y, mouth_y, left_eye_x, right_eye_x):
                        print("Obstacle hit!")
                        hits = hits + 1
                        if obstacle['side'] == 'left':
                            left_hit = left_hit + 1
                        else:
                            right_hit = right_hit + 1
                        obstacle = None
                        hit = True
                    elif is_off_screen(obstacle):
                        print("Obstacle dodged!")
                        obstacle = None
                        hit = False
                    else:
                        hit = False
                else:
                    hit = False

                if (np.sqrt((left_wrist_pos[0] - circles[0][0]) ** 2 + (left_wrist_pos[1] - circles[0][1]) ** 2) <
                        circles[0][2] or np.sqrt(
                                (right_wrist_pos[0] - circles[0][0]) ** 2 + (right_wrist_pos[1] - circles[0][1]) ** 2) <
                        circles[0][2]):
                    if np.sqrt((right_wrist_pos[0] - circles[0][0]) ** 2 + (right_wrist_pos[1] - circles[0][1]) ** 2) < \
                            circles[0][2]:
                        handUsed = "right"
                    else:
                        handUsed = "left"
                    last_4_wrist_frames = current_wrist_path[-4:] if len(current_wrist_path) > 4 else current_wrist_path
                    last_4_elbow_frames = current_elbow_path[-4:] if len(current_elbow_path) > 4 else current_elbow_path

                    handsUsed.append(handUsed)
                    punchType.append(circles[0][4])
                    wrist_paths.append(current_wrist_path[:])  # Save a copy of the path
                    elbow_paths.append(current_elbow_path[:])  # Save a copy of the path
                    current_wrist_path.clear()
                    current_elbow_path.clear()

                left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
                right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

                wrists = [
                    (int(left_wrist.x * frame_width), int(left_wrist.y * frame_height)),
                    (int(right_wrist.x * frame_width), int(right_wrist.y * frame_height))
                ]
                cv2.circle(frame, wrists[0], 20, (255, 255, 0), -1)
                cv2.circle(frame, wrists[1], 20, (255, 255, 0), -1)

                protecting_head = is_protecting_head(left_mouth, left_wrist) or is_protecting_head(right_mouth,
                                                                                                   right_wrist)
                protecting_body = is_protecting_body(left_shoulder, left_elbow) or is_protecting_body(right_shoulder,
                                                                                                      right_elbow)

                if protecting_body:
                    body_def_time = body_def_time + 1
                if protecting_head:
                    head_def_time = head_def_time + 1

                draw_box(frame, "HEAD", (10, 80), protecting_head)
                draw_box(frame, "BODY", (10, 130), protecting_body)
                draw_box(frame, f"Hit Counter:{hits}", (10, 180), hit)
                circles_to_remove = []
                for i, circle in enumerate(circles):
                    for wrist in wrists:
                        distance = np.sqrt((wrist[0] - circle[0]) ** 2 + (wrist[1] - circle[1]) ** 2)
                        if distance < circle[2]:
                            circles_to_remove.append(i)
                            break

                for i in sorted(circles_to_remove, reverse=True):
                    if time.time() - startTime > 0.1:
                        reaction_times.append((time.time() - startTime, circles[0][4]))
                        stances.append(
                            (circles[0][4], [(landmark.x, landmark.y) for landmark in results.pose_landmarks.landmark]))
                    send_image = frame
                    circles.pop(i)

                if (len(circles) > 0):
                    draw_box(frame, circles[0][4], (10, 30), False)

            # Encode the processed frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
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

def dashboard(request):
    accuracy = [(jab_hits, jab_attempts), (uppercut_hits, uppercut_attempts), (hook_hits, hook_attempts)]
    defense = [head_def_time, body_def_time, total_time]
    dodges = [(left_hit, total_left_obstacles), (right_hit, total_right_obstacles)]
    _, buffer = cv2.imencode('.jpg', send_image)
    punchimage = buffer.tobytes()
    data = {
        "Reaction Time": pd.Series(reaction_times),
        "Accuracy": pd.Series(accuracy),
        "Defense": pd.Series(defense),
        "Dodges": pd.Series(dodges),
        "Stance": pd.Series(stances)
    }
    df = pd.DataFrame(data)
    df.to_csv("home/csvfiles/output.csv", index=False)
    userInput = "Analysis!"

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=sys_instruct
        ),
        contents=[userInput])

    context = {
        "text": response.text,
    }

    return render(request,'home/dashboard.html', context)