import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import random

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

ret, frame = cap.read()
frame_height, frame_width = frame.shape[:2]

wrist_paths = []
elbow_paths = []
current_wrist_path = []
current_elbow_path = []
plots = []
handsUsed = []
punchType = []

# Circle parameters
def generate_circle():
    """Generate a new circle at a random position."""
    temp = random.randint(0, 2)
    action = "hook"
    if temp == 0:
        color = (255, 0, 0)
    elif temp == 1:
        color = (0, 255, 0)
        action = "jab"
    else:
        color = (0, 0, 255)
        action = "uppercut"
    x = random.randint(140, frame_width - 140)
    y = random.randint(140, frame_height - 140)
    radius = 30
    return (x, y, radius, action, color)

current_circle = generate_circle()
handUsed = "left"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and process the frame
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]

        left_wrist_pos = (int(left_wrist.x * frame_width), int(left_wrist.y * frame_height))
        right_wrist_pos = (int(right_wrist.x * frame_width), int(right_wrist.y * frame_height))
        left_elbow_pos = (int(left_elbow.x * frame_width), int(left_elbow.y * frame_height))
        right_elbow_pos = (int(right_elbow.x * frame_width), int(right_elbow.y * frame_height))

        current_wrist_path.append({'left': left_wrist_pos, 'right': right_wrist_pos})
        current_elbow_path.append({'left': left_elbow_pos, 'right': right_elbow_pos})

        cv2.circle(frame, (current_circle[0], current_circle[1]), current_circle[2], current_circle[4], -1)

        if (np.sqrt((left_wrist_pos[0] - current_circle[0]) ** 2 + (left_wrist_pos[1] - current_circle[1]) ** 2) < current_circle[2] or
                np.sqrt((right_wrist_pos[0] - current_circle[0]) ** 2 + (right_wrist_pos[1] - current_circle[1]) ** 2) < current_circle[2]):

            if np.sqrt((right_wrist_pos[0] - current_circle[0]) ** 2 + (right_wrist_pos[1] - current_circle[1]) ** 2) < current_circle[2]:
                handUsed = "right"
            else:
                handUsed = "left"

            handsUsed.append(handUsed)
            punchType.append(current_circle[3])
            wrist_paths.append(current_wrist_path[:])  # Save a copy of the path
            elbow_paths.append(current_elbow_path[:])  # Save a copy of the path
            current_wrist_path.clear()
            current_elbow_path.clear()
            current_circle = generate_circle()

    cv2.imshow("Pose Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()

for plot in plots:
    plt.figure(plot.number)
    plt.show()

for i in range(len(wrist_paths)):
    if len(wrist_paths[i]) >= 4:
        last_4_frames_wrists = wrist_paths[i][-4:]
        last_4_frames_elbows = elbow_paths[i][-4:]
        print(punchType[i])

        x_vals_wrists = np.array([pos[handsUsed[i]][0] for pos in last_4_frames_wrists])
        y_vals_wrists = np.array([pos[handsUsed[i]][1] for pos in last_4_frames_wrists])

        x_vals_elbows = np.array([pos[handsUsed[i]][0] for pos in last_4_frames_elbows])
        y_vals_elbows = np.array([pos[handsUsed[i]][1] for pos in last_4_frames_elbows])

        data_to_write = f"{punchType[i]}|{x_vals_wrists}|{y_vals_wrists}|{x_vals_elbows}|{y_vals_elbows}\n"
        with open("out.txt", 'a') as f:
            f.writelines(data_to_write)

        n_wrist = len(x_vals_wrists)
        n_elbow = len(x_vals_elbows)

#RED UPPERCUT
#GREEN JAB
#BLUE HOOK
