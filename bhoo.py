import cv2
import mediapipe as mp
import numpy as np

# List of T-shirt images (or dresses)
tshirt_paths = [
    'C:\Users\bhoom\OneDrive\Desktop\python final\.png',
    'C:\\Users\\vinut\\OneDrive\\virtual_tryon\\fancy.png',
    'C:\\Users\\vinut\\OneDrive\\virtual_tryon\\pink1.png'
]

current_index = 0
tshirt_img = cv2.imread(tshirt_paths[current_index], cv2.IMREAD_UNCHANGED)

# Check if T-shirt image has 4 channels (RGBA) or 3 channels (RGB)
if tshirt_img.shape[2] == 4:
    has_alpha = True
else:
    has_alpha = False

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Webcam
cap = cv2.VideoCapture(0)

# Variables for smooth interpolation
prev_x, prev_y = 0, 0

def rotate_image(img, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    return rotated

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Handle keypress for T-shirt selection
    key = cv2.waitKey(1) & 0xFF
    if key == ord('1'):  # Change to T-shirt 1
        current_index = 0
        tshirt_img = cv2.imread(tshirt_paths[current_index], cv2.IMREAD_UNCHANGED)
    elif key == ord('2'):  # Change to T-shirt 2
        current_index = 1
        tshirt_img = cv2.imread(tshirt_paths[current_index], cv2.IMREAD_UNCHANGED)
    elif key == ord('3'):  # Change to T-shirt 3
        current_index = 2
        tshirt_img = cv2.imread(tshirt_paths[current_index], cv2.IMREAD_UNCHANGED)
    elif key == ord('q'):  # Quit the program
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Get coordinates of left and right shoulders, waist, and hips
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_waist = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_waist = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

        # Mapping to image coordinates
        x1 = int(left_shoulder.x * w)
        y1 = int(left_shoulder.y * h)
        x2 = int(right_shoulder.x * w)
        y2 = int(right_shoulder.y * h)
        x3 = int(left_waist.x * w)
        y3 = int(left_waist.y * h)
        x4 = int(right_waist.x * w)
        y4 = int(right_waist.y * h)

        # Calculate dynamic scaling for T-shirt width and height
        tshirt_width = int(1.2 * abs(x2 - x1))  # Slightly wider width based on shoulder width
        tshirt_height = int(1.5 * abs(y3 - y1))  # Adjusting height based on shoulder to waist distance

        # Calculate torso angle based on shoulders for better alignment
        shoulder_angle = np.arctan2(y1 - y2, x1 - x2) * 180 / np.pi

        # Calculate the center position of the T-shirt
        x_center = int((x1 + x2) / 2)
        y_center = int((y1 + y3) / 2)

        # Smooth interpolation for T-shirt center movement
        x_center = int(0.7 * prev_x + 0.3 * x_center)
        y_center = int(0.7 * prev_y + 0.3 * y_center)

        prev_x, prev_y = x_center, y_center

        x_start = int(x_center - tshirt_width / 2)
        y_start = int(y_center - tshirt_height / 2)

        # Resize and rotate the T-shirt to match the shoulder angle
        resized_tshirt = cv2.resize(tshirt_img, (tshirt_width, tshirt_height))
        resized_tshirt = rotate_image(resized_tshirt, shoulder_angle)

        # Overlay the T-shirt onto the frame using alpha blending
        for i in range(tshirt_height):
            for j in range(tshirt_width):
                if 0 <= x_start + j < w and 0 <= y_start + i < h:
                    if has_alpha:
                        alpha = resized_tshirt[i, j, 3] / 255.0
                    else:
                        alpha = 1.0  # No alpha channel, fully opaque

                    for c in range(3):
                        frame[y_start + i, x_start + j, c] = int(
                            alpha * resized_tshirt[i, j, c] + (1 - alpha) * frame[y_start + i, x_start + j, c]
                        )

    cv2.imshow('Virtual Clothes Try-On', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()