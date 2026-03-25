import cv2
import numpy as np
import joblib
import time

# 1. Linear Interpolation Function (Smooth Transition-er jonno)
def interpolate_landmarks(start_pos, end_pos, frames=10):
    """Duto landmark-er majhe missing frames generate kore"""
    interpolation_steps = []
    for i in range(frames):
        t = i / frames
        # Formula: (1-t)*Start + t*End
        intermediate_frame = (1 - t) * start_pos + t * end_pos
        interpolation_steps.append(intermediate_frame)
    return interpolation_steps

# 2. Render Function (OpenCV diye skeleton banano)
def draw_skeleton(image, landmarks):
    # Landmarks-ke image scale-e niye asha (Assuming 640x480 resolution)
    h, w, _ = image.shape
    for lm in landmarks:
        cx, cy = int(lm[0] * w), int(lm[1] * h)
        cv2.circle(image, (cx, cy), 3, (0, 255, 0), -1)
    
    # Ekhane MediaPipe-er connection logic add kora jay (cv2.line)
    return image

# 3. Main Controller Logic
def play_sign_sequence(gloss_list, landmark_dict):
    canvas = np.zeros((480, 640, 3), dtype="uint8")
    last_frame = None

    for word in gloss_list:
        if word in landmark_dict:
            current_sign_frames = landmark_dict[word] # NumPy array shape: (F, 21, 3)

            # Transition phase: Purono sign theke notun sign-er smooth shift
            if last_frame is not None:
                transition = interpolate_landmarks(last_frame, current_sign_frames[0])
                for t_frame in transition:
                    img = np.zeros_like(canvas)
                    draw_skeleton(img, t_frame)
                    cv2.imshow("Sign Avatar", img)
                    cv2.waitKey(20)

            # Actual Sign phase
            for frame in current_sign_frames:
                img = np.zeros_like(canvas)
                draw_skeleton(img, frame)
                cv2.imshow("Sign Avatar", img)
                last_frame = frame
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break

# Mock execution (Joblib diye load kora files)
# landmark_data = joblib.load('sign_database.pkl') 
# play_sign_sequence(['HELLO', 'NAME', 'WHAT'], landmark_data)