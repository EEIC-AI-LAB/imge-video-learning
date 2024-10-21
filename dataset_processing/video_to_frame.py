import os
import numpy as np
import cv2

input_video_path = "/mnt/c/Users/Pocha/Downloads/UCF50/BaseballPitch/v_BaseballPitch_g01_c01.avi"
output_dir = "data/frames"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(input_video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

capture_seconds = 3
num_frames = int(fps * capture_seconds)

frame_count = 0
previout_frame = None

frame_list = []
diff_list = []
while cap.isOpened() and frame_count < num_frames:
    ret, frame = cap.read()
    if not ret:
        break
    frame_list.append(frame)
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if previout_frame is not None:
        frame_diff = cv2.absdiff(previout_frame, gray_frame)
        diff_sum = np.sum(frame_diff)
        diff_list.append(diff_sum)
        
        print(f"Frame {frame_count}: {diff_sum}")
        previout_frame = gray_frame
    else:
        diff_list.append(-1)
        previout_frame = gray_frame
    frame_count += 1

num_extract_frame = 20
large_diff_idx = np.array(diff_list).argsort()[-num_extract_frame:][::-1]

for idx in large_diff_idx:
    frame = frame_list[idx]
    frame_path = os.path.join(output_dir, f"frame_{idx:04d}.png")
    cv2.imwrite(frame_path, frame)

cap.release()
print(f"Saved {frame_count} frames to {output_dir}")
