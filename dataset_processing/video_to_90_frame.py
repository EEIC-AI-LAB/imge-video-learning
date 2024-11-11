# 指定した動画について、3秒間のフレームを取得し、その中で差分が大きいフレーム30個を抽出する
from pathlib import Path
import os
import cv2


source_video_dir = Path.cwd() / "data/90frame"

for videoname in ["v_BaseballPitch_g06_c01",  "v_BaseballPitch_g08_c02",  "v_BaseballPitch_g13_c07"]:
    file_path = source_video_dir / videoname / f"{videoname}.avi"

    cap = cv2.VideoCapture(str(file_path))
    fps = cap.get(cv2.CAP_PROP_FPS)

    capture_seconds = 3
    num_frames = int(fps * capture_seconds)
    num_extract_frame = 30

    frame_count = 0
    previout_frame = None

    frame_list = []
    diff_list = []

    frame_idx = 0
    while cap.isOpened() and frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(source_video_dir, videoname, f"{videoname}_{frame_idx}.png")
        cv2.imwrite(frame_path, frame)
        frame_count += 1
        frame_idx += 1

    cap.release()
