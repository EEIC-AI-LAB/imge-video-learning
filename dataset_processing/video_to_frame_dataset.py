# train, eval, testの全動画について、3秒間のフレームを取得し、その中で差分が大きいフレーム30個を抽出する
from pathlib import Path
import os
import numpy as np
import cv2


source_video_dir = Path.cwd() / "data/videos_baseball_pitch"
save_frames_dir = Path.cwd() / "data/dataset_baseball_pitch"
save_frames_dir.mkdir(exist_ok=True)

for dataset_type in ["train", "evaluation", "test"]:
    for file_path in (source_video_dir / dataset_type).rglob("*.avi"):
        output_dir = save_frames_dir / dataset_type / file_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(file_path))
        fps = cap.get(cv2.CAP_PROP_FPS)

        capture_seconds = 3
        num_frames = int(fps * capture_seconds)
        num_extract_frame = 30

        frame_count = 0
        previout_frame = None

        frame_list = []
        diff_list = []

        # 前のフレームとの差分を計算し、合計値の上位{num_extract_frame}フレームを保存
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
                previout_frame = gray_frame
            else:
                diff_list.append(-1)
                previout_frame = gray_frame
            frame_count += 1

        # 差分の大きいフレームを保存
        large_diff_idx = np.array(diff_list).argsort()[-num_extract_frame:][::-1]
        for idx in large_diff_idx:
            frame = frame_list[idx]
            frame_path = os.path.join(output_dir, f"{file_path.stem}_{idx:03d}.png")
            cv2.imwrite(frame_path, frame)

        cap.release()
        print(f"Saved {frame_count} frames to {output_dir}")
    print(f"Saved {dataset_type} dataset to {save_frames_dir}")
