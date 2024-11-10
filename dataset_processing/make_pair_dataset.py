# 全動画について、今のフレームをinput、次のフレームをtargetとして保存する
from pathlib import Path
import shutil


source_frames_dir = Path.cwd() / "data/dataset_baseball_pitch"
save_frames_dir = Path.cwd() / "data/dataset"
save_frames_dir.mkdir(exist_ok=True)

for dataset_type in ["train", "evaluation", "test"]:
    output_dir = save_frames_dir / dataset_type
    (output_dir / "input").mkdir(parents=True, exist_ok=True)
    (output_dir / "target").mkdir(parents=True, exist_ok=True)
    for videoname_dir in (source_frames_dir / dataset_type).iterdir():
        videoname_list = sorted(list(videoname_dir.rglob("*.png")))
        shutil.copy(videoname_list[0], output_dir / "input" / f"{videoname_list[0].stem}_input.png")
        for i, file_path in enumerate(videoname_list[1:-1]):
            input_frame_path = output_dir / "input" / f"{file_path.stem}_{i:03d}_input.png"
            target_frame_path = output_dir / "target" / f"{file_path.stem}_{i:03d}_target.png"
            shutil.copy(file_path, input_frame_path)
            shutil.copy(file_path, target_frame_path)
        shutil.copy(videoname_list[-1], output_dir / "target" / f"{videoname_list[-1].stem}_target.png")

        print(f"Saved {videoname_dir} to {output_dir}")
