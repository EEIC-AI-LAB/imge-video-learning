# UCF50データセットのbaseball_pitchの動画をtrain, evaluation, testに分割する
from pathlib import Path
import random
import re
import shutil

source_dir = Path("/mnt/c/Users/Pocha/Downloads/UCF50")
new_folder = Path.cwd() / "data/videos_baseball_pitch"
action_type = "BaseballPitch"

train_folder = new_folder / "train"
evaluation_folder = new_folder / "evaluation"
test_folder = new_folder / "test"
train_folder.mkdir(parents=True, exist_ok=True)
evaluation_folder.mkdir(parents=True, exist_ok=True)
test_folder.mkdir(parents=True, exist_ok=True)

pattern = f"v_{action_type}+_g[0-9]+_c[0-9]+\.avi"
action_type_dir = source_dir / action_type
all_files = [f_path for f_path in action_type_dir.rglob("*.avi") if re.match(pattern, f_path.name)]
random.shuffle(all_files)

train_split = int(0.8 * len(all_files))
eval_split = int(0.1 * len(all_files))

train_files = all_files[:train_split]
eval_files = all_files[train_split:train_split + eval_split]
test_files = all_files[train_split + eval_split:]

for f_path in train_files:
    shutil.copy2(f_path, train_folder)
    print(f"Copied {f_path.name} to {train_folder}")

for f_path in eval_files:
    shutil.copy2(f_path, evaluation_folder)
    print(f"Copied {f_path.name} to {evaluation_folder}")

for f_path in test_files:
    shutil.copy2(f_path, test_folder)
    print(f"Copied {f_path.name} to {test_folder}")

print(f"All files copied to {new_folder}")
