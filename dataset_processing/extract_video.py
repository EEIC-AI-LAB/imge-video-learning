# 50種のアクションから各種3つの動画を抽出する
from pathlib import Path
import re
import shutil

source_dir = Path.cwd() / "UCF50"
new_folder = Path.cwd() / "data/videos"
new_folder.mkdir(exist_ok=True)

pattern = "v_[A-Za-z]+_g0[1-3]_c01\.avi"

for action_type_dir in source_dir.iterdir():
    print(action_type_dir)
    for file_path in action_type_dir.rglob("*.avi"):
        if re.match(pattern, file_path.name):
            shutil.copy2(file_path, new_folder)
            print(f"Copied {file_path.name}")
print(f"All files copied to {new_folder}")
