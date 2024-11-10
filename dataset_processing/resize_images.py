# 画像のリサイズを行う関数(VAEの学習に使用)
import os
from PIL import Image

def resize_images(input_folder, output_folder, target_size=(64, 64)):
    for root, dirs, files in os.walk(input_folder):
        # 入力フォルダからの相対パスを取得
        relative_path = os.path.relpath(root, input_folder)
        # 出力フォルダに対応するパスを作成
        output_path = os.path.join(output_folder, relative_path)

        # 出力フォルダが存在しない場合は作成
        os.makedirs(output_path, exist_ok=True)

        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                input_file = os.path.join(root, file)
                output_file = os.path.join(output_path, file)

                try:
                    with Image.open(input_file) as img:
                        # アスペクト比を維持しながらリサイズ
                        img.thumbnail(target_size)
                        
                        # 背景画像を作成
                        background = Image.new('RGB', target_size, (255, 255, 255))
                        
                        # リサイズした画像を中央に配置
                        offset = ((target_size[0] - img.width) // 2,
                                  (target_size[1] - img.height) // 2)
                        background.paste(img, offset)
                        
                        # 保存
                        background.save(output_file)
                except Exception as e:
                    print(f"Error processing {input_file}: {e}")

    print("画像のリサイズが完了しました。")

# 使用例
input_folder = "data/dataset"
output_folder = "data/dataset64"
resize_images(input_folder, output_folder)
