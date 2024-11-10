# フォルダ内の画像をリサイズして、指定サイズに合わせて背景を白で埋める
import os
from PIL import Image

def resize_images(input_folder, output_folder, target_size=(178, 218)):
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
                        # アスペクト比を維持したリサイズ（幅が基準、もしくは高さが基準になるようにリサイズ）
                        img_ratio = img.width / img.height
                        target_ratio = target_size[0] / target_size[1]

                        if img_ratio > target_ratio:
                            # 幅を基準にリサイズ（高さを調整）
                            new_width = target_size[0]
                            new_height = int(target_size[0] / img_ratio)
                        else:
                            # 高さを基準にリサイズ（幅を調整）
                            new_height = target_size[1]
                            new_width = int(target_size[1] * img_ratio)

                        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                        # 背景画像（CelebAのサイズに合わせた画像）を作成
                        background = Image.new('RGB', target_size, (255, 255, 255))

                        # リサイズした画像を背景の中央に配置
                        offset = ((target_size[0] - new_width) // 2, 
                                  (target_size[1] - new_height) // 2)
                        background.paste(img, offset)

                        # 保存
                        background.save(output_file)
                except Exception as e:
                    print(f"Error processing {input_file}: {e}")

    print("画像のリサイズが完了しました。")

# 使用例
input_folder = "data/dataset"
output_folder = "data/dataset_resized"
resize_images(input_folder, output_folder)
