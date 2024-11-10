from PIL import Image
import os

def create_gif_from_multiple_directories(directories, output_file, duration=100):
    images = []
    
    # 各ディレクトリを順に処理
    for directory in directories:
        # 0から22までのファイル名を順に読み込む
        for i in range(23):  # 0から22までなので23回
            file_path = os.path.join(directory, f"pitcher_{i}.png")  # pitcher_0.png のようにゼロパディングなしで読み込む
            if os.path.exists(file_path):  # ファイルが存在するか確認
                img = Image.open(file_path)
                images.append(img)
            else:
                print(f"Warning: {file_path} not found. Skipping.")

    # GIFを保存
    if images:
        images[0].save(
            output_file,
            save_all=True,
            append_images=images[1:],  # 最初の画像以外を追加
            duration=duration,  # 各フレームの表示時間 (ミリ秒)
            loop=0  # ループの回数（0は無限ループ）
        )
        print(f"GIF saved as {output_file}")
    else:
        print("No images found to create GIF.")

# 使用例
if __name__ == "__main__":
    # 複数のディレクトリをリストで指定
    directories = [
        "data_test/normal_stable_diffusion/example1_1_34.52sec",
        "data_test/normal_stable_diffusion/example1_2_29.67sec",
        "data_test/normal_stable_diffusion/example1_3_28.06sec"
    ]
    output_file = "data_gif/combined_output.gif"  # 出力するGIFファイルの名前
    create_gif_from_multiple_directories(directories, output_file, duration=100)  # durationは500ミリ秒（0.5秒）です
