import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import img_to_array
import os
import re

# 事前訓練済みモデルの読み込み
model = VGG16(weights='imagenet', include_top=False)

# 画像の読み込みと前処理
def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')  # RGBに変換
    img = img.resize((224, 224))  # VGG16の入力サイズにリサイズ
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)  # バッチ次元を追加
    img = preprocess_input(img)  # VGG16用の前処理
    return img

# 特徴抽出関数
def extract_features(image_path):
    img = load_and_preprocess_image(image_path)
    features = model.predict(img)
    return features.flatten()

# 数字2桁を抽出してソートする関数
def load_images_from_directory(directory_path):
    image_files = []
    for file_name in os.listdir(directory_path):
        if file_name.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', 'gif')):
            image_files.append(os.path.join(directory_path, file_name))

    # ファイル名から末尾の数字2桁を抽出し、それに基づいてソート
    image_files.sort(key=lambda x: int(re.search(r'(\d{1,2})(?=\D*$)', x).group(0)))
    return image_files

# ディレクトリのパス
raw_img_dir = '/path/to/raw_img_dir'
eval_img_dir = '/path/to/eval_img_dir'

# 画像の読み込み
raw_images= load_images_from_directory(raw_img_dir)
eval_images = load_images_from_directory(eval_img_dir)

# 結果を保存するファイル
output_file = "./output.txt"
cosine_similarities = []  # コサイン類似度を記録するリスト

# テキストファイルに書き込みモードでオープン
with open(output_file, "w") as f:
    # ディレクトリ内の画像を順番に1対1で比較
    min_length = min(len(raw_images), len(eval_images))

    for i in range(min_length):
        raw_img_path = raw_images[i]
        eval_img_path = eval_images[i]

        # 特徴抽出
        raw_features = extract_features(raw_img_path)
        eval_features = extract_features(eval_img_path)

        # コサイン類似度の計算
        cosine_sim = cosine_similarity([raw_features], [eval_features])
        cosine_similarities.append(cosine_sim[0][0])  # リストに追加

        # 類似度をテキストファイルに書き込み
        f.write(f'Comparing {os.path.basename(raw_img_path)} and {os.path.basename(eval_img_path)}\n')
        f.write(f'Cosine similarity: {cosine_sim[0][0]:.4f}\n\n')

# コサイン類似度をグラフにプロット
highlight_points = [0, 24, 25, 49, 50]  # 赤色にするインデックス

plt.figure(figsize=(10, 6))

# 全てのポイントを青色で線で繋ぐ
plt.plot(cosine_similarities, marker='o', linestyle='-', color='b', label='Cosine Similarity')

# 指定されたインデックスのポイントを赤色でプロット
for i in highlight_points:
    plt.plot(i, cosine_similarities[i], marker='o', color='r', linestyle='None', label='Highlighted Points')

plt.title('Cosine Similarity between --- and ---')
plt.xlabel('Image Pair Index')
plt.ylabel('Cosine Similarity')
plt.grid(True)
plt.show()

