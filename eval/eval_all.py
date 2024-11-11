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
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
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
    image_files.sort(key=lambda x: int(re.search(r'(\d{1,2})(?=\D*$)', x).group(0)))
    return image_files

# 複数のディレクトリから画像を読み込み、比較して1つのプロットにまとめる関数
def eval_images_in_directories_on_single_plot(raw_directory, eval_directories):
    raw_images = load_images_from_directory(raw_directory)
    plt.figure(figsize=(12, 8))

    # 各ディレクトリに対するコサイン類似度を計算してプロット
    for eval_dir in eval_directories:
        eval_images = load_images_from_directory(eval_dir)
        cosine_similarities = []
        min_length = min(len(raw_images), len(eval_images))

        for i in range(min_length):
            raw_img_path = raw_images[i]
            eval_img_path = eval_images[i]

            features_raw = extract_features(raw_img_path)
            features_eval = extract_features(eval_img_path)

            cosine_sim = cosine_similarity([features_raw], [features_eval])
            cosine_similarities.append(cosine_sim[0][0])

        # 各ディレクトリのコサイン類似度をプロット
        plt.plot(cosine_similarities, marker='o', linestyle='-', label=os.path.basename(os.path.dirname(eval_dir)))

    # プロット
    plt.title('Cosine Similarity between raw and generated images')
    plt.xlabel('Image Pair Index')
    plt.ylabel('Cosine Similarity')
    plt.grid(True)
    plt.legend()
    plt.show()

# 実行例
raw_directory = 'path/to/raw_dir'
eval_directories = [
    'path/to/eval_dir1',
    'path/to/eval_dir2',
    'path/to/eval_dir3',
]

eval_images_in_directories_on_single_plot(raw_directory, eval_directories)
