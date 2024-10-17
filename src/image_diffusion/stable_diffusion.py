import os
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch

# ディレクトリの作成
os.makedirs("data", exist_ok=True)
os.makedirs("model", exist_ok=True)

# Stable Diffusion v2のモデル名
model_id = "stabilityai/stable-diffusion-2"

# ノイズスケジューラ
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
# 重みのダウンロード & モデルのロード
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16)

# モデルの保存
pipe.save_pretrained("model")

# GPU使用
pipe = pipe.to("cuda")

# 入力テキスト
prompt = "a robot struggling with reading through a scientific paper at Starbucks"

# 画像生成
image = pipe(prompt).images[0]

# 生成した画像をdataフォルダに保存
image_path = "data/generated_image.png"
image.save(image_path)

print(f"Generated image saved to {image_path}")
