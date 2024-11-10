from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import torch
import os
import time

# 学習済みモデルのロード
def load_finetuned_model(path="./trained_stream_end/fine_tuned_sd_lora_best"):
    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(path, torch_dtype=torch.float16)
    pipeline.to("cuda")
    return pipeline

# 画像生成関数
def generate_image(input_image, pipeline, prompt="next frame of the baseball pitcher throwing a fastball. almost same pose, but different ball position"):
    input_image = input_image.convert("RGB").resize((512, 512))

    # 自動キャストで浮動小数点精度を管理しつつ生成
    with torch.autocast("cuda"):
        image = pipeline(prompt=prompt, image=input_image, strength=0.3, guidance_scale=20.0).images[0]
    
    return image

if __name__ == "__main__":
    input_image_path = "data_test/example3_3.png"  # 初期入力画像
    output_dir = "data_test/sd_stream_3000data/example3_3"   # 出力画像を保存するディレクトリ
    prompt = "next frame of the baseball pitcher throwing a fastball. almost same pose, but different ball position"
    
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    # モデルのロード
    pipeline = load_finetuned_model()
    
    # 初期画像のロード
    init_image = Image.open(input_image_path).resize((512, 512))
    
    # 画像生成ループ
    # 時間計測開始
    start_time = time.time()
    
    for i in range(23):
        print(f"Generating frame {i}")
        
        # 画像生成
        generated_image = generate_image(init_image, pipeline, prompt)
        
        # 次のフレーム生成のために生成画像を初期画像として設定
        init_image = generated_image

        # 出力画像を保存
        output_image_path = os.path.join(output_dir, f"pitcher_{i}.png")
        
        # リサイズして保存 (320x240)
        resized_image = generated_image.resize((320, 240))
        resized_image.save(output_image_path)
        print(f"Saved generated image to {output_image_path}")

    # 時間計測終了
    end_time = time.time()
    total_time = end_time - start_time
    print(f"あああああTotal time for generating 30 frames: {total_time:.2f} seconds")