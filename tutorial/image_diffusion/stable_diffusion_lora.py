# stable diffusionを使ってLoRAを学習するサンプルコード（失敗）
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor

# モデルの準備
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

# LoRAの設定
lora_attn_procs = {}
for name in pipe.attn_processors.keys():
    cross_attention_dim = pipe.config.cross_attention_dim if name.startswith("mid_block") else pipe.unet.config.cross_attention_dim
    lora_attn_procs[name] = LoRAAttnProcessor(
        hidden_size=pipe.unet.config.hidden_size,
        cross_attention_dim=cross_attention_dim,
        rank=4,
    )
pipe.unet.set_attn_processor(lora_attn_procs)

# 学習可能なパラメータの設定
def trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]

trainable_params = trainable_parameters(pipe.unet.attn_processors)
optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)

# 学習ループ
num_epochs = 100
for epoch in range(num_epochs):
    # ここで学習データを準備します
    prompt = "your training prompt"
    negative_prompt = "your negative prompt"
    
    # 順伝播
    with torch.no_grad():
        latents = torch.randn((1, 4, 64, 64)).to("cuda")
    
    # ノイズ除去ステップ
    for t in pipe.scheduler.timesteps:
        with torch.autocast("cuda"):
            noise_pred = pipe.unet(latents, t, encoder_hidden_states=pipe.text_encoder(prompt)[0]).sample
        
        # ここで損失を計算し、最適化ステップを実行します
        # 例: MSE損失を使用する場合
        loss = torch.nn.functional.mse_loss(noise_pred, torch.randn_like(noise_pred))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# 学習済みLoRAの保存
pipe.save_attn_procs("path/to/save/lora")

# LoRAを使用した画像生成
image = pipe("a beautiful landscape", num_inference_steps=25).images[0]
image.save("generated_image.png")
