import torch
import wandb
from transformers import AdamW, get_scheduler, AutoTokenizer
from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
import torch.nn as nn
from dataset import get_dataloader
import loralib as lora  # LoRAのインポート

# LoRAを適用する関数
def apply_lora(model, rank=4):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            in_features, out_features = module.in_features, module.out_features
            lora_module = lora.Linear(in_features, out_features)
            lora_module.weight.data = module.weight.data.to("cuda")  # GPUに移動
              # biasがNoneでない場合のみGPUに移動
            if module.bias is not None:
                lora_module.bias = module.bias.to("cuda")               # GPUに移動
            lora_module.to("cuda", dtype=torch.float16)
            parent = model
            for part in name.split('.')[:-1]:
                parent = getattr(parent, part)
            setattr(parent, name.split('.')[-1], lora_module)

# トレーニング関数
def train_lora(
    pipeline,
    dataloader,
    val_dataloader,
    num_epochs,
    batch_size):
    
    learning_rate = 1e-5
    wandb.init(
        project="my-awesome-project",
        config={
            "learning_rate": learning_rate,
            "architecture": "CNN",
            "dataset": "Custom Dataset",
            "epochs": num_epochs,
        }
    )

    # LoRAの適用
    print("Applying LoRA...")
    apply_lora(pipeline.unet, rank=4)

    # optimizerの設定：UNetとVAEのデコーダ両方を対象
    optimizer = AdamW(
        list(pipeline.unet.parameters()) + list(pipeline.vae.decoder.parameters()),
        lr=learning_rate
    )

    # スケジューラの設定
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_epochs * len(dataloader)
    )

    # 損失関数の設定
    criterion = nn.MSELoss()
    
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    # トレーニングループ
    batch_num = len(dataloader.dataset) / batch_size
    best_loss = float('inf')
    best_val_loss = float('inf')
    best_epoch = -1
    for epoch in range(num_epochs):
        pipeline.unet.train()
        pipeline.vae.decoder.train()
        train_loss, val_loss = 0, 0
        for batch in dataloader:
            # 入力画像とターゲット画像をGPUに移動
            inputs = batch["input"].to("cuda", dtype=torch.float16)
            targets = batch["target"].to("cuda", dtype=torch.float16)
            
            # エンコードとテキストのエンコード
            text_inputs = tokenizer(
                ["make the next pitching frame"] * inputs.size(0),
                return_tensors="pt", padding=True, truncation=True
            ).input_ids.to("cuda")
            text_embeddings = pipeline.text_encoder(text_inputs).last_hidden_state.to("cuda")

            # 入力画像をVAEでエンコードして潜在表現を取得
            latents = pipeline.vae.encode(inputs).latent_dist.sample().to("cuda")
            latents = latents * 0.18215
            noise = torch.randn_like(latents).to("cuda")
            timesteps = torch.randint(0, 1000, (latents.size(0),), device="cuda").long()

            # ノイズ付加とUNetでノイズ予測、各サンプルで個別に処理
            denoised_latents_list = []
            for i in range(latents.size(0)):
                noisy_latents = pipeline.scheduler.add_noise(latents[i:i+1], noise[i:i+1], timesteps[i:i+1])
                pred_noise = pipeline.unet(noisy_latents, timesteps[i:i+1], encoder_hidden_states=text_embeddings[i:i+1], return_dict=False)[0]
                
                # ノイズ予測をもとに潜在表現を更新し、prev_timestepsをスカラーとして処理
                prev_timestep = max(timesteps[i].item() - 1, 0)  # スカラーとして前のステップを計算
                denoised_latents = pipeline.scheduler.step(pred_noise, prev_timestep, noisy_latents).prev_sample
                denoised_latents_list.append(denoised_latents)

            # デコードと損失計算
            denoised_latents = torch.cat(denoised_latents_list, dim=0)
            denoised_latents = denoised_latents / 0.18215  # スケールを戻す
            decoded_images = pipeline.vae.decode(denoised_latents).sample.to("cuda")
            loss = criterion(decoded_images, targets)
            train_loss += loss.item()

            # 勾配計算と更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
        
        train_loss /= len(dataloader.dataset) / batch_size
        # 検証ループ
        pipeline.unet.eval()
        pipeline.vae.decoder.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                inputs = batch["input"].to("cuda", dtype=torch.float16)
                targets = batch["target"].to("cuda", dtype=torch.float16)
                
                text_inputs = tokenizer(
                    ["make the next pitching frame"] * inputs.size(0),
                    return_tensors="pt", padding=True, truncation=True
                ).input_ids.to("cuda")
                text_embeddings = pipeline.text_encoder(text_inputs).last_hidden_state.to("cuda")

                latents = pipeline.vae.encode(inputs).latent_dist.sample().to("cuda")
                latents = latents * 0.18215
                noise = torch.randn_like(latents).to("cuda")
                timesteps = torch.randint(0, 1000, (latents.size(0),), device="cuda").long()

                denoised_latents_list = []
                for i in range(latents.size(0)):
                    noisy_latents = pipeline.scheduler.add_noise(latents[i:i+1], noise[i:i+1], timesteps[i:i+1])
                    pred_noise = pipeline.unet(noisy_latents, timesteps[i:i+1], encoder_hidden_states=text_embeddings[i:i+1], return_dict=False)[0]
                    prev_timestep = max(timesteps[i].item() - 1, 0)
                    denoised_latents = pipeline.scheduler.step(pred_noise, prev_timestep, noisy_latents).prev_sample
                    denoised_latents_list.append(denoised_latents)

                denoised_latents = torch.cat(denoised_latents_list, dim=0)
                denoised_latents = denoised_latents / 0.18215
                decoded_images = pipeline.vae.decode(denoised_latents).sample.to("cuda")
                val_loss += criterion(decoded_images, targets).item()

        val_loss /= len(val_dataloader.dataset) / batch_size
        wandb.log({"train_loss": train_loss, "val_loss": val_loss})

        # ベストモデルの保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            pipeline.save_pretrained(f"./trained_stream_end/fine_tuned_sd_lora_best")
            print(f"Best model saved at epoch {epoch} with val_loss: {val_loss:.6f}")
            
        pipeline.save_pretrained(f"./trained_stream_end/fine_tuned_stable_diffusion_lora_{epoch}")
        print(f"Epoch {epoch+1}/{num_epochs}, Train_Loss: {train_loss}, Val_Loss: {val_loss}")

# トレーニング実行
if __name__ == "__main__":
    print("Loading finetuned model...")
    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
        "stabilityai/sd-turbo", torch_dtype=torch.float16).to("cuda")
    
    # スケジューラをDDIMに設定し、テンソルをGPUに移動
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler.set_timesteps(num_inference_steps=1000)
    device = torch.device("cuda")
     # スケジューラのテンソルをGPUに移動（存在確認を行う）
    if hasattr(pipeline.scheduler, "alphas_cumprod"):
        pipeline.scheduler.alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(device)
    if hasattr(pipeline.scheduler, "sigmas"):
        pipeline.scheduler.sigmas = pipeline.scheduler.sigmas.to(device)
    if hasattr(pipeline.scheduler, "betas"):
        pipeline.scheduler.betas = pipeline.scheduler.betas.to(device)

    print("Finetuned model loaded.")
    
    # データローダーの呼び出し
    input_dir = "./train/input"
    target_dir = "./train/target"
    
    input_val_dir = "./evaluation/input"
    target_val_dir = "./evaluation/target"
    
    batch_size = 8
    num_epochs = 20
    dataloader = get_dataloader(input_dir=input_dir, target_dir=target_dir, batch_size=batch_size)
    val_dataloader = get_dataloader(input_dir=input_val_dir, target_dir=target_val_dir, batch_size=batch_size)
    
    print("Starting training...")
    train_lora(pipeline, dataloader,val_dataloader, num_epochs=num_epochs, batch_size=batch_size)
    wandb.finish()
    print("Training completed.")
    
    pipeline.save_pretrained(f"./trained_stream_end/fine_tuned_stable_diffusion_lora_honban")
