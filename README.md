# 高速動画生成AI

使用したデータ類：[このgoogleドライブ](https://drive.google.com/drive/folders/1h0aLfaVYRGtghsWe6N4Q6GHUyqaAzcKl)に置いています

## 概要
- 一つ前の時刻の画像データを元に一つ後の時刻の画像生成を繰り返し、画像をつなぎ合わせて動画を生成する。
- VAEをファインチューニングして画像生成を試みたディレクトリ(vae)、stablediffusionをloraでimg2imgでファインチューニングしたディレクトリ(sd-ft)、できた画像データと正解データとの誤差を評価するディレクトリ(eval)の3つを提出する。
## stablediffusionのloraファインチューニング(西岡)
- ファイル構成
```
sd-ft/
├── dataset.py          　
├── generate_conti.py
├── makegif.py
├── train_real.py
├── train_stream.py
```
- 各ファイルの説明
    - dataset.py\
    今回使用したデータをRGB形式で読み込むコード

    - generate_conti.py\
    ファインチューニングされたモデルを利用して、23枚の連続した画像を生成するコード

    - makegif.py\
    generate_conti.pyで作成された23枚の画像をgif画像に変換する

    - train_real.py\
    stablediffuisonをloraを用いてファインチューニングするコード。主にtrain_lora関数内でファインチューンしている。トレインするためのループ、テストデータを用いてそのモデルを検証するための二つのループがある

        ```
        optimizer = AdamW(
            list(pipeline.unet.parameters()) + list(pipeline.vae.decoder.parameters()),
            lr=learning_rate
        )
        ```
        簡単に説明すると、この部分でUNet,VAEdecorder部分をファインチューニングすると宣言し
        ```
        loss = criterion(decoded_images, targets)
        ```
        この箇所で、生成された画像と正解画像との誤差を計算することで正解により近づくようにファインチューニングしている。


    - train_stream.py\
    今回発表しなかったが、stablediffuisonよりも高速で画像を生成できるstreamdiffusonも同様にloraでファインチューニングした。train_realと同様の構造であるが、仕様が少し違い調節するのが難しかった。

## 生成された画像の評価(小崎)
- ファイル構成
```
eval/
├── eval.py
├── eval_all.py
├── plots/
│   ├── ...
```

- 各ファイルの説明
    - eval.py\
    生成された画像と元の画像の特徴量をコサイン類似度で比較し、プロットする。ハードコーディングしていた画像へのパスは適当な例に変更している。
    - eval_all.py\
    元の画像と複数の生成された画像を比較して、全てをまとめてプロットする。ハードコーディングしていた画像へのパスは適当な例に変更している。
    - plots/\
    今回の実験で生成された画像と元の画像の特徴量を比較した結果のプロットが保存されている。

## VAEによる次フレーム学習(井澤)
- ファイル構成
```
vae/
├── generate.py
├── model_vae.py
├── paired_loader.py
```

- 各ファイルの説明
    - generate.py\
    学習済みのVAEモデルを使用してtest用データセットの各画像から次のフレーム画像を生成する。
    - model_vae.py\
    学習に使用したVAEモデルを定義。
    - paired_loader.py\
    1枚のフレーム画像と次のフレーム画像をペアにして返す custom dataloader クラス。


## 学習に使用するデータセットの前処理(井澤)
- ファイル構成
```
dataset_processing/
├── extract_video.py
├── make_pair_dataset.py
├── resize_images.py
├── resize_images_square.py
├── split_videos.py
├── video_to_90_frame.py
├── video_to_frame_dataset.py
├── video_to_frame.py
```

- 各ファイルの説明
    - extract_vide.py\
    UCF50データセットの50種のアクションから各種3つの動画を抽出する
    - make_pair_dataset.py\
    全動画について、今のフレームをinput、次のフレームをtargetとして保存する
    - resize_images.py\
    画像のリサイズを行う関数(VAEの学習に使用)
    - resize_images_square.py\
    フォルダ内の画像をリサイズして、指定サイズに合わせて背景を白で埋める(正方形の画像になるようにする)
    - split_videos.py\
    UCF50データセットのbaseball_pitchの動画をtrain, evaluation, testに分割する
    - video_to_90_frame.py\
    指定した動画について、3秒間のフレームを取得し、その中で差分が大きいフレーム30個を抽出する
    - video_to_frame_dataset.py\
    train, eval, testの全動画について、3秒間のフレームを取得し、その中で差分が大きいフレーム30個を抽出する
    - video_to_frame.py\
    全動画について、3秒間のフレームを取得し、その中で差分が大きいフレーム20個を抽出する(video_to_frame_dataset.pyとほぼ同じ)
