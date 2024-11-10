# 高速動画生成AI
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

