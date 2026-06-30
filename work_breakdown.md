# 全体WBS
## フェーズ1: 学習を始めるための最小セット

このフェーズのゴールは、「小さいデータでもよいので、1回 forward/backward して loss が下がる」状態です。精度はまだ気にせず、学習パイプラインの通し確認を優先します。

| # | チケット | 難易度 |
|---|---------|--------|
| 1-1 | 環境セットアップ（conda, CUDA, 依存パッケージのインストール） | ★☆☆ |
| 1-2 | データの置き場所と入出力形式を確定する（生データ / .npy / ラベル） | ★☆☆ |
| 1-3 | 全データからランダムに 2 件のtomoフォルダを抽出する仕組みを作る（seed 固定で再現可能にする） | ★☆☆ |
| 1-4 | `CustomDataset` の最小実装（単一サンプルを読み込んで tensor を返す） | ★★☆ |
| 1-5 | 最小モデルの実装（まずは小さな 3D CNN / 既存 encoder の簡易版） | ★★☆ |
| 1-6 | 最小 loss と最小 train loop の実装（単一GPU, 単一 fold, AMP なしでも可） | ★★☆ |
| 1-7 | 学習ログの出力（loss, 学習率, 1 epoch の確認） | ★☆☆ |

---

## フェーズ2: まずは精度より安定性を上げる

この段階で、前処理や正しいラベル変換を入れて、最小学習を「それっぽい学習」にします。ここまででモデル訓練は成立しています。

| # | チケット | 難易度 |
|---|---------|--------|
| 2-1 | `run.py` — 生データから `.npy` への前処理を実装する | ★★☆ |
| 2-2 | `CustomDataset` を拡張して座標→ラベルマップ変換を入れる | ★★☆ |
| 2-3 | `get_dataset` / `get_dataloader` を実装する（shuffle, batch, num_workers） | ★★☆ |
| 2-4 | ベース設定クラスを実装する（学習率, batch size, 入力サイズ） | ★★☆ |
| 2-5 | 評価指標を最小実装する（train/valid の比較ができる程度） | ★★☆ |

---

## フェーズ3: モデルを強くする

ここからモデルを段階的に本番構成へ寄せます。最初の学習ループは維持したまま、置き換えや追加だけで性能を上げます。

| # | チケット | 難易度 |
|---|---------|--------|
| 3-1 | `resnet3d.py` — ResNet3D エンコーダを実装する | ★★★ |
| 3-2 | `unet3d.py` — UNet デコーダとセグメンテーションヘッドを実装する | ★★★ |
| 3-3 | `unet3d.py` — `Net` クラスを組み立てる（deep supervision は後回しでも可） | ★★★ |
| 3-4 | `_base.py` — `BaseModel` に重みロードと損失接続を追加する | ★★☆ |
| 3-5 | `utils.py` — `get_model` を実装する | ★★☆ |

---

## フェーズ4: 学習品質を上げる

このフェーズでは、学習が安定してから効いてくる要素を足します。先に入れなくても学習は回るので、後追い追加に向いています。

| # | チケット | 難易度 |
|---|---------|--------|
| 4-1 | `base.py` — Dice Loss などのカスタム損失を実装する | ★★☆ |
| 4-2 | `utils.py` — Optimizer / Scheduler / grad norm を実装する | ★★☆ |
| 4-3 | `aug3d.py` — 3D 空間オーグメンテーションを実装する | ★★☆ |
| 4-4 | `mixup.py` / `cutmix.py` — 3D 版 Mixup / CutMix を実装する | ★★★ |
| 4-5 | `utils.py` — `ModelEMA` を実装する | ★★☆ |

---

## フェーズ5: 評価と推論を足す

ここで初めて、本番に近い評価を入れます。学習はすでにできているので、精度改善の反復がしやすくなります。

| # | チケット | 難易度 |
|---|---------|--------|
| 5-1 | `train.py` — `run_eval` を実装する（sliding window 推論） | ★★★ |
| 5-2 | `utils.py` / `torch.py` — NMS などの後処理を実装する | ★★☆ |
| 5-3 | `metric.py` — 評価指標とスコア計算を実装する | ★★☆ |
| 5-4 | `unet3d.py` — Flip TTA を追加する | ★★☆ |

---

## フェーズ6: 大規模化と運用

最後に、DDP やマルチGPU、WandB などの運用要素を足します。最小学習がすでに成立しているため、ここは拡張として扱えます。

| # | チケット | 難易度 |
|---|---------|--------|
| 6-1 | `train.py` — AMP, DDP, fold ループを追加する | ★★★ |
| 6-2 | `run.sh` — マルチGPU 学習スクリプトを整備する | ★☆☆ |
| 6-3 | `utils.py` — WandB ロガーを実装する | ★★☆ |
| 6-4 | `utils.py` — CFG / torch ユーティリティを整備する | ★★☆ |
| 6-5 | E2E 動作確認（小サブセットで学習と評価が回るか確認） | ★★☆ |

---

## 推奨進め方

```
フェーズ1 → フェーズ2 → フェーズ3 → フェーズ4 → フェーズ5 → フェーズ6
```

最初の目標は「小さいデータで 1 epoch 学習できる」ことです。そこに到達した後で、モデル・損失・拡張・評価・分散学習を1つずつ足すと、詰まりにくくなります。

# 3-1のWBS

`3-1 resnet3d.py — ResNet3D エンコーダを実装する` は、実装単位で分けるとかなり切りやすいです。  
依存関係を意識すると、以下くらいの粒度が実務向きです。

---

## チケット分割案

### 3-1-1 3D畳み込みユーティリティを作成する
**内容**
- `conv3x3x3` ヘルパーを実装
- `Conv3d` の kernel / stride / padding / bias を統一

**完了条件**
- `conv3x3x3(ic, oc, stride)` で `nn.Conv3d` を返せる
- `padding=1`, `bias=False` になっている

---

### 3-1-2 BasicBlock を実装する
**内容**
- 2層構成の残差ブロックを実装
- `Conv3d -> BN -> ReLU -> Conv3d -> BN`
- skip connection を加算
- `DropPath` を適用

**完了条件**
- `BasicBlock.forward(x)` が residual 加算込みで動作する
- `downsample=False` のケースで入力と整合する
- `drop_path_rate` を受け取れる

---

### 3-1-3 BasicBlock のダウンサンプル分岐を実装する
**内容**
- shape が変わる場合の residual branch を実装
- `1x1x1 Conv3d + norm` による downsample を追加

**完了条件**
- stride変更時に residual の shape を合わせられる
- `downsample=True` で forward が成立する

---

### 3-1-4 Bottleneck ブロックを実装する
**内容**
- `1x1x1 -> 3x3x3 -> 1x1x1` の bottleneck 構造を実装
- expansion を考慮した出力チャネル設計を追加
- skip connection / activation / DropPath を実装

**完了条件**
- `Bottleneck.forward(x)` が動作する
- expansion後のチャネル数と residual が整合する

---

### 3-1-5 Bottleneck のダウンサンプル分岐を実装する
**内容**
- Bottleneck 用 residual projection を実装
- expansion_factor に応じた stride / 出力チャネルを調整

**完了条件**
- `downsample=True` の bottleneck が shape mismatch なく動作する

---

### 3-1-6 ResnetEncoder3d の backbone 設定テーブルを実装する
**内容**
- `cfg.backbone` に応じて block種別と layer構成を切り替える
- 少なくとも `r3d18`, `r3d200` をサポート

**完了条件**
- backbone名から `layers` と `block` を引ける
- 未対応 backbone で適切に例外が出る

---

### 3-1-7 Stem 部分を実装する
**内容**
- 入力直後の stem を実装
- `7x7x7 Conv3d -> BN -> ReLU -> MaxPool3d`
- `in_stride`, `in_dilation`, `padding` を反映

**完了条件**
- stem 通過後に想定どおり空間サイズが縮小する
- 3Dテンソル入力で動作する

---

### 3-1-8 ResNet layer を組み立てる `_make_layer` を実装する
**内容**
- block を複数積む関数を実装
- 先頭 block のみ downsample / stride 変更を反映
- 後続 block は通常接続

**完了条件**
- `layer1`〜`layer4` を `nn.Sequential` として構築できる
- block数が設定どおりになる

---

### 3-1-9 DropPath rate の段階的割り当てを実装する
**内容**
- 全 block 数から linearly scaled な drop path rate を生成
- 各 layer / block に分配

**完了条件**
- 各 block に rate が割り当てられる
- 先頭から末尾に向けて増加する

---

### 3-1-10 事前学習重みロードを実装する
**内容**
- backbone 名に応じて重みファイルパスを解決
- `load_weights(self, wpath)` を呼ぶ
- `inference_mode` 時はスキップ

**完了条件**
- 学習時のみ重みロードされる
- 未ロード時でもモデル生成は可能

---

### 3-1-11 入力チャネル数の変換処理を実装する
**内容**
- `cfg.in_chans` に合わせて stem の `conv1` を更新
- pretrained weight を平均・複製して多チャネル/単チャネル対応

**完了条件**
- `in_chans != 3` でも forward 可能
- `conv1.in_channels` が設定値に一致する

---

### 3-1-12 gradient checkpoint 対応を追加する
**内容**
- `use_checkpoint` フラグを実装
- stem / 各 layer に対して `checkpoint(...)` を適用可能にする

**完了条件**
- `use_checkpoint=True/False` で切り替えられる
- forward がどちらでも動作する

---

### 3-1-13 `forward_features` を実装する
**内容**
- stem 出力と各 stage 出力をリストで返す
- multi-scale feature 抽出器として使える形にする

**完了条件**
- 中間特徴が5段階程度で取得できる
- 各特徴マップのチャネル数を確認できる

---

### 3-1-14 `forward` を実装する
**内容**
- 最終 stage の特徴のみ返す通常 forward を実装

**完了条件**
- 入力から最終特徴マップを返せる

---

### 3-1-15 encoder 出力チャネル情報の自動計算を実装する
**内容**
- ダミー入力で `forward_features` を実行
- 各出力のチャネル数を `self.channels` に保存

**完了条件**
- `self.channels` に各段の出力チャネルが入る

---

### 3-1-16 動作確認用の `__main__` テストコードを追加する
**内容**
- ダミー `cfg` を作成
- モデル生成、パラメータ数表示、ダミー入力の forward を確認

**完了条件**
- 単体実行で shape と parameter count を確認できる

---

## もう少し実務向けにまとめた版
もしチケット数を増やしすぎたくないなら、以下の **8分割** でも扱いやすいです。

1. **3D ResNet 基本ユーティリティ実装**
   - `conv3x3x3`, 共通部品

2. **BasicBlock 実装**
   - 通常経路 + downsample 含む

3. **Bottleneck 実装**
   - 通常経路 + downsample 含む

4. **ResnetEncoder3d stem / stage構築実装**
   - backbone config, stem, `_make_layer`

5. **DropPath / checkpoint 対応**
   - stochastic depth と省メモリ実行

6. **pretrained weight 読込と入力チャネル変換**
   - `load_weights`, `in_chans` 対応

7. **forward / forward_features / channels 実装**
   - multi-scale feature 抽出含む

8. **単体テスト・動作確認コード追加**
   - shape, backbone切替, in_chans差分確認

---

## おすすめの切り方
`work_breakdown.md` に書くなら、**実装順に依存がわかる粒度**が良いので、この並びがおすすめです。

- 3-1-1 ユーティリティ
- 3-1-2 BasicBlock
- 3-1-3 Bottleneck
- 3-1-4 backbone設定とstem
- 3-1-5 `_make_layer`
- 3-1-6 DropPath / checkpoint
- 3-1-7 pretrained weight / input channel対応
- 3-1-8 forward / forward_features / channels
- 3-1-9 動作確認・テスト

---

必要なら次に、`work_breakdown.md` にそのまま貼れるように  
**「番号付きのWBS形式」** で整形して出します。

---

# 3-2のWBS

`3-2 unet3d.py — UNet デコーダとセグメンテーションヘッドを実装する` は、
エンコーダ出力の受け取り仕様を先に固定し、デコーダブロック→全体組み立て→ヘッド→検証の順で分けると進めやすいです。

---

## チケット分割案

### 3-2-1 3D Conv-BN-Act 基本ブロックを作成する
**内容**
- `ConvBnAct3d` を実装
- `Conv3d -> BatchNorm3d -> ReLU` を共通化

**完了条件**
- `in/out_channels`, `kernel_size`, `padding`, `stride` を指定して生成できる
- 入力テンソルを通して期待 shape の出力を返せる

---

### 3-2-2 DecoderBlock3d の upsample 経路を実装する
**内容**
- 1段分のデコーダブロック `DecoderBlock3d` を実装
- upsample 後に畳み込み2層で特徴を整形

**完了条件**
- `in_channels`, `out_channels`, `scale_factor`, `upsample_mode` を受け取れる
- skip 入力なしでも forward が動作する

---

### 3-2-3 DecoderBlock3d の skip 接続を実装する
**内容**
- upsample 後に skip とチャネル方向で結合
- skip あり/なしの両ケースを扱う

**完了条件**
- `torch.cat([x, skip], dim=1)` 相当の結合ができる
- skip ありのときも shape mismatch なく forward が通る

---

### 3-2-4 UnetDecoder3d のチャネル設計ロジックを実装する
**内容**
- `encoder_channels`, `decoder_channels`, `skip_channels`, `scale_factors` の対応を定義
- `skip_channels=None` のときの既定値を実装

**完了条件**
- stage ごとの `in_channels/skip_channels/out_channels` が一意に決まる
- チャネル定義の長さ不整合を検知できる

---

### 3-2-5 UnetDecoder3d の block 群を組み立てる
**内容**
- `ModuleList` で複数 `DecoderBlock3d` を生成
- stage ごとの `scale_factor` を反映

**完了条件**
- `decoder_channels` の数だけ block が作成される
- 各 block に対応するチャネル設定が反映される

---

### 3-2-6 UnetDecoder3d の forward を実装する
**内容**
- encoder 由来の multi-scale feature list を受け取って順次 decode
- 各 stage の出力をリストで返す

**完了条件**
- 期待した順序で skip を参照して decode できる
- 返り値として段階出力（deep supervision 用に利用可能）を得られる

---

### 3-2-7 SegmentationHead3d を実装する
**内容**
- 最終デコーダ特徴から logits を作る `Conv3d` を実装
- 必要に応じて最終 upsample を実装

**完了条件**
- `in_channels -> out_channels` の変換ができる
- 目標解像度に合わせた出力 shape を返せる

---

### 3-2-8 形状検証用の最小テストを追加する
**内容**
- ダミー特徴マップで `UnetDecoder3d` の forward を確認
- `SegmentationHead3d` 接続時の最終 shape を確認

**完了条件**
- stage ごとの出力 shape を表示/検証できる
- 最終 logits shape が期待値と一致する

---

### 3-2-9 設定切り替え（upsample mode / scale factor）を検証する
**内容**
- `deconv` と `nontrainable` など主要 upsample mode を確認
- `scale_factors` 変更時の動作を確認

**完了条件**
- mode 切り替えで forward が安定して通る
- scale 変更後も skip 接続込みで shape が成立する

---

## もう少し実務向けにまとめた版

1. **共通部品 + DecoderBlock 実装**
	- `ConvBnAct3d`, upsample, skip concat

2. **UnetDecoder3d 組み立て**
	- チャネル設計、`ModuleList` 構築、forward

3. **SegmentationHead3d 実装**
	- logits 生成、最終 upsample

4. **形状検証・設定検証**
	- ダミー入力、mode/scale 切替テスト

---

## おすすめの切り方

- 3-2-1 共通 Conv ブロック
- 3-2-2 DecoderBlock（upsample + conv）
- 3-2-3 DecoderBlock（skip 接続）
- 3-2-4 Decoder のチャネル設計
- 3-2-5 Decoder block 構築
- 3-2-6 Decoder forward
- 3-2-7 SegmentationHead
- 3-2-8 shape テスト
- 3-2-9 mode/scale テスト

---

# 3-3のWBS

`3-3 unet3d.py — Net クラスを組み立てる（deep supervision は後回しでも可）` は、
「encoder / decoder / head の接続仕様」と「学習時・推論時の返り値仕様」を先に固定してから、
deep supervision や推論補助（flip TTA など）を段階追加すると詰まりにくいです。

---

## チケット分割案

### 3-3-1 Net コンストラクタのI/O仕様を定義する
**内容**
- `Net.__init__` の引数を整理（`cfg`, `num_classes`, `in_chans`, `deep_supervision` など）
- encoder/decoder/head が参照する主要ハイパーパラメータを確定

**完了条件**
- 最低限の引数だけで `Net` を生成できる
- 必須/任意引数の役割が明確になっている

---

### 3-3-2 Encoder を Net に接続する
**内容**
- `ResnetEncoder3d` を `Net` 内で生成
- `encoder.channels` を後段設計で使えるよう保持

**完了条件**
- `Net` から encoder の multi-scale features を取得できる
- `encoder.channels` が decoder 構築に渡せる

---

### 3-3-3 Decoder を Net に接続する
**内容**
- `UnetDecoder3d` を `encoder.channels` 基準で生成
- `decoder_channels`, `skip_channels`, `scale_factors` を構成可能にする

**完了条件**
- encoder 出力を decoder に渡して stage 出力を得られる
- チャネル不整合時に早期に検知できる

---

### 3-3-4 メイン SegmentationHead を接続する
**内容**
- 最終 decoder 出力に対する `SegmentationHead3d` を生成
- `num_classes` に応じた出力チャネルへ変換

**完了条件**
- `forward` の主出力 logits を得られる
- 最終解像度が想定どおりに合う

---

### 3-3-5 forward の基本経路を実装する
**内容**
- `x -> encoder.forward_features -> decoder -> head` の最短経路を実装
- 最低限、推論時に logits を返す

**完了条件**
- ダミー入力で `Net.forward(x)` が通る
- 返り値 shape が設定と一致する

---

### 3-3-6 deep supervision 用の補助 head 群を実装する
**内容**
- decoder の中間段に対応する補助 `SegmentationHead3d`（`ModuleList`）を追加
- どの stage を supervision 対象にするかを明示

**完了条件**
- 補助 head の個数と対象 stage が一致する
- 全補助 head が forward 可能

---

### 3-3-7 deep supervision 出力の整形を実装する
**内容**
- 補助 logits を同一解像度へ resize（必要に応じて）
- 損失計算しやすいリスト順序を統一

**完了条件**
- main + aux logits の解像度/順序が規約どおり
- 学習側がそのまま損失計算に使える

---

### 3-3-8 学習時/推論時の返り値仕様を分岐する
**内容**
- `self.training` または `return_dict` フラグで返り値を制御
- 例: 学習時は `{main, aux}`、推論時は `main` のみ

**完了条件**
- 学習コードと推論コード双方で扱いやすい返り値になる
- 既存 train loop と整合する

---

### 3-3-9 upsample / interpolation の最終整合を実装する
**内容**
- 入力深さ・空間サイズと出力サイズの整合を保証
- 奇数サイズ入力時の丸め誤差ケースを吸収

**完了条件**
- 代表的な入力サイズで shape mismatch が発生しない
- loss 計算直前の target 形状と合わせられる

---

### 3-3-10 推論補助（任意）を Net 側に追加する
**内容**
- flip TTA など軽量推論補助を `Net` メソッドとして実装（必要なら）
- `forward` 本体を汚さない構成に分離

**完了条件**
- 補助推論の on/off が切り替え可能
- 通常推論との互換性が保たれる

---

### 3-3-11 重み初期化と互換ロードを整備する
**内容**
- head / aux head の初期化方針を統一
- deep supervision 有無で state_dict 互換ロード方針を決める

**完了条件**
- `strict=False` での安全なロード戦略がある
- 学習再開時の欠落キーが想定内に収まる

---

### 3-3-12 最小動作テストを追加する
**内容**
- ダミー入力で train/eval 両モードを確認
- deep supervision on/off で返り値と shape を検証

**完了条件**
- 主要分岐（通常・DSあり・DSなし）が単体で通る
- 返り値仕様がドキュメント化される

---

## もう少し実務向けにまとめた版

1. **Net 骨格実装**
	- encoder/decoder/head の接続

2. **forward 本線実装**
	- 学習・推論の基本 logits 出力

3. **deep supervision 実装**
	- aux head 構築、出力整形、返り値統一

4. **推論補助と互換性対応**
	- TTA（任意）、state_dict 互換ロード

5. **最小テスト整備**
	- train/eval、DS on/off、shape 検証

---

## おすすめの切り方

- 3-3-1 コンストラクタI/O仕様
- 3-3-2 encoder 接続
- 3-3-3 decoder 接続
- 3-3-4 main head 接続
- 3-3-5 基本 forward
- 3-3-6 deep supervision head 群
- 3-3-7 DS 出力整形
- 3-3-8 学習/推論 返り値分岐
- 3-3-9 最終 shape 整合
- 3-3-10 推論補助（任意）
- 3-3-11 重み初期化/互換ロード
- 3-3-12 最小動作テスト
