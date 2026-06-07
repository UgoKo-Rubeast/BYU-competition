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
