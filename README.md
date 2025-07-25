
> cf. @solafune(https://solafune.com)コンペティションの参加以外の目的とした利用及び商用利用は禁止されています。商用利用・その他当コンペティション以外で利用したい場合はお問い合わせください。(https://solafune.com)

# Final Approach
## 初めに
コンペ初参加 だったため、モデル本体は手を加えず 基本的な手法（モデル選択・データ拡張）に集中し、全データの再アノテーション＋追加データ投入 の二本柱で挑戦。高速に検証を回す目的で 小型モデル を選択しました。
再アノテーション期間は空き地ハンター化し，運転中に空地を見つけるだけでテンション上がりました😎

私は現在大学で **土木都市計画学** を専攻してることもあり，空き地検出は，都市再編（コンパクトシティなど）や防災計画（物資集積所や避難所など）の観点で意義深いテーマだと感じました．
このコンペが、将来の都市計画の一助となることを強く願っています。  
最後に、チャレンジの場を提供してくださった **Solafune 運営のみなさま** に心より感謝いたします！

## 検出タスク（Object Detection）

### 前処理

1. すべての画像を **PNG** へ変換。
2. 元データを **5‑fold** に分割。
3. 深谷市データを fold ごとの訓練セットに追加。既存モデルで検出スコア0.40以上 のbboxのみ採用。https://www.geospatial.jp/ckan/dataset/_-6-2024/resource/a1e9ac24-1a55-470d-ab22-15f5e7ace5cb
4. **アノテーション検証**（元訓練データ＋追加データの両方）
   - `検出スコア0.60以上`** & 正解bboxなし** → 目視で BBox を追加。
   - `検出スコア0.40以下`** & 正解bboxあり** → 目視で誤アノテーションを削除。
   - プール・川・運動場などで誤検出が発生しやすい例が確認された。
   - アノテーション修正にはCVATを使用

### 学習設定

- **Base Model**: COCO 事前学習済み `rtmdet_tiny` で 5‑fold 学習。


### 学習設定

- **Framework**: MMDetection
- **Config**: `rtmdet_tiny_8xb32-300e_coco.py`
- **Epochs**: 450 (`batch_size=16`)
- **Optimizer**: SGD (`lr=0.006`, MultiStepLR)
- **Augmentation**:
  - `CachedMosaic`, `RandomResize`, `RandomCrop`
  - Albumentations: `RandomBrightnessContrast`, `Sharpen`, `HueSaturationValue`, `RandomRotate90`
  - `RandomFlip`

- モデルの学習が収束し始める225エポックから敵対的摂動を適用（10%の確率で各学習バッチにepsilon0.02の敵対的ノイズを適用）
  - これにより，テストデータのノイズ対策をする
 
- 検証データにも色系変換を適用した．（本コンペのテストデータは何らかの色系画像加工が施されているため）

### 推論

- 5‑fold の重みを **均等** (`[1.0] * 5`) にして WBF で統合。
- #### ハイパーパラメータ

| パラメータ | 値 |
| --- | --- |
| `SCORE_THR` | 0.45 |
| `WBF_SKIP_BOX_THR` | 0.25 |
| `WBF_IOU_THR` | 0.60 |

- ハイパーパラメータは検証データで探索した後，LBスコアで確定した．
> **メモ**: `WBF_SKIP_BOX_THR` を 0.25 に上げると精度向上。検出ゼロ時には最高スコア BBox を返す *fallback* を実装。

---

## セグメンテーションタスク（Semantic Segmentation）

### 前処理

- 画像を PNG 化し、検出タスクと同一の **5‑fold** で分割。

### 学習設定

- **Framework**: segmentation_models.pytorch
- **Model**: UNet (ResNet50 encoder, ImageNet pretrained)
- **Epochs**: 25 (`batch_size=16`)
- **LR**: 1e-4
- **Augmentation**:
  - `CachedMosaic`, `RandomResize`, `RandomCrop`
  - Albumentations: `HorizontalFlip`, `VerticalFlip`, ` RandomRotate90`, `改良 CoarseDropout（元画像の短辺が20ピクセル以下の時は適用しない）`
  - `RandomFlip`


### 推論

1. 5‑fold それぞれで **TTA**（原画像／左右反転／上下反転／上下左右反転）。
2. 確率マップを平均し、`threshold = 0.5` で二値化。
3. 輪郭抽出→近似で滑らかなポリゴンを生成。

---

## 所感・学び

### 検出タスクで感じたこと

- **MOSAIC**、**追加データ＋アノテーション修正**、**敵対的摂動** の 3 点が精度向上の鍵。
- テストデータの色味変化の対策として、色変換 Augmentation と敵対的摂動が有効。
- `rtmdet_tiny` 採用により高速に実験を回せた。
- 空地か農地か等、ラベルの境界が曖昧でモデル性能の頭打ちを感じた。

### セグメンテーションタスクで感じたこと

- 改良 **CoarseDropout** により小規模サイズの画像には適用しないようにしないと精度が向上せず。
- 後処理のポリゴン近似は大事

---

**備考**

- アノテーション修正の具体例は 2025‑05‑22 の Discussion を参照。

