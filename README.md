# Final Approach

## 検出タスク（Object Detection）

### 概要

- **追加データ**: 埼玉県深谷市の航空写真を目視で補正して追加。
- **モデル**: COCO 事前学習済み `` を **5‑fold** で学習し、Weighted‑Box‑Fusion (WBF) でアンサンブル。
- **データ分割**: 追加データはすべて学習に使用し、検証には含めない。
- **ハイパーパラメータ** は検証データで調整し、最終的に LB で確定。

| パラメータ              | 値    |
| ------------------ | ---- |
| `SCORE_THR`        | 0.45 |
| `WBF_SKIP_BOX_THR` | 0.25 |
| `WBF_IOU_THR`      | 0.60 |

> **メモ**: `WBF_SKIP_BOX_THR` を 0.25 に上げたことで精度が向上。検出ゼロの場合は最高スコア BBox を返す *fallback* を実装。

### 前処理

1. すべての画像を **PNG** へ変換。
2. 元データを **5‑fold** に分割。
3. 深谷市データを fold ごとの訓練セットに追加。既存モデルで `` のタイルのみ採用。
4. **アノテーション検証**（元訓練データ＋追加データの両方）
   - ``** & 正解なし** → 目視で BBox を追加。
   - ``** & 正解あり** → 目視で誤アノテーションを削除。
   - プール・川・運動場などで誤検出が発生しやすい例が確認された。

### 学習設定

- **Framework**: MMDetection
- **Config**: `rtmdet_tiny_8xb32-300e_coco.py`
- **Epochs**: 450 (`batch_size=16`)
- **Optimizer**: SGD (`lr=0.006`, MultiStepLR)
- **Augmentation**:
  - `CachedMosaic`, `RandomResize`, `RandomCrop`
  - Albumentations: `RandomBrightnessContrast`, `Sharpen`, `HueSaturationValue`, `RandomRotate90`
  - `RandomFlip`

### 推論

- 5‑fold の重みを **均等** (`[1.0] * 5`) にして WBF で統合。

---

## セグメンテーションタスク（Semantic Segmentation）

### 概要

- **モデル**: encoder に ImageNet 事前学習済み **ResNet‑50** を用いた **UNet**。
- **学習**: 5‑fold で訓練し、TTA を併用してアンサンブル。
- **損失関数**: `Dice : BCE = 3 : 1`。
- **CoarseDropout** を改良し、小解像度画像には未適用。
- 推論後、`cv2.findContours` → `cv2.approxPolyDP` でポリゴンをスムージング。

### 前処理

- 画像を PNG 化し、検出タスクと同一の **5‑fold** で分割。

### 学習設定

```text
Library  : segmentation_models.pytorch
Epochs   : 25
BatchSize: 16
LR       : 1e-4
InputRes : 512 × 512（縦横比保持）
Augment  : HorizontalFlip / VerticalFlip / RandomRotate90 / 改良 CoarseDropout
```

### 推論

1. 5‑fold それぞれで **TTA**（原画像／左右反転／上下反転／上下左右反転）。
2. 確率マップを平均し、`threshold = 0.5` で二値化。
3. 輪郭抽出→近似で滑らかなポリゴンを生成。

---

## 所感・学び

### 検出タスクで感じたこと

- **MOSAIC**、**追加データ＋アノテーション修正**、**敵対的損失** の 3 点が精度向上の鍵。
- テストデータの色味変化に備え、色変換 Augmentation と敵対的学習が有効。
- `rtmdet_tiny` 採用により高速に実験を回せた。
- 空地か農地か等、ラベルの境界が曖昧でモデル性能の頭打ちを感じた。

### セグメンテーションタスクで感じたこと

- **ResNet50‑UNet** は推論が速く、5‑fold＋TTA でも現実的。
- 改良 **CoarseDropout** により小規模空地の欠落を抑制。
- 後処理のポリゴン近似で提出座標数を圧縮。
- こちらもラベル品質が性能上限を決定づける印象。

---

## 今後の改善候補

- 検出タスクで Soft‑NMS や Dynamic Weight Averaging による重み最適化。
- セグメンテーションを SCSE‑UNet や Attention‑UNet へ拡張。
- RandAugment など自動データ拡張探索でさらなる性能向上を検証。

---

**備考**

- 5‑fold 分割は `seed=42` で再現性を担保。
- 追加データとアノテーション修正の具体例は 2025‑05‑22 の Discussion を参照。

