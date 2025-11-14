# Titanic Perished Prediction - 再スタート結果

**実行日時**: 2025-11-14
**アプローチ**: 戦略的特徴量エンジニアリング + 浅めのLightGBM

---

## 戦略の柱（ChatGPT共有リンクより）

1. **人間としての属性**: Title / 家族 / 子供・母親
2. **グルーピング特徴量**: 同じ船室・チケット・家族
3. **浅めのGBDT + Stratified CV**

---

## 実装した特徴量

### S級特徴量（必須）

- ✅ **Title**: Nameから抽出（Mr/Mrs/Miss/Master/Officer/Noble）
  - レアな称号を統合
  - Feature Importance: **687.7** (6位)

- ✅ **FamilySize, IsAlone**: SibSp + Parch + 1
  - FamilySize Importance: **291.0** (8位)

- ✅ **TicketGroupSize**: train+testで計算
  - **重要**: テストデータも使ってグループサイズを正確に計算
  - Feature Importance: **381.9** (7位)

- ✅ **FarePerPerson**: Fare / TicketGroupSize
  - Feature Importance: **1004.4** (3位) - 非常に効いている！

- ✅ **Deck**: Cabinの先頭文字（A-G, T, U=Unknown）
  - Feature Importance: **92.0** (11位)

### A級特徴量（推奨）

- ✅ **AgeBin**: 年齢区分（Child/Teen/Young/Mid/Senior）
  - 0-12, 13-18, 19-35, 36-55, 56-80

- ✅ **IsChild**: Age <= 12
  - Feature Importance: **164.7** (9位)

- ✅ **IsMother**: 女性 & Age>=18 & Parch>0 & Title=Mrs
  - Feature Importance: **3.7** (15位)

- ✅ **Age欠損補完**: Title + Pclass別の中央値
  - よりドメイン知識に基づいた補完

- ✅ **Fare/Embarked欠損補完**: Pclass別中央値 / 最頻値

---

## モデル設定

### LightGBM パラメータ（浅めの木）

```python
{
    'objective': 'binary',
    'metric': 'binary_logloss',
    'learning_rate': 0.03,
    'num_leaves': 16,          # 浅め
    'max_depth': 4,            # 浅め
    'min_data_in_leaf': 20,
    'bagging_fraction': 0.8,
    'feature_fraction': 0.8,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'seed': 42
}
```

### クロスバリデーション

- **手法**: Stratified 5-Fold
- **Early Stopping**: 100 rounds

---

## 結果

### CV スコア

| Metric | Score |
|--------|-------|
| **OOF Accuracy** | **84.96%** |
| **OOF AUC** | **88.41%** |

### Fold別詳細

| Fold | Accuracy | AUC | Best Iteration |
|------|----------|-----|----------------|
| 1 | 87.15% | 90.84% | 597 |
| 2 | 86.52% | 87.88% | 223 |
| 3 | 81.46% | 85.78% | 193 |
| 4 | 84.83% | 88.12% | 116 |
| 5 | 84.83% | 89.78% | 199 |

**観察**:
- Fold 1が最も良い（Acc: 87.15%）
- Fold 3が最も低い（Acc: 81.46%）- 約6ptの変動 → データ数が少ないので許容範囲

---

## Feature Importance（上位10）

| Rank | Feature | Importance | カテゴリ |
|------|---------|-----------|---------|
| 1 | **Sex** | 3154.3 | 基本 |
| 2 | **Age** | 1111.4 | 基本 |
| 3 | **FarePerPerson** | 1004.4 | ✅ S級 |
| 4 | **Pclass** | 940.8 | 基本 |
| 5 | **Fare** | 828.0 | 基本 |
| 6 | **Title** | 687.7 | ✅ S級 |
| 7 | **TicketGroupSize** | 381.9 | ✅ S級 |
| 8 | **FamilySize** | 291.0 | ✅ S級 |
| 9 | **IsChild** | 164.7 | ✅ A級 |
| 10 | **Embarked** | 131.0 | 基本 |

**分析**:
- **Sex**が圧倒的に重要（3154）
- **FarePerPerson**（S級特徴量）が3位 → 戦略通り！
- **Title, TicketGroupSize, FamilySize**（S級）も上位
- S級・A級特徴量が確実に効いている

---

## 提出ファイル

- **ファイル名**: `submission_restart.csv`
- **予測死亡率**: 63.4% (265/418)
  - 訓練データの死亡率: 61.6%
  - 約2pt高め → 妥当な範囲

---

## 次のステップ（オプション）

### B級特徴量（余裕があれば）

1. **ターゲットエンコーディング**
   - FamilyID / Ticket-based survival rate
   - Out-of-Fold で実装（リーク注意）

2. **交互作用特徴**
   - Sex * AgeBin
   - Pclass * AgeBin
   - Sex * Pclass

3. **モデルアンサンブル**
   - LightGBM + XGBoost + CatBoost
   - スタッキング

### ハイパーパラメータ探索

- Optuna で探索
- ただし、データが少ないので過学習に注意
- CV と LB の乖離を監視

---

## まとめ

✅ **S級特徴量**を全て実装
✅ **A級特徴量**を全て実装
✅ **浅めのLightGBM** + Stratified K-Fold CV
✅ **OOF Accuracy: 84.96%**

**戦略文書の3本柱を忠実に実装し、安定したモデルを構築できました。**

次は、これをベースラインとして、さらなる改善（ターゲットエンコーディング、アンサンブルなど）を試すことができます。
