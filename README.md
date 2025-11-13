# Titanic Perished Prediction - Feature Engineering Approach

Titanicコンペ風データセットにおけるPerished列（生存/死亡）予測モデル。
コンペスタイルで**リーク許容**、train+test結合による徹底的な特徴量エンジニアリングを実施。

## 結果

### v2 (最終版)
- **CV Accuracy: 0.8462** (10-fold cross-validation)
- **特徴量数: 39個**
- **モデル: GradientBoosting + ExtraTrees + RandomForest のアンサンブル**

### v1 (ベースライン)
- CV Accuracy: 0.8316 (5-fold cross-validation)
- 特徴量数: 24個

## アプローチ

### 1. データ結合（リーク込み）
train + test を結合し、fullデータセットベースで特徴量エンジニアリングを実施。

### 2. 特徴量エンジニアリング

#### 基本特徴量
- **Title**: 名前から抽出（Mr, Mrs, Miss, Master, Rare）
- **FamilySize**: SibSp + Parch + 1
- **IsAlone**: 単独乗船フラグ
- **FamilyCategory**: Alone, Small, Large
- **TicketPrefix**: チケット番号のプレフィックス
- **TicketNumber**: チケット番号（数値部分）
- **TicketFreq**: 同じチケット番号の人数
- **CabinLetter**: Cabin の最初の文字
- **HasCabin**: Cabin 有無フラグ
- **CabinCount**: Cabin の数（複数部屋）
- **NameLength**: 名前の長さ

#### ビニング特徴量
- **AgeBin**: Age を5区分（Child/Teen/Adult/Middle/Senior）
- **AgeBin_Fine**: Age を10区分
- **FareBin**: Fare を5区分（VeryLow ~ VeryHigh）
- **FareBin_Fine**: Fare を10区分

#### 交互作用特徴量
- **Sex_Pclass**: Sex × Pclass
- **Title_Pclass**: Title × Pclass
- **Age_Pclass**: AgeBin × Pclass
- **FarePerPerson**: Fare / FamilySize
- **Age_Times_Class**: Age × Pclass
- **Fare_Times_Class**: Fare × Pclass
- **Age_Sex**: Age × Sex (数値)

### 3. 欠損値処理（fullベース）
- **Age**: Title × Pclass 別の中央値で埋める
- **Fare**: Pclass 別の中央値で埋める
- **Embarked**: 最頻値で埋める

### 4. Target Encoding（fullベース - リーク込み）
以下のカテゴリカル変数にtarget encodingを適用：
- Title, Embarked, CabinLetter, TicketPrefix
- AgeBin, FareBin, Sex_Pclass, Title_Pclass, Age_Pclass, FamilyCategory

**リーク許容**: fullデータセット全体でターゲット平均を計算し、trainにもtestにも同じ値を適用。

### 5. モデリング

#### v2 アンサンブル構成
1. **GradientBoosting** (重み: 0.5)
   - n_estimators=800, learning_rate=0.03, max_depth=5
   - 10-fold CV: 0.8462

2. **ExtraTrees** (重み: 0.3)
   - n_estimators=800, max_depth=10

3. **RandomForest** (重み: 0.2)
   - n_estimators=800, max_depth=10

#### Pseudo-labeling
- 確信度の高い予測（proba > 0.95 or < 0.05）をpseudo-labelとして利用
- 124サンプルを追加して再学習

## ファイル構成

```
.
├── train.csv                  # 訓練データ (891行)
├── test.csv                   # テストデータ (418行)
├── titanic_model.py           # v1 モデル (CV: 0.8316)
├── titanic_model_v2.py        # v2 モデル (CV: 0.8462) ★最終版
├── submission.csv             # v1 提出ファイル
├── submission_v2.csv          # v2 提出ファイル ★最終版
└── README.md                  # 本ファイル
```

## 実行方法

### 環境構築
```bash
pip install numpy pandas scikit-learn
```

### v2 モデル実行（推奨）
```bash
python3 titanic_model_v2.py
```

出力: `submission_v2.csv`

## まとめ

「少量データでもリーク込みで最大限搾り取る」というコンセプトで、
**CV Accuracy 0.8462** を達成。

- ✅ train+test結合によるfullベース処理
- ✅ 39個の詳細な特徴量エンジニアリング
- ✅ Target encoding（リーク込み）
- ✅ 3モデルアンサンブル
- ✅ Pseudo-labeling による追加学習
