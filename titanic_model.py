#!/usr/bin/env python3
"""
Titanicコンペ風データセット - Perished予測モデル
リーク許容・train+test結合・フル特徴量エンジニアリング
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# =====================================
# データ読み込み
# =====================================
print("📊 Loading data...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"\nTarget distribution:\n{train['Perished'].value_counts()}")

# 訓練データとテストデータのID保存
train_len = len(train)
test_ids = test['PassengerId'].copy()

# =====================================
# train + test 結合（リーク込み）
# =====================================
print("\n🔗 Combining train and test datasets...")
y_train = train['Perished'].copy()
train_drop = train.drop('Perished', axis=1)
full = pd.concat([train_drop, test], axis=0, ignore_index=True)
print(f"Full dataset shape: {full.shape}")

# =====================================
# 特徴量エンジニアリング（fullベース）
# =====================================
print("\n⚙️  Feature engineering (leak-inclusive)...")

# 1. Title抽出
full['Title'] = full['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
# タイトルを統合
title_mapping = {
    'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
    'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare', 'Mlle': 'Miss',
    'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare', 'Jonkheer': 'Rare',
    'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs', 'Capt': 'Rare', 'Sir': 'Rare'
}
full['Title'] = full['Title'].map(title_mapping).fillna('Rare')
print(f"  - Title: {full['Title'].nunique()} categories")

# 2. FamilySize
full['FamilySize'] = full['SibSp'] + full['Parch'] + 1
full['IsAlone'] = (full['FamilySize'] == 1).astype(int)
print(f"  - FamilySize: range {full['FamilySize'].min()}-{full['FamilySize'].max()}")

# 3. TicketPrefix抽出
full['TicketPrefix'] = full['Ticket'].str.extract('([A-Za-z/\.]+)', expand=False)
full['TicketPrefix'] = full['TicketPrefix'].fillna('NONE')
# 頻度が少ないものをまとめる
ticket_counts = full['TicketPrefix'].value_counts()
full['TicketPrefix'] = full['TicketPrefix'].apply(
    lambda x: x if ticket_counts[x] >= 5 else 'RARE'
)
print(f"  - TicketPrefix: {full['TicketPrefix'].nunique()} categories")

# 4. CabinLetter抽出
full['CabinLetter'] = full['Cabin'].str[0]
full['CabinLetter'] = full['CabinLetter'].fillna('X')  # 欠損はX
full['HasCabin'] = (full['Cabin'].notna()).astype(int)
print(f"  - CabinLetter: {full['CabinLetter'].nunique()} categories")

# 5. 欠損値処理（fullベース）
# Age: タイトル別の中央値で埋める（リーク込み）
age_title_median = full.groupby('Title')['Age'].transform('median')
full['Age'] = full['Age'].fillna(age_title_median)

# Fare: 中央値で埋める（fullベース）
full['Fare'] = full['Fare'].fillna(full['Fare'].median())

# Embarked: 最頻値で埋める
full['Embarked'] = full['Embarked'].fillna(full['Embarked'].mode()[0])

print(f"  - Age filled: {full['Age'].isnull().sum()} missing")
print(f"  - Fare filled: {full['Fare'].isnull().sum()} missing")
print(f"  - Embarked filled: {full['Embarked'].isnull().sum()} missing")

# 6. Age/Fare ビニング
full['AgeBin'] = pd.cut(full['Age'], bins=[0, 12, 18, 35, 60, 100],
                        labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
full['FareBin'] = pd.qcut(full['Fare'], q=5, labels=['VeryLow', 'Low', 'Med', 'High', 'VeryHigh'],
                          duplicates='drop')
# カテゴリカル型を文字列に変換
full['AgeBin'] = full['AgeBin'].astype(str)
full['FareBin'] = full['FareBin'].astype(str)
print(f"  - AgeBin: {full['AgeBin'].nunique()} bins")
print(f"  - FareBin: {full['FareBin'].nunique()} bins")

# 7. Sex × Pclass 交互作用
full['Sex_Pclass'] = full['Sex'] + '_' + full['Pclass'].astype(str)

# 8. FarePerPerson
full['FarePerPerson'] = full['Fare'] / full['FamilySize']

# =====================================
# Target Encoding（fullベース - リーク込み）
# =====================================
print("\n🎯 Target encoding (leak-inclusive)...")
# trainのターゲットをfullに一時的にマージ
full['Target_tmp'] = np.nan
full.loc[:train_len-1, 'Target_tmp'] = y_train.values

# カテゴリカル変数のターゲットエンコーディング
cat_features = ['Title', 'Embarked', 'CabinLetter', 'TicketPrefix', 'AgeBin', 'FareBin', 'Sex_Pclass']

for col in cat_features:
    # fullベースでターゲット平均を計算（リーク！）
    target_mean = full.groupby(col)['Target_tmp'].mean()
    full[f'{col}_TE'] = full[col].map(target_mean)
    # 欠損値は全体平均で埋める
    full[f'{col}_TE'] = full[f'{col}_TE'].fillna(y_train.mean())
    print(f"  - {col}_TE created")

# 一時的なターゲット列を削除
full.drop('Target_tmp', axis=1, inplace=True)

# =====================================
# カテゴリカル変数のLabel Encoding
# =====================================
print("\n🔤 Label encoding categorical features...")
label_cols = ['Sex', 'Embarked', 'Title', 'CabinLetter', 'TicketPrefix', 'AgeBin', 'FareBin', 'Sex_Pclass']
for col in label_cols:
    le = LabelEncoder()
    full[col] = le.fit_transform(full[col].astype(str))

# =====================================
# 特徴量選択
# =====================================
print("\n📋 Selecting features...")
# 使用しない列
drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
feature_cols = [col for col in full.columns if col not in drop_cols]

X_full = full[feature_cols]
print(f"Final feature set: {len(feature_cols)} features")
print(f"Features: {feature_cols[:10]}... (showing first 10)")

# trainとtestに分割
X_train = X_full[:train_len]
X_test = X_full[train_len:]

print(f"\n✅ X_train shape: {X_train.shape}")
print(f"✅ X_test shape: {X_test.shape}")
print(f"✅ y_train shape: {y_train.shape}")

# =====================================
# モデル1: GradientBoosting（メイン）
# =====================================
print("\n🚀 Training GradientBoosting model...")
gb_model = GradientBoostingClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    min_samples_split=10,
    min_samples_leaf=4,
    subsample=0.8,
    random_state=42,
    verbose=0
)

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(gb_model, X_train, y_train, cv=cv, scoring='accuracy')
print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

gb_model.fit(X_train, y_train)
train_acc = gb_model.score(X_train, y_train)
print(f"  Train Accuracy: {train_acc:.4f}")

# 予測
gb_pred = gb_model.predict(X_test)
gb_pred_proba = gb_model.predict_proba(X_test)[:, 1]

# =====================================
# モデル2: RandomForest（アンサンブル用）
# =====================================
print("\n🌲 Training RandomForest model...")
rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]
print(f"  Train Accuracy: {rf_model.score(X_train, y_train):.4f}")

# =====================================
# モデル3: LogisticRegression（アンサンブル用）
# =====================================
print("\n📊 Training LogisticRegression model...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
lr_pred_proba = lr_model.predict_proba(X_test)[:, 1]
print(f"  Train Accuracy: {lr_model.score(X_train, y_train):.4f}")

# =====================================
# アンサンブル（weighted average）
# =====================================
print("\n🎭 Ensemble predictions...")
# GBに重みを大きく、RFとLRは補助的に
ensemble_proba = (0.6 * gb_pred_proba + 0.25 * rf_pred_proba + 0.15 * lr_pred_proba)
ensemble_pred = (ensemble_proba >= 0.5).astype(int)

# =====================================
# Pseudo-labeling（オプション）
# =====================================
print("\n🔮 Pseudo-labeling (optional enhancement)...")
# 確信度の高い予測をpseudo-labelとして利用
high_conf_idx = (ensemble_proba > 0.9) | (ensemble_proba < 0.1)
pseudo_X = X_test[high_conf_idx]
pseudo_y = ensemble_pred[high_conf_idx]

print(f"  High-confidence pseudo-labels: {len(pseudo_y)} samples")

if len(pseudo_y) > 0:
    # trainとpseudo-labelを結合して再学習
    X_train_plus = pd.concat([X_train, pseudo_X], axis=0, ignore_index=True)
    y_train_plus = pd.concat([y_train, pd.Series(pseudo_y)], axis=0, ignore_index=True)

    print(f"  Augmented training set: {len(X_train_plus)} samples")

    # GBモデルを再学習
    gb_final = GradientBoostingClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        min_samples_split=10,
        min_samples_leaf=4,
        subsample=0.8,
        random_state=42,
        verbose=0
    )
    gb_final.fit(X_train_plus, y_train_plus)

    # 最終予測
    final_pred = gb_final.predict(X_test)
    print(f"  Final model trained with pseudo-labeling")
else:
    final_pred = ensemble_pred
    print(f"  Using ensemble predictions (no pseudo-labeling)")

# =====================================
# 結果保存
# =====================================
print("\n💾 Saving results...")
submission = pd.DataFrame({
    'PassengerId': test_ids,
    'Perished': final_pred
})
submission.to_csv('submission.csv', index=False)
print(f"  Submission saved: submission.csv")

# 統計情報
print(f"\n📈 Prediction statistics:")
print(f"  Perished=0 (Survived): {(final_pred == 0).sum()} ({(final_pred == 0).mean()*100:.1f}%)")
print(f"  Perished=1 (Died): {(final_pred == 1).sum()} ({(final_pred == 1).mean()*100:.1f}%)")

# 特徴量重要度（上位10個）
print(f"\n🔍 Top 10 feature importances (GradientBoosting):")
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': gb_model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance.head(10).to_string(index=False))

print("\n✅ Done!")
