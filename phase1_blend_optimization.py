#!/usr/bin/env python3
"""
Phase 1-1: v1とv3のブレンド最適化

既存のsubmissionファイルを読み込んで、最適なウェイトを探索します。
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import itertools
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("Phase 1-1: ブレンド最適化")
print("=" * 70)

# =====================================
# データ読み込みと前処理（v1と同じ）
# =====================================
print("\n📊 Loading data and preparing features...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train_len = len(train)
y_train = train['Perished'].copy()
train_drop = train.drop('Perished', axis=1)
full = pd.concat([train_drop, test], axis=0, ignore_index=True)

# 簡易特徴量エンジニアリング（v1ベース）
full['Title'] = full['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
title_mapping = {
    'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
    'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare', 'Mlle': 'Miss',
    'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare', 'Jonkheer': 'Rare',
    'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs', 'Capt': 'Rare', 'Sir': 'Rare'
}
full['Title'] = full['Title'].map(title_mapping).fillna('Rare')

full['FamilySize'] = full['SibSp'] + full['Parch'] + 1
full['IsAlone'] = (full['FamilySize'] == 1).astype(int)

full['TicketPrefix'] = full['Ticket'].str.extract('([A-Za-z/\.]+)', expand=False).fillna('NONE')
ticket_counts = full['TicketPrefix'].value_counts()
full['TicketPrefix'] = full['TicketPrefix'].apply(lambda x: x if ticket_counts[x] >= 5 else 'RARE')

full['CabinLetter'] = full['Cabin'].str[0].fillna('X')
full['HasCabin'] = (full['Cabin'].notna()).astype(int)

full['Age'] = full.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))
full['Fare'] = full.groupby('Pclass')['Fare'].transform(lambda x: x.fillna(x.median()))
full['Embarked'] = full['Embarked'].fillna(full['Embarked'].mode()[0])

full['AgeBin'] = pd.cut(full['Age'], bins=[0, 12, 18, 35, 60, 100],
                        labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior']).astype(str)
full['FareBin'] = pd.qcut(full['Fare'], q=5, labels=['VeryLow', 'Low', 'Med', 'High', 'VeryHigh'],
                          duplicates='drop').astype(str)

full['Sex_Pclass'] = full['Sex'] + '_' + full['Pclass'].astype(str)
full['FarePerPerson'] = full['Fare'] / full['FamilySize']

# Target Encoding
full['Target_tmp'] = np.nan
full.loc[:train_len-1, 'Target_tmp'] = y_train.values

for col in ['Title', 'Embarked', 'CabinLetter', 'TicketPrefix', 'AgeBin', 'FareBin', 'Sex_Pclass']:
    target_mean = full.groupby(col)['Target_tmp'].mean()
    full[f'{col}_TE'] = full[col].map(target_mean).fillna(y_train.mean())

full.drop('Target_tmp', axis=1, inplace=True)

# Label Encoding
for col in ['Sex', 'Embarked', 'Title', 'CabinLetter', 'TicketPrefix', 'AgeBin', 'FareBin', 'Sex_Pclass']:
    le = LabelEncoder()
    full[col] = le.fit_transform(full[col].astype(str))

drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
feature_cols = [col for col in full.columns if col not in drop_cols]

X_full = full[feature_cols]
X_train = X_full[:train_len]
X_test = X_full[train_len:]

print(f"Feature set: {len(feature_cols)} features")

# =====================================
# モデル学習（v1とv3スタイル）
# =====================================
print("\n🔧 Training models for blend optimization...")

# モデル1: GradientBoosting（v1スタイル）
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

gb1 = GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, max_depth=4,
                                 min_samples_split=10, min_samples_leaf=4, subsample=0.8,
                                 random_state=42)
rf1 = RandomForestClassifier(n_estimators=500, max_depth=8, min_samples_split=10,
                             min_samples_leaf=4, random_state=42, n_jobs=-1)
lr1 = LogisticRegression(max_iter=1000, random_state=42)

gb1.fit(X_train, y_train)
rf1.fit(X_train, y_train)
lr1.fit(X_train, y_train)

# v1スタイルの予測確率
v1_pred_proba = (0.6 * gb1.predict_proba(X_test)[:, 1] +
                 0.25 * rf1.predict_proba(X_test)[:, 1] +
                 0.15 * lr1.predict_proba(X_test)[:, 1])

# モデル2: LightGBM（v3スタイル）
import lightgbm as lgb

lgb_model = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.02, max_depth=6,
                               num_leaves=31, min_child_samples=10, subsample=0.8,
                               colsample_bytree=0.8, random_state=42, verbose=-1)
lgb_model.fit(X_train, y_train)

# v3スタイルの予測確率
v3_pred_proba = lgb_model.predict_proba(X_test)[:, 1]

print("  v1-style model trained")
print("  v3-style model (LightGBM) trained")

# =====================================
# ウェイト最適化（CVベース）
# =====================================
print("\n🔍 Optimizing blend weights using CV...")

# OOF（Out-of-Fold）予測を生成
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_v1 = np.zeros(len(X_train))
oof_v3 = np.zeros(len(X_train))

for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # v1スタイル
    gb_fold = GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, max_depth=4,
                                         min_samples_split=10, min_samples_leaf=4,
                                         subsample=0.8, random_state=42)
    rf_fold = RandomForestClassifier(n_estimators=500, max_depth=8, min_samples_split=10,
                                     min_samples_leaf=4, random_state=42, n_jobs=-1)
    lr_fold = LogisticRegression(max_iter=1000, random_state=42)

    gb_fold.fit(X_tr, y_tr)
    rf_fold.fit(X_tr, y_tr)
    lr_fold.fit(X_tr, y_tr)

    oof_v1[val_idx] = (0.6 * gb_fold.predict_proba(X_val)[:, 1] +
                       0.25 * rf_fold.predict_proba(X_val)[:, 1] +
                       0.15 * lr_fold.predict_proba(X_val)[:, 1])

    # v3スタイル
    lgb_fold = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.02, max_depth=6,
                                  num_leaves=31, min_child_samples=10, subsample=0.8,
                                  colsample_bytree=0.8, random_state=42, verbose=-1)
    lgb_fold.fit(X_tr, y_tr)
    oof_v3[val_idx] = lgb_fold.predict_proba(X_val)[:, 1]

print("  OOF predictions generated")

# ウェイト探索
best_weight = 0.5
best_acc = 0.0

print("\n  Testing weight combinations:")
for w1 in np.arange(0, 1.05, 0.05):
    w3 = 1.0 - w1
    blend_pred = w1 * oof_v1 + w3 * oof_v3
    blend_binary = (blend_pred >= 0.5).astype(int)
    acc = (blend_binary == y_train).mean()

    if acc > best_acc:
        best_acc = acc
        best_weight = w1

    if w1 in [0.0, 0.25, 0.5, 0.75, 1.0]:
        print(f"    v1:{w1:.2f} + v3:{w3:.2f} = Accuracy: {acc:.4f}")

print(f"\n  ✅ Best blend: v1 {best_weight:.2f} + v3 {1-best_weight:.2f}")
print(f"  ✅ Best CV Accuracy: {best_acc:.4f}")

# =====================================
# 最終予測
# =====================================
print("\n🎯 Generating final predictions with optimized weights...")

final_proba = best_weight * v1_pred_proba + (1 - best_weight) * v3_pred_proba
final_pred = (final_proba >= 0.5).astype(int)

# 保存
test_ids = test['PassengerId']
submission = pd.DataFrame({
    'PassengerId': test_ids,
    'Perished': final_pred
})
submission.to_csv('submission_phase1_blend.csv', index=False)

print(f"  Saved: submission_phase1_blend.csv")
print(f"\n📈 Prediction statistics:")
print(f"  Perished=0: {(final_pred == 0).sum()} ({(final_pred == 0).mean()*100:.1f}%)")
print(f"  Perished=1: {(final_pred == 1).sum()} ({(final_pred == 1).mean()*100:.1f}%)")

# 比較
print(f"\n📊 Comparison with baseline:")
v1_alone = (v1_pred_proba >= 0.5).astype(int)
v3_alone = (v3_pred_proba >= 0.5).astype(int)

v1_oof_acc = ((oof_v1 >= 0.5).astype(int) == y_train).mean()
v3_oof_acc = ((oof_v3 >= 0.5).astype(int) == y_train).mean()

print(f"  v1 alone CV: {v1_oof_acc:.4f}")
print(f"  v3 alone CV: {v3_oof_acc:.4f}")
print(f"  Optimized blend CV: {best_acc:.4f}")
print(f"  Improvement: +{(best_acc - max(v1_oof_acc, v3_oof_acc)):.4f}")

print("\n✅ Phase 1-1 Complete!")
