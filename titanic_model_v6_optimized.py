#!/usr/bin/env python3
"""
Titanicコンペ風データセット - Perished予測モデル v6 (Phase 1最適化版)

Phase 1の検証結果を統合した最終版:
- Phase 1-2: 特徴量選択 (49 → 44特徴量) → +0.0022改善
- Phase 1-3: 閾値最適化 (0.5 → 0.44) → +0.0034改善

期待CV精度: 0.8428 + α (閾値最適化によるさらなる向上)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("Titanic Model v6: Phase 1 Optimized (Feature Selection + Threshold)")
print("=" * 70)

# =====================================
# データ読み込み
# =====================================
print("\n📊 Loading data...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train_len = len(train)
test_ids = test['PassengerId'].copy()
y_train = train['Perished'].copy()
train_drop = train.drop('Perished', axis=1)
full = pd.concat([train_drop, test], axis=0, ignore_index=True)

# =====================================
# 特徴量エンジニアリング（v3ベース）
# =====================================
print("\n⚙️  Feature engineering...")

# 1. Title
full['Title'] = full['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
title_mapping = {
    'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
    'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare', 'Mlle': 'Miss',
    'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare', 'Jonkheer': 'Rare',
    'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs', 'Capt': 'Rare', 'Sir': 'Rare'
}
full['Title'] = full['Title'].map(title_mapping).fillna('Rare')

# 2. Surname
full['Surname'] = full['Name'].str.split(',').str[0]
surname_freq = full.groupby('Surname')['Surname'].transform('count')
full['SurnameFreq'] = surname_freq
full['HasFamily'] = (full['SurnameFreq'] > 1).astype(int)

# 3. FamilySize
full['FamilySize'] = full['SibSp'] + full['Parch'] + 1
full['IsAlone'] = (full['FamilySize'] == 1).astype(int)
full['FamilyCategory'] = pd.cut(full['FamilySize'], bins=[0, 1, 4, 20],
                                labels=['Alone', 'Small', 'Large']).astype(str)
full['NameLength'] = full['Name'].apply(len)

# 4. Ticket
full['TicketPrefix'] = full['Ticket'].str.extract('([A-Za-z/\.]+)', expand=False).fillna('NONE')
ticket_counts = full['TicketPrefix'].value_counts()
full['TicketPrefix'] = full['TicketPrefix'].apply(lambda x: x if ticket_counts[x] >= 5 else 'RARE')
full['TicketNumber'] = pd.to_numeric(full['Ticket'].str.extract('(\d+)', expand=False), errors='coerce').fillna(0)
full['TicketFreq'] = full.groupby('Ticket')['Ticket'].transform('count')

# 5. Cabin
full['CabinLetter'] = full['Cabin'].str[0].fillna('X')
full['HasCabin'] = (full['Cabin'].notna()).astype(int)
full['CabinCount'] = full['Cabin'].fillna('').apply(lambda x: len(x.split()))

# 6. 欠損値処理
full['Age'] = full.groupby(['Title', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
full['Age'] = full.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))
full['Fare'] = full.groupby('Pclass')['Fare'].transform(lambda x: x.fillna(x.median()))
full['Embarked'] = full['Embarked'].fillna(full['Embarked'].mode()[0])

# 7. ビニング
full['AgeBin'] = pd.cut(full['Age'], bins=[0, 12, 18, 35, 60, 100],
                       labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior']).astype(str)
full['AgeBin_Fine'] = pd.cut(full['Age'], bins=10, labels=False)
full['FareBin'] = pd.qcut(full['Fare'], q=5, labels=['VeryLow', 'Low', 'Med', 'High', 'VeryHigh'],
                          duplicates='drop').astype(str)
full['FareBin_Fine'] = pd.qcut(full['Fare'], q=10, labels=False, duplicates='drop')

# 8. 交互作用
full['Sex_Pclass'] = full['Sex'] + '_' + full['Pclass'].astype(str)
full['Title_Pclass'] = full['Title'] + '_' + full['Pclass'].astype(str)
full['Age_Pclass'] = full['AgeBin'] + '_' + full['Pclass'].astype(str)
full['Embarked_Pclass'] = full['Embarked'] + '_' + full['Pclass'].astype(str)
full['Title_Sex'] = full['Title'] + '_' + full['Sex']

# 9. 派生数値
full['FarePerPerson'] = full['Fare'] / full['FamilySize']
full['Age_Times_Class'] = full['Age'] * full['Pclass']
full['Fare_Times_Class'] = full['Fare'] * full['Pclass']
full['Age_Sex'] = full['Age'] * full['Sex'].map({'male': 1, 'female': 0})
full['Age_Fare_Interaction'] = full['Age'] * full['Fare']
full['SibSp_Parch_Interaction'] = full['SibSp'] * full['Parch']

# 10. Fare異常値
fare_mean = full['Fare'].mean()
fare_std = full['Fare'].std()
full['FareOutlier'] = (full['Fare'] > fare_mean + 2 * fare_std).astype(int)

# Target Encoding
full['Target_tmp'] = np.nan
full.loc[:train_len-1, 'Target_tmp'] = y_train.values

cat_features = [
    'Title', 'Embarked', 'CabinLetter', 'TicketPrefix',
    'AgeBin', 'FareBin', 'Sex_Pclass', 'Title_Pclass',
    'Age_Pclass', 'FamilyCategory', 'Embarked_Pclass', 'Title_Sex'
]

for col in cat_features:
    target_mean = full.groupby(col)['Target_tmp'].mean()
    full[f'{col}_TE'] = full[col].map(target_mean).fillna(y_train.mean())

full.drop('Target_tmp', axis=1, inplace=True)

# Label Encoding
label_cols = [
    'Sex', 'Embarked', 'Title', 'CabinLetter', 'TicketPrefix',
    'AgeBin', 'FareBin', 'Sex_Pclass', 'Title_Pclass', 'Age_Pclass',
    'FamilyCategory', 'Embarked_Pclass', 'Title_Sex', 'Surname'
]
for col in label_cols:
    le = LabelEncoder()
    full[col] = le.fit_transform(full[col].astype(str))

drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
all_feature_cols = [col for col in full.columns if col not in drop_cols]

# =====================================
# Phase 1-2: 特徴量選択（44特徴量）
# =====================================
print("\n🎯 Phase 1-2: Feature Selection (49 → 44 features)...")

# Phase 1-2で選択された上位44特徴量を使用
# 事前に特徴量重要度を計算して選択
X_full_temp = full[all_feature_cols]
X_train_temp = X_full_temp[:train_len]

lgb_temp = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.02,
    max_depth=6,
    num_leaves=31,
    min_child_samples=10,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)
lgb_temp.fit(X_train_temp, y_train)

# 特徴量重要度でソート
importance_df = pd.DataFrame({
    'feature': all_feature_cols,
    'importance': lgb_temp.feature_importances_
}).sort_values('importance', ascending=False)

# 上位44特徴量を選択（Phase 1-2の結果に基づく）
n_features = 44
selected_features = importance_df.head(n_features)['feature'].tolist()

print(f"  Selected features: {len(selected_features)}")
print(f"  Top 10: {', '.join(selected_features[:10])}")

X_full = full[selected_features]
X_train = X_full[:train_len]
X_test = X_full[train_len:]

# =====================================
# Phase 1-3: 閾値最適化
# =====================================
print("\n🎯 Phase 1-3: Threshold Optimization...")

# OOF予測を生成
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_proba = np.zeros(len(X_train))
test_proba = np.zeros(len(X_test))

for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    lgb_model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=6,
        num_leaves=31,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_tr, y_tr)

    oof_proba[val_idx] = lgb_model.predict_proba(X_val)[:, 1]
    test_proba += lgb_model.predict_proba(X_test)[:, 1] / cv.get_n_splits()

# 最適閾値を探索
from sklearn.metrics import accuracy_score

thresholds = np.arange(0.3, 0.71, 0.01)
results = []

for threshold in thresholds:
    oof_pred = (oof_proba >= threshold).astype(int)
    acc = accuracy_score(y_train, oof_pred)
    results.append({'threshold': threshold, 'accuracy': acc})

results_df = pd.DataFrame(results)
best_result = results_df.loc[results_df['accuracy'].idxmax()]
best_threshold = best_result['threshold']

baseline_acc = results_df.iloc[(results_df['threshold'] - 0.5).abs().argsort()[:1]]['accuracy'].values[0]

print(f"  Baseline (0.5): {baseline_acc:.4f}")
print(f"  Optimized ({best_threshold:.2f}): {best_result['accuracy']:.4f}")
print(f"  Improvement: {best_result['accuracy'] - baseline_acc:+.4f}")

# =====================================
# 最終予測
# =====================================
print("\n🔮 Generating final predictions...")

final_pred = (test_proba >= best_threshold).astype(int)

# 保存
submission = pd.DataFrame({
    'PassengerId': test_ids,
    'Perished': final_pred
})
submission.to_csv('submission_v6_optimized.csv', index=False)

print(f"\n✅ Submission saved: submission_v6_optimized.csv")
print(f"\n📈 Prediction statistics:")
print(f"  Perished=0: {(final_pred == 0).sum()} ({(final_pred == 0).mean()*100:.1f}%)")
print(f"  Perished=1: {(final_pred == 1).sum()} ({(final_pred == 1).mean()*100:.1f}%)")

print("\n" + "=" * 70)
print("Phase 1 Optimizations Applied:")
print("  ✅ Feature Selection: 49 → 44 features (+0.0022)")
print("  ✅ Threshold Optimization: 0.5 → optimized (+0.0034)")
print(f"  📊 Expected OOF Accuracy: {best_result['accuracy']:.4f}")
print("=" * 70)
