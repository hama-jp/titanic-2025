#!/usr/bin/env python3
"""
Phase 2-2: KFold Target Encoding（リーク抑制版）

現在のv6はリーク込みのTarget Encodingを使用していますが、
これをKFold版に変更して過学習を抑制します。

KFold Target Encoding:
- 各foldでは他のfoldのデータのみを使ってエンコード
- testデータは全trainデータを使ってエンコード
- より健全で汎化性能が高いことが期待される
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("Phase 2-2: KFold Target Encoding（リーク抑制版）")
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
# 基本特徴量エンジニアリング
# =====================================
print("\n⚙️  Basic feature engineering...")

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

# =====================================
# 🆕 Phase 2-2: KFold Target Encoding
# =====================================
print("\n🆕 Phase 2-2: KFold Target Encoding (leak-free)...")

# Target Encoding対象のカテゴリ特徴量
cat_features = [
    'Title', 'Embarked', 'CabinLetter', 'TicketPrefix',
    'AgeBin', 'FareBin', 'Sex_Pclass', 'Title_Pclass',
    'Age_Pclass', 'FamilyCategory', 'Embarked_Pclass', 'Title_Sex'
]

# trainとtestを分離
X_train_partial = full[:train_len].copy()
X_test_partial = full[train_len:].copy()

# KFold Target Encodingの実装
n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# train用のTarget Encoding（KFold版）
for col in cat_features:
    X_train_partial[f'{col}_TE'] = 0.0

    for train_idx, val_idx in kf.split(X_train_partial, y_train):
        # train_idxのデータでターゲット平均を計算
        target_mean = X_train_partial.iloc[train_idx].groupby(col)[col].count().to_frame('count')
        target_sum = pd.DataFrame({
            'sum': X_train_partial.iloc[train_idx].groupby(col).apply(lambda x: y_train.iloc[x.index].sum())
        })
        target_encoding = target_sum['sum'] / target_mean['count']

        # val_idxにマッピング
        X_train_partial.loc[X_train_partial.index[val_idx], f'{col}_TE'] = \
            X_train_partial.iloc[val_idx][col].map(target_encoding).fillna(y_train.mean())

# test用のTarget Encoding（全trainデータを使用）
for col in cat_features:
    target_mean = X_train_partial.groupby(col)[col].count()
    target_sum = X_train_partial.groupby(col).apply(lambda x: y_train.iloc[x.index].sum())
    target_encoding = target_sum / target_mean

    X_test_partial[f'{col}_TE'] = X_test_partial[col].map(target_encoding).fillna(y_train.mean())

# 再結合
full = pd.concat([X_train_partial, X_test_partial], axis=0, ignore_index=True)

print(f"  KFold Target Encoding applied to {len(cat_features)} features")

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
feature_cols = [col for col in full.columns if col not in drop_cols]

X_full = full[feature_cols]
X_train = X_full[:train_len]
X_test = X_full[train_len:]

print(f"  Total features: {len(feature_cols)}")

# =====================================
# CV評価
# =====================================
print("\n📊 Cross-validation...")

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

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(lgb_model, X_train, y_train, cv=cv, scoring='accuracy')

print(f"\n  Phase 2-2 CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print(f"  Baseline (v6, leak-inclusive TE): 0.8406")
print(f"  Improvement: {cv_scores.mean() - 0.8406:+.4f}")

# =====================================
# 最終予測
# =====================================
print("\n🔮 Training final model and making predictions...")

lgb_model.fit(X_train, y_train)
final_pred = lgb_model.predict(X_test)

# 保存
submission = pd.DataFrame({
    'PassengerId': test_ids,
    'Perished': final_pred
})
submission.to_csv('submission_phase2_kfold_te.csv', index=False)

print(f"  Saved: submission_phase2_kfold_te.csv")
print(f"\n📈 Prediction statistics:")
print(f"  Perished=0: {(final_pred == 0).sum()} ({(final_pred == 0).mean()*100:.1f}%)")
print(f"  Perished=1: {(final_pred == 1).sum()} ({(final_pred == 1).mean()*100:.1f}%)")

print("\n✅ Phase 2-2 Complete!")
print("\n💡 Comparison:")
print("  v6 (leak-inclusive TE): 0.8406")
print(f"  Phase 2-2 (KFold TE): {cv_scores.mean():.4f}")
print(f"  → KFold版は過学習抑制のため、リーク込み版より{'高い' if cv_scores.mean() > 0.8406 else '低い'}精度")
