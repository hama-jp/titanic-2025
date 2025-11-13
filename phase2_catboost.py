#!/usr/bin/env python3
"""
Phase 2-3: CatBoost導入

CatBoostはカテゴリ変数の扱いに優れたGradient Boostingアルゴリズム:
- カテゴリ変数を直接扱える（Label Encoding不要）
- 内蔵の順序付きTarget Encoding機能
- 過学習に強い

LightGBMとCatBoostのアンサンブルで精度向上を狙います。
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("Phase 2-3: CatBoost導入")
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

# Target Encoding (リーク込み版、LightGBM用)
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

# =====================================
# 🆕 Phase 2-3: CatBoost用のデータ準備
# =====================================
print("\n🆕 Phase 2-3: Preparing data for CatBoost...")

# CatBoostはカテゴリ変数を直接扱えるので、Label Encodingしないバージョンも保持
cat_features_for_catboost = [
    'Sex', 'Embarked', 'Title', 'CabinLetter', 'TicketPrefix',
    'AgeBin', 'FareBin', 'Sex_Pclass', 'Title_Pclass', 'Age_Pclass',
    'FamilyCategory', 'Embarked_Pclass', 'Title_Sex', 'Surname'
]

# 数値変換が必要な列のみLabel Encoding
label_cols_for_lgb = cat_features_for_catboost.copy()
full_lgb = full.copy()
for col in label_cols_for_lgb:
    le = LabelEncoder()
    full_lgb[col] = le.fit_transform(full_lgb[col].astype(str))

# 特徴量選択
drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
feature_cols = [col for col in full_lgb.columns if col not in drop_cols]

X_train_lgb = full_lgb[feature_cols][:train_len]
X_test_lgb = full_lgb[feature_cols][train_len:]

X_train_cat = full[feature_cols][:train_len]
X_test_cat = full[feature_cols][train_len:]

print(f"  Total features: {len(feature_cols)}")
print(f"  Categorical features for CatBoost: {len(cat_features_for_catboost)}")

# =====================================
# LightGBM
# =====================================
print("\n📊 Training LightGBM...")

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
lgb_cv_scores = cross_val_score(lgb_model, X_train_lgb, y_train, cv=cv, scoring='accuracy')
print(f"  LightGBM CV: {lgb_cv_scores.mean():.4f} (+/- {lgb_cv_scores.std():.4f})")

lgb_model.fit(X_train_lgb, y_train)
lgb_pred_proba = lgb_model.predict_proba(X_test_lgb)[:, 1]

# =====================================
# CatBoost
# =====================================
print("\n📊 Training CatBoost...")

catboost_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.02,
    depth=6,
    l2_leaf_reg=3,
    random_seed=42,
    verbose=0,
    cat_features=cat_features_for_catboost
)

catboost_cv_scores = cross_val_score(catboost_model, X_train_cat, y_train, cv=cv, scoring='accuracy')
print(f"  CatBoost CV: {catboost_cv_scores.mean():.4f} (+/- {catboost_cv_scores.std():.4f})")

catboost_model.fit(X_train_cat, y_train, cat_features=cat_features_for_catboost)
catboost_pred_proba = catboost_model.predict_proba(X_test_cat)[:, 1]

# =====================================
# アンサンブル
# =====================================
print("\n🎭 Ensemble predictions...")

# LightGBM 60% + CatBoost 40%
ensemble_proba = 0.6 * lgb_pred_proba + 0.4 * catboost_pred_proba
ensemble_pred = (ensemble_proba >= 0.5).astype(int)

# 保存
submission = pd.DataFrame({
    'PassengerId': test_ids,
    'Perished': ensemble_pred
})
submission.to_csv('submission_phase2_catboost.csv', index=False)

print(f"\n  Saved: submission_phase2_catboost.csv")
print(f"\n📈 Prediction statistics:")
print(f"  Perished=0: {(ensemble_pred == 0).sum()} ({(ensemble_pred == 0).mean()*100:.1f}%)")
print(f"  Perished=1: {(ensemble_pred == 1).sum()} ({(ensemble_pred == 1).mean()*100:.1f}%)")

# アンサンブルCV推定
ensemble_cv = (0.6 * lgb_cv_scores.mean() + 0.4 * catboost_cv_scores.mean())
print(f"\n📊 Estimated Ensemble CV: {ensemble_cv:.4f}")
print(f"  Baseline (v6): 0.8406")
print(f"  Improvement: {ensemble_cv - 0.8406:+.4f}")

print("\n✅ Phase 2-3 Complete!")
