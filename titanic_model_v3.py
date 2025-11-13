#!/usr/bin/env python3
"""
Titanicコンペ風データセット - Perished予測モデル v3
LightGBM/XGBoost + Stacking + 複数seed平均で最高スコアを狙う
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
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
# 特徴量エンジニアリング v3（さらに強化）
# =====================================
print("\n⚙️  Feature engineering v3 (enhanced++)...")

# 1. Title抽出
full['Title'] = full['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
title_mapping = {
    'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
    'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare', 'Mlle': 'Miss',
    'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare', 'Jonkheer': 'Rare',
    'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs', 'Capt': 'Rare', 'Sir': 'Rare'
}
full['Title'] = full['Title'].map(title_mapping).fillna('Rare')

# 2. 姓（Surname）の抽出と頻度
full['Surname'] = full['Name'].str.split(',').str[0]
surname_freq = full.groupby('Surname')['Surname'].transform('count')
full['SurnameFreq'] = surname_freq
# 姓が複数人いる = 家族グループ
full['HasFamily'] = (full['SurnameFreq'] > 1).astype(int)

# 3. FamilySize関連
full['FamilySize'] = full['SibSp'] + full['Parch'] + 1
full['IsAlone'] = (full['FamilySize'] == 1).astype(int)
full['FamilyCategory'] = pd.cut(full['FamilySize'], bins=[0, 1, 4, 20],
                                 labels=['Alone', 'Small', 'Large']).astype(str)

# 4. Name length
full['NameLength'] = full['Name'].apply(len)

# 5. Ticket関連
full['TicketPrefix'] = full['Ticket'].str.extract('([A-Za-z/\.]+)', expand=False)
full['TicketPrefix'] = full['TicketPrefix'].fillna('NONE')
ticket_counts = full['TicketPrefix'].value_counts()
full['TicketPrefix'] = full['TicketPrefix'].apply(
    lambda x: x if ticket_counts[x] >= 5 else 'RARE'
)
full['TicketNumber'] = full['Ticket'].str.extract('(\d+)', expand=False)
full['TicketNumber'] = pd.to_numeric(full['TicketNumber'], errors='coerce').fillna(0)
full['TicketFreq'] = full.groupby('Ticket')['Ticket'].transform('count')

# 6. Cabin関連
full['CabinLetter'] = full['Cabin'].str[0]
full['CabinLetter'] = full['CabinLetter'].fillna('X')
full['HasCabin'] = (full['Cabin'].notna()).astype(int)
full['CabinCount'] = full['Cabin'].fillna('').apply(lambda x: len(x.split()))

# 7. 欠損値処理（fullベース）
full['Age'] = full.groupby(['Title', 'Pclass'])['Age'].transform(
    lambda x: x.fillna(x.median())
)
full['Age'] = full.groupby('Title')['Age'].transform(
    lambda x: x.fillna(x.median())
)
full['Fare'] = full.groupby('Pclass')['Fare'].transform(
    lambda x: x.fillna(x.median())
)
full['Embarked'] = full['Embarked'].fillna(full['Embarked'].mode()[0])

# 8. ビニング
full['AgeBin'] = pd.cut(full['Age'], bins=[0, 12, 18, 35, 60, 100],
                        labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior']).astype(str)
full['AgeBin_Fine'] = pd.cut(full['Age'], bins=10, labels=False)
full['FareBin'] = pd.qcut(full['Fare'], q=5, labels=['VeryLow', 'Low', 'Med', 'High', 'VeryHigh'],
                          duplicates='drop').astype(str)
full['FareBin_Fine'] = pd.qcut(full['Fare'], q=10, labels=False, duplicates='drop')

# 9. 交互作用特徴量（v3で追加）
full['Sex_Pclass'] = full['Sex'] + '_' + full['Pclass'].astype(str)
full['Title_Pclass'] = full['Title'] + '_' + full['Pclass'].astype(str)
full['Age_Pclass'] = full['AgeBin'] + '_' + full['Pclass'].astype(str)
# NEW: Embarked × Pclass
full['Embarked_Pclass'] = full['Embarked'] + '_' + full['Pclass'].astype(str)
# NEW: Title × Sex
full['Title_Sex'] = full['Title'] + '_' + full['Sex']

# 10. 派生数値特徴量
full['FarePerPerson'] = full['Fare'] / full['FamilySize']
full['Age_Times_Class'] = full['Age'] * full['Pclass']
full['Fare_Times_Class'] = full['Fare'] * full['Pclass']
full['Age_Sex'] = full['Age'] * full['Sex'].map({'male': 1, 'female': 0})
# NEW: より複雑な相互作用
full['Age_Fare_Interaction'] = full['Age'] * full['Fare']
full['SibSp_Parch_Interaction'] = full['SibSp'] * full['Parch']

# 11. Fare異常値フラグ
fare_mean = full['Fare'].mean()
fare_std = full['Fare'].std()
full['FareOutlier'] = (full['Fare'] > fare_mean + 2 * fare_std).astype(int)

print(f"  Total features created: {full.shape[1]}")

# =====================================
# Target Encoding v3
# =====================================
print("\n🎯 Target encoding v3...")
full['Target_tmp'] = np.nan
full.loc[:train_len-1, 'Target_tmp'] = y_train.values

cat_features = [
    'Title', 'Embarked', 'CabinLetter', 'TicketPrefix',
    'AgeBin', 'FareBin', 'Sex_Pclass', 'Title_Pclass',
    'Age_Pclass', 'FamilyCategory', 'Embarked_Pclass', 'Title_Sex'
]

for col in cat_features:
    target_mean = full.groupby(col)['Target_tmp'].mean()
    full[f'{col}_TE'] = full[col].map(target_mean)
    full[f'{col}_TE'] = full[f'{col}_TE'].fillna(y_train.mean())

full.drop('Target_tmp', axis=1, inplace=True)

# =====================================
# Label Encoding
# =====================================
print("\n🔤 Label encoding...")
label_cols = [
    'Sex', 'Embarked', 'Title', 'CabinLetter', 'TicketPrefix',
    'AgeBin', 'FareBin', 'Sex_Pclass', 'Title_Pclass', 'Age_Pclass',
    'FamilyCategory', 'Embarked_Pclass', 'Title_Sex', 'Surname'
]
for col in label_cols:
    le = LabelEncoder()
    full[col] = le.fit_transform(full[col].astype(str))

# =====================================
# 特徴量選択
# =====================================
print("\n📋 Selecting features...")
drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
feature_cols = [col for col in full.columns if col not in drop_cols]

X_full = full[feature_cols]
print(f"Final feature set: {len(feature_cols)} features")

X_train = X_full[:train_len]
X_test = X_full[train_len:]

print(f"\n✅ X_train shape: {X_train.shape}")
print(f"✅ X_test shape: {X_test.shape}")

# =====================================
# 複数seedで学習（アンサンブル用）
# =====================================
print("\n🎲 Training multiple models with different seeds...")

seeds = [42, 123, 456, 789, 2025]
all_predictions = []

for seed in seeds:
    print(f"\n  Seed: {seed}")

    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=6,
        num_leaves=31,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    lgb_pred = lgb_model.predict_proba(X_test)[:, 1]

    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        eval_metric='logloss',
        verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict_proba(X_test)[:, 1]

    # GradientBoosting
    gb_model = GradientBoostingClassifier(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=5,
        min_samples_split=15,
        min_samples_leaf=5,
        subsample=0.85,
        max_features='sqrt',
        random_state=seed
    )
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict_proba(X_test)[:, 1]

    # ExtraTrees
    et_model = ExtraTreesClassifier(
        n_estimators=800,
        max_depth=10,
        min_samples_split=15,
        min_samples_leaf=5,
        random_state=seed,
        n_jobs=-1
    )
    et_model.fit(X_train, y_train)
    et_pred = et_model.predict_proba(X_test)[:, 1]

    # RandomForest
    rf_model = RandomForestClassifier(
        n_estimators=800,
        max_depth=10,
        min_samples_split=15,
        min_samples_leaf=5,
        random_state=seed,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict_proba(X_test)[:, 1]

    # 各seedでの重み付きアンサンブル
    seed_ensemble = (0.3 * lgb_pred + 0.3 * xgb_pred +
                     0.2 * gb_pred + 0.1 * et_pred + 0.1 * rf_pred)
    all_predictions.append(seed_ensemble)
    print(f"    Ensemble for seed {seed} completed")

# =====================================
# 複数seedの平均（最終予測）
# =====================================
print("\n🎭 Averaging predictions across seeds...")
final_proba = np.mean(all_predictions, axis=0)
final_pred = (final_proba >= 0.5).astype(int)

# =====================================
# CV評価（代表seed=42で）
# =====================================
print("\n📊 Cross-validation (LightGBM, seed=42)...")
lgb_cv = lgb.LGBMClassifier(
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
cv_scores = cross_val_score(lgb_cv, X_train, y_train, cv=cv, scoring='accuracy')
print(f"  LightGBM CV Accuracy (10-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

print("\n📊 Cross-validation (XGBoost, seed=42)...")
xgb_cv = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.02,
    max_depth=6,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',
    verbosity=0
)
cv_scores_xgb = cross_val_score(xgb_cv, X_train, y_train, cv=cv, scoring='accuracy')
print(f"  XGBoost CV Accuracy (10-fold): {cv_scores_xgb.mean():.4f} (+/- {cv_scores_xgb.std():.4f})")

# =====================================
# 結果保存
# =====================================
print("\n💾 Saving results...")
submission = pd.DataFrame({
    'PassengerId': test_ids,
    'Perished': final_pred
})
submission.to_csv('submission_v3.csv', index=False)
print(f"  Submission saved: submission_v3.csv")

print(f"\n📈 Prediction statistics:")
print(f"  Perished=0 (Survived): {(final_pred == 0).sum()} ({(final_pred == 0).mean()*100:.1f}%)")
print(f"  Perished=1 (Died): {(final_pred == 1).sum()} ({(final_pred == 1).mean()*100:.1f}%)")

print("\n✅ Done! v3 with LightGBM/XGBoost + multi-seed ensemble")
print(f"  Seeds used: {seeds}")
print(f"  Models per seed: 5 (LightGBM, XGBoost, GB, ET, RF)")
print(f"  Total models: {len(seeds) * 5} = {len(seeds)*5}")
