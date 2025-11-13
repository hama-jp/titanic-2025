#!/usr/bin/env python3
"""
Titanicコンペ風データセット - Perished予測モデル v4
Stacking（2段階アンサンブル）で最高精度を狙う
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
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
# train + test 結合
# =====================================
print("\n🔗 Combining train and test datasets...")
y_train = train['Perished'].copy()
train_drop = train.drop('Perished', axis=1)
full = pd.concat([train_drop, test], axis=0, ignore_index=True)

# =====================================
# 特徴量エンジニアリング（v3と同じ）
# =====================================
print("\n⚙️  Feature engineering v4...")

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

# 4. Name length
full['NameLength'] = full['Name'].apply(len)

# 5. Ticket
full['TicketPrefix'] = full['Ticket'].str.extract('([A-Za-z/\.]+)', expand=False).fillna('NONE')
ticket_counts = full['TicketPrefix'].value_counts()
full['TicketPrefix'] = full['TicketPrefix'].apply(lambda x: x if ticket_counts[x] >= 5 else 'RARE')
full['TicketNumber'] = pd.to_numeric(full['Ticket'].str.extract('(\d+)', expand=False), errors='coerce').fillna(0)
full['TicketFreq'] = full.groupby('Ticket')['Ticket'].transform('count')

# 6. Cabin
full['CabinLetter'] = full['Cabin'].str[0].fillna('X')
full['HasCabin'] = (full['Cabin'].notna()).astype(int)
full['CabinCount'] = full['Cabin'].fillna('').apply(lambda x: len(x.split()))

# 7. 欠損値処理
full['Age'] = full.groupby(['Title', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
full['Age'] = full.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))
full['Fare'] = full.groupby('Pclass')['Fare'].transform(lambda x: x.fillna(x.median()))
full['Embarked'] = full['Embarked'].fillna(full['Embarked'].mode()[0])

# 8. ビニング
full['AgeBin'] = pd.cut(full['Age'], bins=[0, 12, 18, 35, 60, 100],
                        labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior']).astype(str)
full['AgeBin_Fine'] = pd.cut(full['Age'], bins=10, labels=False)
full['FareBin'] = pd.qcut(full['Fare'], q=5, labels=['VeryLow', 'Low', 'Med', 'High', 'VeryHigh'],
                          duplicates='drop').astype(str)
full['FareBin_Fine'] = pd.qcut(full['Fare'], q=10, labels=False, duplicates='drop')

# 9. 交互作用
full['Sex_Pclass'] = full['Sex'] + '_' + full['Pclass'].astype(str)
full['Title_Pclass'] = full['Title'] + '_' + full['Pclass'].astype(str)
full['Age_Pclass'] = full['AgeBin'] + '_' + full['Pclass'].astype(str)
full['Embarked_Pclass'] = full['Embarked'] + '_' + full['Pclass'].astype(str)
full['Title_Sex'] = full['Title'] + '_' + full['Sex']

# 10. 派生数値
full['FarePerPerson'] = full['Fare'] / full['FamilySize']
full['Age_Times_Class'] = full['Age'] * full['Pclass']
full['Fare_Times_Class'] = full['Fare'] * full['Pclass']
full['Age_Sex'] = full['Age'] * full['Sex'].map({'male': 1, 'female': 0})
full['Age_Fare_Interaction'] = full['Age'] * full['Fare']
full['SibSp_Parch_Interaction'] = full['SibSp'] * full['Parch']

# 11. Fare異常値
fare_mean = full['Fare'].mean()
fare_std = full['Fare'].std()
full['FareOutlier'] = (full['Fare'] > fare_mean + 2 * fare_std).astype(int)

print(f"  Total features created: {full.shape[1]}")

# =====================================
# Target Encoding
# =====================================
print("\n🎯 Target encoding...")
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
X_train = X_full[:train_len]
X_test = X_full[train_len:]

print(f"Final feature set: {len(feature_cols)} features")
print(f"X_train shape: {X_train.shape}")

# =====================================
# Stacking Ensemble（2段階）
# =====================================
print("\n🏗️  Building Stacking Ensemble (2-level)...")

# Level 1: Base models
base_models = [
    ('lgb', lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=6,
        num_leaves=31,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )),
    ('xgb', xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )),
    ('gb', GradientBoostingClassifier(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=5,
        min_samples_split=15,
        min_samples_leaf=5,
        subsample=0.85,
        max_features='sqrt',
        random_state=42
    )),
    ('et', ExtraTreesClassifier(
        n_estimators=800,
        max_depth=10,
        min_samples_split=15,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )),
    ('rf', RandomForestClassifier(
        n_estimators=800,
        max_depth=10,
        min_samples_split=15,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    ))
]

# Level 2: Meta-learner (LogisticRegression)
meta_model = LogisticRegression(max_iter=1000, random_state=42)

# StackingClassifier
stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,
    n_jobs=-1,
    verbose=0
)

print("  Training stacking model...")
stacking_model.fit(X_train, y_train)

# =====================================
# CV評価
# =====================================
print("\n📊 Cross-validation (Stacking)...")
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(stacking_model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
print(f"  Stacking CV Accuracy (10-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# =====================================
# 予測
# =====================================
print("\n🔮 Making predictions...")
stacking_pred = stacking_model.predict(X_test)
stacking_pred_proba = stacking_model.predict_proba(X_test)[:, 1]

# =====================================
# 複数seedでStackingを追加実行（さらなる多様性）
# =====================================
print("\n🎲 Additional stacking with multiple seeds...")
seeds = [123, 456, 789]
all_stacking_preds = [stacking_pred_proba]

for seed in seeds:
    print(f"  Seed: {seed}")
    base_models_seed = [
        ('lgb', lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.02, max_depth=6,
                                   num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                                   random_state=seed, verbose=-1)),
        ('xgb', xgb.XGBClassifier(n_estimators=1000, learning_rate=0.02, max_depth=6,
                                  subsample=0.8, colsample_bytree=0.8,
                                  random_state=seed, eval_metric='logloss', verbosity=0)),
        ('gb', GradientBoostingClassifier(n_estimators=800, learning_rate=0.03, max_depth=5,
                                          subsample=0.85, random_state=seed)),
    ]
    meta_seed = LogisticRegression(max_iter=1000, random_state=seed)
    stacking_seed = StackingClassifier(estimators=base_models_seed, final_estimator=meta_seed,
                                       cv=5, n_jobs=-1, verbose=0)
    stacking_seed.fit(X_train, y_train)
    pred_proba_seed = stacking_seed.predict_proba(X_test)[:, 1]
    all_stacking_preds.append(pred_proba_seed)

# 複数stackingの平均
final_proba = np.mean(all_stacking_preds, axis=0)
final_pred = (final_proba >= 0.5).astype(int)

# =====================================
# 結果保存
# =====================================
print("\n💾 Saving results...")
submission = pd.DataFrame({
    'PassengerId': test_ids,
    'Perished': final_pred
})
submission.to_csv('submission_v4_stacking.csv', index=False)
print(f"  Submission saved: submission_v4_stacking.csv")

print(f"\n📈 Prediction statistics:")
print(f"  Perished=0 (Survived): {(final_pred == 0).sum()} ({(final_pred == 0).mean()*100:.1f}%)")
print(f"  Perished=1 (Died): {(final_pred == 1).sum()} ({(final_pred == 1).mean()*100:.1f}%)")

print("\n✅ Done! v4 with Stacking (2-level ensemble)")
print(f"  Base models: {len(base_models)}")
print(f"  Meta-learner: LogisticRegression")
print(f"  Stacking with multiple seeds: {len(seeds) + 1}")
