#!/usr/bin/env python3
"""
Titanicコンペ風データセット - Perished予測モデル v7 Ensemble版

Phase 2-1（家族グループ特徴量）+ Phase 2-3（CatBoost）の組み合わせ:
- 63特徴量（基本49 + 家族グループ14）
- LightGBM + CatBoost のアンサンブル
- 閾値最適化

期待CV精度: 0.99+
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("Titanic Model v7 Ensemble: LightGBM + CatBoost + Family Features")
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

# =====================================
# Phase 2-1: 家族グループ特徴量
# =====================================
print("\n🆕 Phase 2-1: Family group features...")

full['Target_tmp'] = np.nan
full.loc[:train_len-1, 'Target_tmp'] = y_train.values

# 姓ベースの家族グループ生存率
surname_survival = full.groupby('Surname')['Target_tmp'].agg(['mean', 'count', 'std'])
full['FamilyGroupSurvivalRate'] = full['Surname'].map(surname_survival['mean']).fillna(0.5)
full['FamilyGroupSize'] = full['Surname'].map(surname_survival['count'])
full['FamilyGroupSurvivalStd'] = full['Surname'].map(surname_survival['std']).fillna(0)

# チケットベースのグループ生存率
ticket_survival = full.groupby('Ticket')['Target_tmp'].agg(['mean', 'count'])
full['TicketGroupSurvivalRate'] = full['Ticket'].map(ticket_survival['mean']).fillna(0.5)
full['TicketGroupSize'] = full['Ticket'].map(ticket_survival['count'])

# 家族グループの平均年齢・運賃
surname_age = full.groupby('Surname')['Age'].mean()
surname_fare = full.groupby('Surname')['Fare'].mean()
full['FamilyGroupMeanAge'] = full['Surname'].map(surname_age)
full['FamilyGroupMeanFare'] = full['Surname'].map(surname_fare)

# チケットグループの平均年齢・運賃
ticket_age = full.groupby('Ticket')['Age'].mean()
ticket_fare = full.groupby('Ticket')['Fare'].mean()
full['TicketGroupMeanAge'] = full['Ticket'].map(ticket_age)
full['TicketGroupMeanFare'] = full['Ticket'].map(ticket_fare)

# 家族内での年齢順位
full['FamilyAgeRank'] = full.groupby('Surname')['Age'].rank(method='dense')

# 家族グループでの性別構成
surname_sex_count = full.groupby('Surname')['Sex'].apply(lambda x: (x == 'female').sum())
full['FamilyFemaleCount'] = full['Surname'].map(surname_sex_count)
surname_male_count = full.groupby('Surname')['Sex'].apply(lambda x: (x == 'male').sum())
full['FamilyMaleCount'] = full['Surname'].map(surname_male_count)

# 年齢と家族平均年齢の差
full['AgeDiffFromFamilyMean'] = full['Age'] - full['FamilyGroupMeanAge']

# 運賃と家族平均運賃の差
full['FareDiffFromFamilyMean'] = full['Fare'] - full['FamilyGroupMeanFare']

# Target Encoding
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
# データ準備: LightGBM用とCatBoost用
# =====================================
print("\n🔧 Preparing data for models...")

# CatBoost用のカテゴリ特徴量リスト
cat_features_for_catboost = [
    'Sex', 'Embarked', 'Title', 'CabinLetter', 'TicketPrefix',
    'AgeBin', 'FareBin', 'Sex_Pclass', 'Title_Pclass', 'Age_Pclass',
    'FamilyCategory', 'Embarked_Pclass', 'Title_Sex', 'Surname'
]

# LightGBM用: Label Encoding
full_lgb = full.copy()
for col in cat_features_for_catboost:
    le = LabelEncoder()
    full_lgb[col] = le.fit_transform(full_lgb[col].astype(str))

drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
feature_cols = [col for col in full.columns if col not in drop_cols]

X_train_lgb = full_lgb[feature_cols][:train_len]
X_test_lgb = full_lgb[feature_cols][train_len:]

X_train_cat = full[feature_cols][:train_len]
X_test_cat = full[feature_cols][train_len:]

print(f"  Total features: {len(feature_cols)}")
print(f"  Categorical features for CatBoost: {len(cat_features_for_catboost)}")

# =====================================
# モデル訓練: LightGBM
# =====================================
print("\n📊 Training LightGBM...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# OOF predictions for threshold optimization
oof_proba_lgb = np.zeros(len(X_train_lgb))
test_proba_lgb = np.zeros(len(X_test_lgb))

for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train_lgb, y_train)):
    X_tr, X_val = X_train_lgb.iloc[train_idx], X_train_lgb.iloc[val_idx]
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

    oof_proba_lgb[val_idx] = lgb_model.predict_proba(X_val)[:, 1]
    test_proba_lgb += lgb_model.predict_proba(X_test_lgb)[:, 1] / cv.get_n_splits()

lgb_cv_acc = accuracy_score(y_train, (oof_proba_lgb >= 0.5).astype(int))
print(f"  LightGBM OOF Accuracy (0.5): {lgb_cv_acc:.4f}")

# =====================================
# モデル訓練: CatBoost
# =====================================
print("\n📊 Training CatBoost...")

oof_proba_cat = np.zeros(len(X_train_cat))
test_proba_cat = np.zeros(len(X_test_cat))

for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train_cat, y_train)):
    X_tr, X_val = X_train_cat.iloc[train_idx], X_train_cat.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    cat_model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=6,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=0,
        cat_features=cat_features_for_catboost
    )
    cat_model.fit(X_tr, y_tr, cat_features=cat_features_for_catboost)

    oof_proba_cat[val_idx] = cat_model.predict_proba(X_val)[:, 1]
    test_proba_cat += cat_model.predict_proba(X_test_cat)[:, 1] / cv.get_n_splits()

cat_cv_acc = accuracy_score(y_train, (oof_proba_cat >= 0.5).astype(int))
print(f"  CatBoost OOF Accuracy (0.5): {cat_cv_acc:.4f}")

# =====================================
# アンサンブル + 閾値最適化
# =====================================
print("\n🎭 Ensemble optimization...")

# ウェイト最適化（簡易版: 0.6 LGB + 0.4 CAT）
ensemble_proba = 0.6 * oof_proba_lgb + 0.4 * oof_proba_cat
test_ensemble_proba = 0.6 * test_proba_lgb + 0.4 * test_proba_cat

# 閾値最適化
thresholds = np.arange(0.3, 0.71, 0.01)
results = []

for threshold in thresholds:
    oof_pred = (ensemble_proba >= threshold).astype(int)
    acc = accuracy_score(y_train, oof_pred)
    results.append({'threshold': threshold, 'accuracy': acc})

results_df = pd.DataFrame(results)
best_result = results_df.loc[results_df['accuracy'].idxmax()]
best_threshold = best_result['threshold']

baseline_acc = results_df.iloc[(results_df['threshold'] - 0.5).abs().argsort()[:1]]['accuracy'].values[0]

print(f"\n  Ensemble weights: 0.6 LightGBM + 0.4 CatBoost")
print(f"  Baseline (0.5): {baseline_acc:.4f}")
print(f"  Optimized ({best_threshold:.2f}): {best_result['accuracy']:.4f}")
print(f"  Improvement: {best_result['accuracy'] - baseline_acc:+.4f}")

# =====================================
# 最終予測
# =====================================
print("\n🔮 Generating final predictions...")

final_pred = (test_ensemble_proba >= best_threshold).astype(int)

# 保存
submission = pd.DataFrame({
    'PassengerId': test_ids,
    'Perished': final_pred
})
submission.to_csv('submission_v7_ensemble.csv', index=False)

print(f"\n✅ Submission saved: submission_v7_ensemble.csv")
print(f"\n📈 Prediction statistics:")
print(f"  Perished=0: {(final_pred == 0).sum()} ({(final_pred == 0).mean()*100:.1f}%)")
print(f"  Perished=1: {(final_pred == 1).sum()} ({(final_pred == 1).mean()*100:.1f}%)")

print("\n" + "=" * 70)
print("v7 Ensemble Model Summary:")
print("  ✅ Family Group Features (Phase 2-1)")
print("  ✅ LightGBM + CatBoost Ensemble (Phase 2-3)")
print("  ✅ Threshold Optimization (Phase 1-3)")
print(f"  📊 Final OOF Accuracy: {best_result['accuracy']:.4f}")
print(f"  🎯 Optimized Threshold: {best_threshold:.2f}")
print("  🏆 Best of both Phase 2 strategies!")
print("=" * 70)

# 比較
print("\n📊 Model Comparison:")
print(f"  v7 (LightGBM only):        0.9921")
print(f"  v7 Ensemble (LGB+CatBoost): {best_result['accuracy']:.4f}")
print(f"  Difference: {best_result['accuracy'] - 0.9921:+.4f}")
