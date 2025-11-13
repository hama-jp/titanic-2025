#!/usr/bin/env python3
"""
Titanicコンペ風データセット - Perished予測モデル v5: Optuna最適化版

Overview
--------
Optunaを使用したベイズ最適化により、LightGBMとXGBoostのハイパーパラメータを
自動探索し、最適なモデル設定を見つけることで予測精度の向上を目指します。

Features
--------
- **自動ハイパーパラメータ探索**: Optunaによる30試行のベイズ最適化
- **対象モデル**: LightGBM, XGBoost
- **最適化指標**: 5-fold cross-validation accuracy
- **特徴量**: 49個の詳細な特徴量（v3と同じ構成）
- **リーク込み戦略**: train+test結合によるfullデータセット処理

Optimized Hyperparameters
--------------------------
各モデルで以下のハイパーパラメータを最適化:
- n_estimators: 推定器の数 (500-2000)
- learning_rate: 学習率 (0.01-0.1, log scale)
- max_depth: 木の最大深さ (3-10)
- subsample: サンプリング率 (0.6-1.0)
- colsample_bytree: 特徴量サンプリング率 (0.6-1.0)
- その他モデル固有パラメータ (num_leaves, min_child_weight等)

Workflow
--------
1. データ読み込み (train.csv, test.csv)
2. train + test 結合 (leak-inclusive)
3. 特徴量エンジニアリング (49 features)
   - Title/Surname/FamilySize/Ticket/Cabin解析
   - Age/Fare ビニング
   - 交互作用特徴量 (Sex×Pclass, Title×Pclass等)
   - Target encoding (leak-inclusive)
4. Optuna最適化
   - LightGBM: 30 trials
   - XGBoost: 30 trials
5. 最適モデルで学習・予測
6. アンサンブル (LightGBM 55% + XGBoost 45%)
7. 予測結果保存 (submission_v5_optuna.csv)

Usage
-----
    $ python3 titanic_model_v5_optuna.py

    Output: submission_v5_optuna.csv

Requirements
------------
- numpy
- pandas
- scikit-learn
- lightgbm
- xgboost
- optuna

Performance
-----------
- 最適化により、デフォルト設定比で精度向上が期待される
- 実行時間: 約5-10分（30 trials × 2 models）
- v2比較: v2 (0.8462) との比較により最適化効果を評価可能

Notes
-----
- Optunaの最適化はランダムシードに依存するため、実行ごとに結果が変わる可能性あり
- より多くの試行数（n_trials増加）でさらなる改善が見込める
- 最適化されたパラメータは出力ログに表示される

Author
------
Generated for Titanic-style competition with leak-inclusive strategy

Version
-------
v5 - Optuna Hyperparameter Optimization
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
import optuna
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =====================================
# データ読み込み
# =====================================
print("📊 Loading data...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train_len = len(train)
test_ids = test['PassengerId'].copy()

# =====================================
# train + test 結合
# =====================================
y_train = train['Perished'].copy()
train_drop = train.drop('Perished', axis=1)
full = pd.concat([train_drop, test], axis=0, ignore_index=True)

# =====================================
# 特徴量エンジニアリング（v3と同じ）
# =====================================
print("\n⚙️  Feature engineering v5...")

# Title
full['Title'] = full['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
title_mapping = {
    'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
    'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare', 'Mlle': 'Miss',
    'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare', 'Jonkheer': 'Rare',
    'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs', 'Capt': 'Rare', 'Sir': 'Rare'
}
full['Title'] = full['Title'].map(title_mapping).fillna('Rare')

# Surname
full['Surname'] = full['Name'].str.split(',').str[0]
surname_freq = full.groupby('Surname')['Surname'].transform('count')
full['SurnameFreq'] = surname_freq
full['HasFamily'] = (full['SurnameFreq'] > 1).astype(int)

# FamilySize
full['FamilySize'] = full['SibSp'] + full['Parch'] + 1
full['IsAlone'] = (full['FamilySize'] == 1).astype(int)
full['FamilyCategory'] = pd.cut(full['FamilySize'], bins=[0, 1, 4, 20],
                                 labels=['Alone', 'Small', 'Large']).astype(str)
full['NameLength'] = full['Name'].apply(len)

# Ticket
full['TicketPrefix'] = full['Ticket'].str.extract('([A-Za-z/\.]+)', expand=False).fillna('NONE')
ticket_counts = full['TicketPrefix'].value_counts()
full['TicketPrefix'] = full['TicketPrefix'].apply(lambda x: x if ticket_counts[x] >= 5 else 'RARE')
full['TicketNumber'] = pd.to_numeric(full['Ticket'].str.extract('(\d+)', expand=False), errors='coerce').fillna(0)
full['TicketFreq'] = full.groupby('Ticket')['Ticket'].transform('count')

# Cabin
full['CabinLetter'] = full['Cabin'].str[0].fillna('X')
full['HasCabin'] = (full['Cabin'].notna()).astype(int)
full['CabinCount'] = full['Cabin'].fillna('').apply(lambda x: len(x.split()))

# 欠損値処理
full['Age'] = full.groupby(['Title', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
full['Age'] = full.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))
full['Fare'] = full.groupby('Pclass')['Fare'].transform(lambda x: x.fillna(x.median()))
full['Embarked'] = full['Embarked'].fillna(full['Embarked'].mode()[0])

# ビニング
full['AgeBin'] = pd.cut(full['Age'], bins=[0, 12, 18, 35, 60, 100],
                        labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior']).astype(str)
full['AgeBin_Fine'] = pd.cut(full['Age'], bins=10, labels=False)
full['FareBin'] = pd.qcut(full['Fare'], q=5, labels=['VeryLow', 'Low', 'Med', 'High', 'VeryHigh'],
                          duplicates='drop').astype(str)
full['FareBin_Fine'] = pd.qcut(full['Fare'], q=10, labels=False, duplicates='drop')

# 交互作用
full['Sex_Pclass'] = full['Sex'] + '_' + full['Pclass'].astype(str)
full['Title_Pclass'] = full['Title'] + '_' + full['Pclass'].astype(str)
full['Age_Pclass'] = full['AgeBin'] + '_' + full['Pclass'].astype(str)
full['Embarked_Pclass'] = full['Embarked'] + '_' + full['Pclass'].astype(str)
full['Title_Sex'] = full['Title'] + '_' + full['Sex']

# 派生数値
full['FarePerPerson'] = full['Fare'] / full['FamilySize']
full['Age_Times_Class'] = full['Age'] * full['Pclass']
full['Fare_Times_Class'] = full['Fare'] * full['Pclass']
full['Age_Sex'] = full['Age'] * full['Sex'].map({'male': 1, 'female': 0})
full['Age_Fare_Interaction'] = full['Age'] * full['Fare']
full['SibSp_Parch_Interaction'] = full['SibSp'] * full['Parch']

# Fare異常値
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

# 特徴量選択
drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
feature_cols = [col for col in full.columns if col not in drop_cols]

X_full = full[feature_cols]
X_train = X_full[:train_len]
X_test = X_full[train_len:]

print(f"Final feature set: {len(feature_cols)} features")

# =====================================
# Optuna: LightGBM最適化
# =====================================
print("\n🔍 Optuna: Optimizing LightGBM hyperparameters...")

def objective_lgb(trial):
    """
    LightGBM用のOptuna目的関数

    ベイズ最適化により、LightGBMのハイパーパラメータを探索し、
    5-fold cross-validationのaccuracyを最大化する。

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optunaのトライアルオブジェクト。パラメータ提案に使用。

    Returns
    -------
    float
        5-fold cross-validationの平均accuracy

    Optimized Parameters
    --------------------
    - n_estimators: 推定器の数 [500, 2000] (step=100)
    - learning_rate: 学習率 [0.01, 0.1] (log scale)
    - max_depth: 木の最大深さ [3, 10]
    - num_leaves: 葉ノード数 [15, 63]
    - min_child_samples: 子ノードの最小サンプル数 [5, 30]
    - subsample: サンプリング率 [0.6, 1.0]
    - colsample_bytree: 特徴量サンプリング率 [0.6, 1.0]

    Notes
    -----
    - 5-fold StratifiedKFoldで評価
    - random_state=42で再現性を確保
    - verbose=-1で警告抑制
    """
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 15, 63),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42,
        'verbose': -1
    }

    model = lgb.LGBMClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    return scores.mean()

study_lgb = optuna.create_study(direction='maximize', study_name='lgb_optim')
study_lgb.optimize(objective_lgb, n_trials=30, show_progress_bar=False)

print(f"  Best LightGBM CV Accuracy: {study_lgb.best_value:.4f}")
print(f"  Best params: {study_lgb.best_params}")

# =====================================
# Optuna: XGBoost最適化
# =====================================
print("\n🔍 Optuna: Optimizing XGBoost hyperparameters...")

def objective_xgb(trial):
    """
    XGBoost用のOptuna目的関数

    ベイズ最適化により、XGBoostのハイパーパラメータを探索し、
    5-fold cross-validationのaccuracyを最大化する。

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optunaのトライアルオブジェクト。パラメータ提案に使用。

    Returns
    -------
    float
        5-fold cross-validationの平均accuracy

    Optimized Parameters
    --------------------
    - n_estimators: 推定器の数 [500, 2000] (step=100)
    - learning_rate: 学習率 [0.01, 0.1] (log scale)
    - max_depth: 木の最大深さ [3, 10]
    - min_child_weight: 子ノードの最小重み [1, 10]
    - subsample: サンプリング率 [0.6, 1.0]
    - colsample_bytree: 特徴量サンプリング率 [0.6, 1.0]

    Notes
    -----
    - 5-fold StratifiedKFoldで評価
    - random_state=42で再現性を確保
    - eval_metric='logloss'で損失関数を指定
    - verbosity=0で警告抑制
    """
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42,
        'eval_metric': 'logloss',
        'verbosity': 0
    }

    model = xgb.XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    return scores.mean()

study_xgb = optuna.create_study(direction='maximize', study_name='xgb_optim')
study_xgb.optimize(objective_xgb, n_trials=30, show_progress_bar=False)

print(f"  Best XGBoost CV Accuracy: {study_xgb.best_value:.4f}")
print(f"  Best params: {study_xgb.best_params}")

# =====================================
# 最適化されたモデルで学習・予測
# =====================================
print("\n🚀 Training optimized models...")

# Best LightGBM
best_lgb = lgb.LGBMClassifier(**study_lgb.best_params)
best_lgb.fit(X_train, y_train)
lgb_pred = best_lgb.predict_proba(X_test)[:, 1]

# Best XGBoost
best_xgb = xgb.XGBClassifier(**study_xgb.best_params)
best_xgb.fit(X_train, y_train)
xgb_pred = best_xgb.predict_proba(X_test)[:, 1]

# アンサンブル
ensemble_proba = (0.55 * lgb_pred + 0.45 * xgb_pred)
final_pred = (ensemble_proba >= 0.5).astype(int)

# =====================================
# 結果保存
# =====================================
print("\n💾 Saving results...")
submission = pd.DataFrame({
    'PassengerId': test_ids,
    'Perished': final_pred
})
submission.to_csv('submission_v5_optuna.csv', index=False)
print(f"  Submission saved: submission_v5_optuna.csv")

print(f"\n📈 Prediction statistics:")
print(f"  Perished=0 (Survived): {(final_pred == 0).sum()} ({(final_pred == 0).mean()*100:.1f}%)")
print(f"  Perished=1 (Died): {(final_pred == 1).sum()} ({(final_pred == 1).mean()*100:.1f}%)")

print("\n✅ Done! v5 with Optuna hyperparameter optimization")
print(f"  LightGBM best CV: {study_lgb.best_value:.4f}")
print(f"  XGBoost best CV: {study_xgb.best_value:.4f}")
