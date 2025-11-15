"""
Titanic Perished Prediction - Optuna Ultra Shallow Trees
元の16特徴量 + 超浅い木 + 閾値0.5固定

戦略:
1. v2.6の特徴量セット（16個）を使用
2. 超浅い木に絞って探索:
   - num_leaves: 6 ~ 8（さらに浅く）
   - max_depth: 3（固定）
3. 閾値は0.5に固定（過学習防止）
4. 正則化を強化
5. 250 trials で探索
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler

# ====================================================================
# 1. 特徴量エンジニアリング関数（v2.6と同じ）
# ====================================================================

def extract_title(name: str) -> str:
    """Name列からTitleを抽出"""
    m = re.search(r',\s*([^\.]+)\.', str(name))
    if m:
        return m.group(1).strip()
    return "Unknown"

def extract_surname(name: str) -> str:
    """Name列からSurname（姓）を抽出"""
    if pd.isna(name):
        return "Unknown"
    parts = str(name).split(',')
    if len(parts) > 0:
        return parts[0].strip()
    return "Unknown"

def normalize_ticket(ticket: str) -> str:
    """Ticketを正規化（空白除去・大文字化）"""
    return ''.join(str(ticket).split()).upper()

def extract_deck(cabin) -> str:
    """Cabinの先頭文字をDeckとして抽出"""
    if pd.isna(cabin):
        return 'U'
    return str(cabin)[0]

def normalize_cabin(cabin) -> str:
    """Cabinを正規化（複数ある場合は最初のもの）"""
    if pd.isna(cabin):
        return 'Unknown'
    cabins = str(cabin).split()
    if len(cabins) > 0:
        return cabins[0].strip()
    return 'Unknown'

def make_features(train_raw, test_raw):
    """全特徴量を生成する関数（v2.6と同じ）"""

    train = train_raw.copy()
    test = test_raw.copy()

    print("特徴量生成中...")
    for df in [train, test]:
        df['Title'] = df['Name'].apply(extract_title)

    title_mapping = {
        'Capt': 'Officer', 'Col': 'Officer', 'Major': 'Officer',
        'Dr': 'Officer', 'Rev': 'Officer',
        'Don': 'Noble', 'Sir': 'Noble', 'the Countess': 'Noble',
        'Countess': 'Noble', 'Lady': 'Noble', 'Jonkheer': 'Noble',
        'Dona': 'Noble',
        'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'
    }

    for df in [train, test]:
        df['Title'] = df['Title'].replace(title_mapping)

    for df in [train, test]:
        df['Surname'] = df['Name'].apply(extract_surname)

    for df in [train, test]:
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    for df in [train, test]:
        df['TicketNorm'] = df['Ticket'].apply(normalize_ticket)

    full = pd.concat([train, test], sort=False)
    ticket_group_size = full.groupby('TicketNorm')['PassengerId'].transform('size')
    train['TicketGroupSize'] = ticket_group_size[:len(train)].values
    test['TicketGroupSize'] = ticket_group_size[len(train):].values

    for df in [train, test]:
        df['FamilyID'] = df['Surname'] + '_' + df['FamilySize'].astype(str)

    full = pd.concat([train, test], sort=False)
    family_group_size = full.groupby('FamilyID')['PassengerId'].transform('size')
    train['FamilyGroupSize'] = family_group_size[:len(train)].values
    test['FamilyGroupSize'] = family_group_size[len(train):].values

    for df in [train, test]:
        df['HasFamilyMatch'] = (df['FamilyGroupSize'] > df['FamilySize']).astype(int)

    for df in [train, test]:
        df['CabinNorm'] = df['Cabin'].apply(normalize_cabin)

    full = pd.concat([train, test], sort=False)
    cabin_group_size = full.groupby('CabinNorm')['PassengerId'].transform('size')
    train['CabinGroupSize'] = cabin_group_size[:len(train)].values
    test['CabinGroupSize'] = cabin_group_size[len(train):].values

    for df in [train, test]:
        df.loc[df['CabinNorm'] == 'Unknown', 'CabinGroupSize'] = 1

    for df in [train, test]:
        df['FarePerPerson'] = df['Fare'] / df['TicketGroupSize']

    for df in [train, test]:
        df['Deck'] = df['Cabin'].apply(extract_deck)

    age_by_title_pclass = train.groupby(['Title', 'Pclass'])['Age'].median()

    for df in [train, test]:
        mask_missing = df['Age'].isnull()
        for idx in df[mask_missing].index:
            title = df.loc[idx, 'Title']
            pclass = df.loc[idx, 'Pclass']
            if (title, pclass) in age_by_title_pclass.index:
                df.loc[idx, 'Age'] = age_by_title_pclass[(title, pclass)]
            else:
                df.loc[idx, 'Age'] = train[train['Title'] == title]['Age'].median()
        df['Age'] = df['Age'].fillna(train['Age'].median())

    age_bins = [0, 12, 18, 35, 55, 80]
    age_labels = ['Child', 'Teen', 'Young', 'Mid', 'Senior']

    for df in [train, test]:
        df['AgeBin'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, include_lowest=True)
        df['AgeBin'] = df['AgeBin'].astype(str)

    for df in [train, test]:
        df['IsChild'] = (df['Age'] <= 12).astype(int)
        df['IsMother'] = (
            (df['Sex'] == 'female') &
            (df['Age'] >= 18) &
            (df['Parch'] > 0) &
            (df['Title'] == 'Mrs')
        ).astype(int)

    fare_by_pclass = train.groupby('Pclass')['Fare'].median()
    for df in [train, test]:
        for pclass in [1, 2, 3]:
            mask = (df['Fare'].isnull()) & (df['Pclass'] == pclass)
            df.loc[mask, 'Fare'] = fare_by_pclass[pclass]

    most_common_embarked = train['Embarked'].mode()[0]
    for df in [train, test]:
        df['Embarked'] = df['Embarked'].fillna(most_common_embarked)

    for df in [train, test]:
        df['FarePerPerson'] = df['Fare'] / df['TicketGroupSize']

    print("✓ 特徴量生成完了！")
    return train, test

# ====================================================================
# 2. Optuna目的関数（超浅い木 + 閾値0.5固定）
# ====================================================================

def objective(trial, X, y, features, categorical_features, n_splits=5, seed=42):
    """Optunaの目的関数：超浅い木 + 閾値0.5固定"""

    # ハイパーパラメータの探索空間
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 6, 8),  # 6-8に絞る
        'max_depth': 3,  # 3に固定
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 40),  # 増やす
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 40),  # 増やす
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 2.0),  # 範囲拡大
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.9),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.9),
        'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 5.0),  # 正則化強化
        'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 5.0),  # 正則化強化
        'verbosity': -1,
        'seed': seed
    }

    # アーリーストッピングのラウンド数も最適化
    early_stopping_rounds = trial.suggest_int('early_stopping_rounds', 50, 150)

    # 閾値は0.5固定
    threshold = 0.5

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_predictions = np.zeros(len(X))

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        train_data = lgb.Dataset(
            X_train, label=y_train,
            categorical_feature=categorical_features
        )
        val_data = lgb.Dataset(
            X_val, label=y_val,
            categorical_feature=categorical_features,
            reference=train_data
        )

        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            valid_names=['valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0)
            ]
        )

        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        oof_predictions[val_idx] = val_pred

    # 閾値0.5固定でAccuracyを計算
    oof_acc = accuracy_score(y, (oof_predictions > threshold).astype(int))
    return oof_acc

# ====================================================================
# 3. モデル訓練・予測
# ====================================================================

def train_model_with_params(train_df, features, categorical_features, params, target='Perished', n_splits=5, seed=42):
    """指定したパラメータでLightGBMを訓練"""

    print("=" * 60)
    print("最適パラメータでモデル訓練")
    print("=" * 60)

    X = train_df[features].copy()
    y = train_df[target]

    for col in categorical_features:
        X[col] = X[col].astype('category')

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof_predictions = np.zeros(len(X))
    fold_scores = []
    models = []

    threshold = 0.5  # 閾値0.5固定

    print(f"特徴量数: {len(features)}")
    print(f"カテゴリ変数: {categorical_features}")
    print(f"閾値: {threshold:.1f}（固定）")
    print(f"\nCV: {n_splits}-Fold Stratified")
    print("-" * 60)

    # early_stopping_roundsを取り出す
    early_stopping_rounds = params.pop('early_stopping_rounds', 100)

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        train_data = lgb.Dataset(
            X_train, label=y_train,
            categorical_feature=categorical_features
        )
        val_data = lgb.Dataset(
            X_val, label=y_val,
            categorical_feature=categorical_features,
            reference=train_data
        )

        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0)
            ]
        )

        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        oof_predictions[val_idx] = val_pred

        val_acc = accuracy_score(y_val, (val_pred > threshold).astype(int))
        val_auc = roc_auc_score(y_val, val_pred)
        fold_scores.append({'fold': fold, 'accuracy': val_acc, 'auc': val_auc})

        print(f"Fold {fold}: Accuracy={val_acc:.4f}, AUC={val_auc:.4f}, Best Iteration={model.best_iteration}")

        models.append(model)

    oof_acc = accuracy_score(y, (oof_predictions > threshold).astype(int))
    oof_auc = roc_auc_score(y, oof_predictions)

    print("-" * 60)
    print(f"OOF Accuracy: {oof_acc:.4f}")
    print(f"OOF AUC: {oof_auc:.4f}")
    print("=" * 60)

    # Feature Importance
    print("\nFeature Importance (上位15):")
    importance = pd.DataFrame()
    for i, model in enumerate(models):
        fold_importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importance(importance_type='gain')
        })
        importance = pd.concat([importance, fold_importance])

    importance = importance.groupby('feature')['importance'].mean().sort_values(ascending=False)
    print(importance.head(15))

    return models, oof_predictions, fold_scores

def predict_test(models, test_df, features, categorical_features):
    """複数モデルの平均でテストデータを予測"""
    X_test = test_df[features].copy()

    for col in categorical_features:
        X_test[col] = X_test[col].astype('category')

    predictions = np.zeros(len(X_test))

    for model in models:
        pred = model.predict(X_test, num_iteration=model.best_iteration)
        predictions += pred

    predictions /= len(models)

    return predictions

# ====================================================================
# 4. メイン実行
# ====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Titanic - Optuna Ultra Shallow Trees")
    print("元の16特徴量 + num_leaves 6-8 + 閾値0.5固定")
    print("=" * 60)
    print()

    # データ読み込み
    print("データ読み込み...")
    train_raw = pd.read_csv('train.csv')
    test_raw = pd.read_csv('test.csv')

    print(f"Train shape: {train_raw.shape}")
    print(f"Test shape: {test_raw.shape}")
    print()

    # 特徴量生成
    train, test = make_features(train_raw, test_raw)

    # 使用する特徴量（v2.6と同じ16個）
    features = [
        'Fare', 'FarePerPerson',
        'FamilySize', 'TicketGroupSize', 'FamilyGroupSize', 'CabinGroupSize',
        'Sex', 'Pclass', 'Embarked', 'Title', 'Deck', 'AgeBin',
        'IsAlone', 'IsChild', 'IsMother', 'HasFamilyMatch'
    ]

    categorical_features = ['Sex', 'Pclass', 'Embarked', 'Title', 'Deck', 'AgeBin']

    print(f"\n使用特徴量: {len(features)}個（v2.6と同じ）")
    print()

    # データ準備
    X = train[features].copy()
    y = train['Perished']

    for col in categorical_features:
        X[col] = X[col].astype('category')

    # Optuna最適化
    print("=" * 60)
    print("Optunaでハイパーパラメータ最適化開始")
    print("=" * 60)
    print("探索回数: 250 trials")
    print("探索範囲:")
    print("  - num_leaves: 6-8（超浅い木）")
    print("  - max_depth: 3（固定）")
    print("  - min_data_in_leaf: 20-40（増やす）")
    print("  - min_child_samples: 20-40（増やす）")
    print("  - lambda_l1/l2: 0.0-5.0（正則化強化）")
    print("  - threshold: 0.5（固定）")
    print("最適化目標: OOF Accuracy")
    print()

    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42)
    )

    study.optimize(
        lambda trial: objective(trial, X, y, features, categorical_features, n_splits=5, seed=42),
        n_trials=250,
        show_progress_bar=True
    )

    print("\n" + "=" * 60)
    print("最適化完了")
    print("=" * 60)
    print(f"Best OOF Accuracy: {study.best_value:.4f}")
    print(f"v2 (閾値最適化版): 0.8586")
    print(f"v2.6 Baseline: 0.8519")
    diff_v2 = study.best_value - 0.8586
    diff_baseline = study.best_value - 0.8519

    print(f"\nv2比較: {diff_v2:+.4f} ({diff_v2*100:+.2f}pt)")
    print(f"Baseline比較: {diff_baseline:+.4f} ({diff_baseline*100:+.2f}pt)")

    print("\nBest Parameters:")
    for key, value in study.best_params.items():
        if key == 'num_leaves':
            print(f"  {key}: {value} (範囲: 6-8)")
        else:
            print(f"  {key}: {value}")
    print()

    # 最適パラメータでモデル訓練
    best_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'max_depth': 3,
        'verbosity': -1,
        'seed': 42
    }
    best_params.update(study.best_params)

    models, oof_predictions, fold_scores = train_model_with_params(
        train, features, categorical_features, best_params,
        target='Perished', n_splits=5, seed=42
    )

    # テスト予測（閾値0.5固定）
    print("\nテストデータ予測中...")
    test_predictions = predict_test(models, test, features, categorical_features)

    # 提出ファイル作成
    submission = pd.DataFrame({
        'PassengerId': test_raw['PassengerId'],
        'Perished': (test_predictions > 0.5).astype(int)  # 閾値0.5固定
    })

    submission_file = 'submission_optuna_ultra_shallow.csv'
    submission.to_csv(submission_file, index=False)

    print(f"\n提出ファイル作成: {submission_file}")
    print(f"予測Perished分布:")
    print(submission['Perished'].value_counts().sort_index())
    print(f"予測死亡率: {submission['Perished'].mean():.3f}")

    print("\n" + "=" * 60)
    print("最終結果サマリー")
    print("=" * 60)
    print(f"v2.6 ベースライン: 85.19% OOF Accuracy")
    print(f"v2 (閾値最適化): 85.86% OOF Accuracy")
    print(f"v3 (超浅い木): {study.best_value:.2%} OOF Accuracy")

    print("\n改良内容:")
    print("  ✅ num_leavesを6-8に絞る（超浅い木）")
    print("  ✅ max_depthを3に固定")
    print("  ✅ 閾値を0.5に固定（過学習防止）")
    print("  ✅ 正則化を強化（lambda_l1/l2: 0-5）")
    print("  ✅ min_data_in_leaf, min_child_samplesを増やす")

    print("\n" + "=" * 60)
    print("完了！")
    print("=" * 60)
