"""
Titanic Perished Prediction v2.1
HasFamilyMatchを削除（洗練版）

v2からの変更:
- HasFamilyMatch を削除（重要度5.10, 最下位）
- 特徴量数: 19個 → 18個
- ノイズ削減による改善を期待

理由:
- HasFamilyMatch = (FamilyGroupSize > FamilySize) は情報が重複
- GBDTは自分でこの関係を学習できる
- 全バージョンで重要度が最下位クラス
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
import lightgbm as lgb

# ====================================================================
# 特徴量エンジニアリング（v2と同じ、HasFamilyMatchのみ削除）
# ====================================================================

def extract_title(name: str) -> str:
    m = re.search(r',\s*([^\.]+)\.', str(name))
    if m:
        return m.group(1).strip()
    return "Unknown"

def extract_surname(name: str) -> str:
    if pd.isna(name):
        return "Unknown"
    parts = str(name).split(',')
    if len(parts) > 0:
        return parts[0].strip()
    return "Unknown"

def normalize_ticket(ticket: str) -> str:
    return ''.join(str(ticket).split()).upper()

def extract_deck(cabin) -> str:
    if pd.isna(cabin):
        return 'U'
    return str(cabin)[0]

def normalize_cabin(cabin) -> str:
    if pd.isna(cabin):
        return 'Unknown'
    cabins = str(cabin).split()
    if len(cabins) > 0:
        return cabins[0].strip()
    return 'Unknown'

def make_features(train_raw, test_raw):
    """v2.1: v2からHasFamilyMatchを削除"""

    train = train_raw.copy()
    test = test_raw.copy()

    print("=" * 60)
    print("特徴量生成（v2.1: HasFamilyMatch削除版）")
    print("=" * 60)

    # Title
    for df in [train, test]:
        df['Title'] = df['Name'].apply(extract_title)

    title_mapping = {
        'Capt': 'Officer', 'Col': 'Officer', 'Major': 'Officer',
        'Dr': 'Officer', 'Rev': 'Officer',
        'Don': 'Noble', 'Sir': 'Noble', 'the Countess': 'Noble',
        'Countess': 'Noble', 'Lady': 'Noble', 'Jonkheer': 'Noble', 'Dona': 'Noble',
        'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'
    }

    for df in [train, test]:
        df['Title'] = df['Title'].replace(title_mapping)

    # Surname, FamilySize
    for df in [train, test]:
        df['Surname'] = df['Name'].apply(extract_surname)
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # train+testでグループサイズ計算
    for df in [train, test]:
        df['TicketNorm'] = df['Ticket'].apply(normalize_ticket)
        df['FamilyID'] = df['Surname'] + '_' + df['FamilySize'].astype(str)
        df['CabinNorm'] = df['Cabin'].apply(normalize_cabin)

    full = pd.concat([train, test], sort=False)

    # TicketGroupSize
    ticket_group_size = full.groupby('TicketNorm')['PassengerId'].transform('size')
    train['TicketGroupSize'] = ticket_group_size[:len(train)].values
    test['TicketGroupSize'] = ticket_group_size[len(train):].values

    # FamilyGroupSize
    family_group_size = full.groupby('FamilyID')['PassengerId'].transform('size')
    train['FamilyGroupSize'] = family_group_size[:len(train)].values
    test['FamilyGroupSize'] = family_group_size[len(train):].values

    # ✗ HasFamilyMatch は削除（v2.1の改善点）

    # CabinGroupSize
    cabin_group_size = full.groupby('CabinNorm')['PassengerId'].transform('size')
    train['CabinGroupSize'] = cabin_group_size[:len(train)].values
    test['CabinGroupSize'] = cabin_group_size[len(train):].values

    for df in [train, test]:
        df.loc[df['CabinNorm'] == 'Unknown', 'CabinGroupSize'] = 1

    # FarePerPerson
    for df in [train, test]:
        df['FarePerPerson'] = df['Fare'] / df['TicketGroupSize']
        df['Deck'] = df['Cabin'].apply(extract_deck)

    # Age欠損補完
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

    # AgeBin
    age_bins = [0, 12, 18, 35, 55, 80]
    age_labels = ['Child', 'Teen', 'Young', 'Mid', 'Senior']

    for df in [train, test]:
        df['AgeBin'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, include_lowest=True)
        df['AgeBin'] = df['AgeBin'].astype(str)

    # IsChild, IsMother
    for df in [train, test]:
        df['IsChild'] = (df['Age'] <= 12).astype(int)
        df['IsMother'] = (
            (df['Sex'] == 'female') &
            (df['Age'] >= 18) &
            (df['Parch'] > 0) &
            (df['Title'] == 'Mrs')
        ).astype(int)

    # Fare/Embarked欠損補完
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

    print("\n✓ 特徴量生成完了（v2.1）")
    print(f"  v2からの変更: HasFamilyMatch を削除")
    print(f"  特徴量数: 19個 → 18個")

    return train, test

# ====================================================================
# モデル訓練・予測（v2と同じ）
# ====================================================================

def train_model(train_df, features, target='Perished', n_splits=5, seed=42):
    """Stratified K-FoldでLightGBMを訓練"""

    print("\n" + "=" * 60)
    print("モデル訓練開始")
    print("=" * 60)

    X = train_df[features].copy()
    y = train_df[target]

    categorical_features = [
        'Sex', 'Pclass', 'Embarked', 'Title', 'Deck', 'AgeBin'
    ]
    categorical_features = [f for f in categorical_features if f in features]

    for col in categorical_features:
        X[col] = X[col].astype('category')

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': 0.03,
        'num_leaves': 16,
        'max_depth': 4,
        'min_data_in_leaf': 20,
        'bagging_fraction': 0.8,
        'feature_fraction': 0.8,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'verbosity': -1,
        'seed': seed
    }

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof_predictions = np.zeros(len(X))
    fold_scores = []
    models = []

    print(f"特徴量数: {len(features)}")
    print(f"カテゴリ変数: {len(categorical_features)}個")
    print(f"\nCV: {n_splits}-Fold Stratified")
    print("-" * 60)

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
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=0)
            ]
        )

        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        oof_predictions[val_idx] = val_pred

        val_acc = accuracy_score(y_val, (val_pred > 0.5).astype(int))
        val_auc = roc_auc_score(y_val, val_pred)
        fold_scores.append({'fold': fold, 'accuracy': val_acc, 'auc': val_auc})

        print(f"Fold {fold}: Accuracy={val_acc:.4f}, AUC={val_auc:.4f}, Best Iteration={model.best_iteration}")

        models.append(model)

    oof_acc = accuracy_score(y, (oof_predictions > 0.5).astype(int))
    oof_auc = roc_auc_score(y, oof_predictions)

    print("-" * 60)
    print(f"OOF Accuracy: {oof_acc:.4f}")
    print(f"OOF AUC: {oof_auc:.4f}")
    print("=" * 60)

    # Feature Importance（上位20）
    print("\nFeature Importance (上位18):")
    importance = pd.DataFrame()
    for i, model in enumerate(models):
        fold_importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importance(importance_type='gain')
        })
        importance = pd.concat([importance, fold_importance])

    importance = importance.groupby('feature')['importance'].mean().sort_values(ascending=False)
    print(importance)

    return models, oof_predictions, fold_scores

def predict_test(models, test_df, features):
    """複数モデルの平均でテストデータを予測"""
    X_test = test_df[features].copy()

    categorical_features = [
        'Sex', 'Pclass', 'Embarked', 'Title', 'Deck', 'AgeBin'
    ]
    categorical_features = [f for f in categorical_features if f in features]

    for col in categorical_features:
        X_test[col] = X_test[col].astype('category')

    predictions = np.zeros(len(X_test))

    for model in models:
        pred = model.predict(X_test, num_iteration=model.best_iteration)
        predictions += pred

    predictions /= len(models)

    return predictions

# ====================================================================
# メイン実行
# ====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Titanic Perished Prediction v2.1")
    print("HasFamilyMatch削除（洗練版）")
    print("=" * 60)
    print()

    # データ読み込み
    train_raw = pd.read_csv('train.csv')
    test_raw = pd.read_csv('test.csv')

    print(f"Train shape: {train_raw.shape}")
    print(f"Test shape: {test_raw.shape}")
    print()

    # 特徴量生成
    train, test = make_features(train_raw, test_raw)

    # v2.1の特徴量（HasFamilyMatchを削除）
    features = [
        # 数値
        'Age', 'Fare', 'FarePerPerson',
        'SibSp', 'Parch', 'FamilySize',
        'TicketGroupSize', 'FamilyGroupSize', 'CabinGroupSize',
        # カテゴリ
        'Sex', 'Pclass', 'Embarked', 'Title', 'Deck', 'AgeBin',
        # フラグ
        'IsAlone', 'IsChild', 'IsMother'
        # ✗ HasFamilyMatch は削除
    ]

    print(f"\n使用特徴量: {len(features)}個")
    print("v2からの変更:")
    print("  - HasFamilyMatch を削除（重要度5.10, 最下位）")
    print("  - 特徴量数: 19個 → 18個")
    print()

    # モデル訓練
    models, oof_predictions, fold_scores = train_model(
        train, features, target='Perished', n_splits=5, seed=42
    )

    # テスト予測
    print("\nテストデータ予測中...")
    test_predictions = predict_test(models, test, features)

    # 提出ファイル作成
    submission = pd.DataFrame({
        'PassengerId': test_raw['PassengerId'],
        'Perished': (test_predictions > 0.5).astype(int)
    })

    submission_file = 'submission_restart_v2_1.csv'
    submission.to_csv(submission_file, index=False)

    print(f"\n提出ファイル作成: {submission_file}")
    print(f"予測Perished分布:")
    print(submission['Perished'].value_counts().sort_index())
    print(f"予測死亡率: {submission['Perished'].mean():.3f}")

    print("\n" + "=" * 60)
    print("v2.1 サマリー")
    print("=" * 60)
    print("✓ v2から改善:")
    print("  - HasFamilyMatch を削除（ノイズ削減）")
    print("  - 特徴量数削減: 19個 → 18個")
    print()
    print("次のステップ:")
    print("  1. v2とのCVスコア比較")
    print("  2. 改善確認できたら最終版として採用")
    print("=" * 60)
