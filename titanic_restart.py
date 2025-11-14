"""
Titanic Perished Prediction - 戦略的アプローチ
再スタート版

目的変数: Perished (1=死亡, 0=生存)

重要な特徴量（優先度順）:
S級: Title, FamilySize, TicketGroupSize, Deck
A級: AgeBin, IsMother, IsChild, 適切な欠損値補完

モデル: 浅めのGBDT (LightGBM)
CV: Stratified K-Fold
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
import lightgbm as lgb

# ====================================================================
# 1. 特徴量エンジニアリング関数
# ====================================================================

def extract_title(name: str) -> str:
    """Name列からTitleを抽出
    例: "Braund, Mr. Owen Harris" -> "Mr"
    """
    m = re.search(r',\s*([^\.]+)\.', str(name))
    if m:
        return m.group(1).strip()
    return "Unknown"

def normalize_ticket(ticket: str) -> str:
    """Ticketを正規化（空白除去・大文字化）"""
    return ''.join(str(ticket).split()).upper()

def extract_deck(cabin) -> str:
    """Cabinの先頭文字をDeckとして抽出
    欠損は 'U' (Unknown)
    """
    if pd.isna(cabin):
        return 'U'
    return str(cabin)[0]

def make_features(train_raw, test_raw):
    """
    全特徴量を生成する関数

    S級特徴量:
    - Title (Nameから抽出 + rare title統合)
    - FamilySize, IsAlone
    - TicketGroupSize (train+testで計算)
    - FarePerPerson
    - Deck (Cabinの先頭文字)

    A級特徴量:
    - AgeBin (年齢区分)
    - IsMother, IsChild
    - 欠損値補完 (Title/Pclass/Embarkedベース)
    """

    train = train_raw.copy()
    test = test_raw.copy()

    # ====================================================================
    # S級特徴量
    # ====================================================================

    # A. Title抽出
    print("特徴量生成: Title")
    for df in [train, test]:
        df['Title'] = df['Name'].apply(extract_title)

    # Titleの統合（レアな称号をまとめる）
    title_mapping = {
        'Capt': 'Officer',
        'Col': 'Officer',
        'Major': 'Officer',
        'Dr': 'Officer',
        'Rev': 'Officer',
        'Don': 'Noble',
        'Sir': 'Noble',
        'the Countess': 'Noble',
        'Countess': 'Noble',
        'Lady': 'Noble',
        'Jonkheer': 'Noble',
        'Dona': 'Noble',
        'Mlle': 'Miss',
        'Ms': 'Miss',
        'Mme': 'Mrs'
    }

    for df in [train, test]:
        df['Title'] = df['Title'].replace(title_mapping)

    print(f"  Train Title分布: {train['Title'].value_counts().to_dict()}")
    print(f"  Test Title分布: {test['Title'].value_counts().to_dict()}")

    # B. FamilySize, IsAlone
    print("特徴量生成: FamilySize, IsAlone")
    for df in [train, test]:
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # C. TicketGroupSize (train+testで計算 - 重要！)
    print("特徴量生成: TicketGroupSize, FarePerPerson")
    for df in [train, test]:
        df['TicketNorm'] = df['Ticket'].apply(normalize_ticket)

    # train+testを結合してグループサイズを計算
    full = pd.concat([train, test], sort=False)
    group_size = full.groupby('TicketNorm')['PassengerId'].transform('size')

    train['TicketGroupSize'] = group_size[:len(train)].values
    test['TicketGroupSize'] = group_size[len(train):].values

    # FarePerPerson
    for df in [train, test]:
        df['FarePerPerson'] = df['Fare'] / df['TicketGroupSize']

    # D. Deck (Cabinの先頭文字)
    print("特徴量生成: Deck")
    for df in [train, test]:
        df['Deck'] = df['Cabin'].apply(extract_deck)

    print(f"  Train Deck分布: {train['Deck'].value_counts().to_dict()}")

    # ====================================================================
    # A級特徴量
    # ====================================================================

    # E. Age欠損補完（Title・Pclass・Embarked別の中央値）
    print("欠損値補完: Age")
    # まずTitleとPclass別の中央値を計算
    age_by_title_pclass = train.groupby(['Title', 'Pclass'])['Age'].median()

    for df in [train, test]:
        # Age欠損を埋める
        mask_missing = df['Age'].isnull()
        for idx in df[mask_missing].index:
            title = df.loc[idx, 'Title']
            pclass = df.loc[idx, 'Pclass']
            if (title, pclass) in age_by_title_pclass.index:
                df.loc[idx, 'Age'] = age_by_title_pclass[(title, pclass)]
            else:
                # フォールバック: Title全体の中央値
                df.loc[idx, 'Age'] = train[train['Title'] == title]['Age'].median()

        # それでもNaNなら全体中央値
        df['Age'].fillna(train['Age'].median(), inplace=True)

    # F. AgeBin（年齢区分）
    print("特徴量生成: AgeBin")
    age_bins = [0, 12, 18, 35, 55, 80]
    age_labels = ['Child', 'Teen', 'Young', 'Mid', 'Senior']

    for df in [train, test]:
        df['AgeBin'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, include_lowest=True)
        df['AgeBin'] = df['AgeBin'].astype(str)

    # G. IsChild, IsMother
    print("特徴量生成: IsChild, IsMother")
    for df in [train, test]:
        df['IsChild'] = (df['Age'] <= 12).astype(int)

        # IsMother: 女性 & 年齢18歳以上 & Parch>=1 & Title=Mrs
        df['IsMother'] = (
            (df['Sex'] == 'female') &
            (df['Age'] >= 18) &
            (df['Parch'] > 0) &
            (df['Title'] == 'Mrs')
        ).astype(int)

    # H. その他の欠損値補完
    print("欠損値補完: Fare, Embarked")
    # Fare: Pclass別の中央値
    fare_by_pclass = train.groupby('Pclass')['Fare'].median()
    for df in [train, test]:
        for pclass in [1, 2, 3]:
            mask = (df['Fare'].isnull()) & (df['Pclass'] == pclass)
            df.loc[mask, 'Fare'] = fare_by_pclass[pclass]

    # Embarked: 最頻値
    most_common_embarked = train['Embarked'].mode()[0]
    for df in [train, test]:
        df['Embarked'].fillna(most_common_embarked, inplace=True)

    # FarePerPersonも再計算（Fareが補完されたので）
    for df in [train, test]:
        df['FarePerPerson'] = df['Fare'] / df['TicketGroupSize']

    print("\n特徴量生成完了！")
    return train, test

# ====================================================================
# 2. モデル訓練・予測
# ====================================================================

def train_model(train_df, features, target='Perished', n_splits=5, seed=42):
    """
    Stratified K-FoldでLightGBMを訓練

    パラメータ:
    - 浅めの木 (max_depth=4, num_leaves=16)
    - 学習率 0.03
    - 過学習防止 (bagging, feature_fraction, 正則化)
    """

    print("=" * 60)
    print("モデル訓練開始")
    print("=" * 60)

    X = train_df[features]
    y = train_df[target]

    # カテゴリ変数を指定
    categorical_features = [
        'Sex', 'Pclass', 'Embarked', 'Title', 'Deck', 'AgeBin'
    ]
    categorical_features = [f for f in categorical_features if f in features]

    # カテゴリ変数をカテゴリ型に変換
    for col in categorical_features:
        X[col] = X[col].astype('category')

    # LightGBMパラメータ（浅めの木）
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': 0.03,
        'num_leaves': 16,          # 浅め
        'max_depth': 4,            # 浅め
        'min_data_in_leaf': 20,
        'bagging_fraction': 0.8,
        'feature_fraction': 0.8,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'verbosity': -1,
        'seed': seed
    }

    # Stratified K-Fold
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof_predictions = np.zeros(len(X))
    fold_scores = []
    models = []

    print(f"特徴量数: {len(features)}")
    print(f"カテゴリ変数: {categorical_features}")
    print(f"\nCV: {n_splits}-Fold Stratified")
    print("-" * 60)

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # LightGBM Dataset
        train_data = lgb.Dataset(
            X_train, label=y_train,
            categorical_feature=categorical_features
        )
        val_data = lgb.Dataset(
            X_val, label=y_val,
            categorical_feature=categorical_features,
            reference=train_data
        )

        # 訓練
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=0)  # ログ出力を抑制
            ]
        )

        # 予測
        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        oof_predictions[val_idx] = val_pred

        # スコア計算
        val_acc = accuracy_score(y_val, (val_pred > 0.5).astype(int))
        val_auc = roc_auc_score(y_val, val_pred)
        fold_scores.append({'fold': fold, 'accuracy': val_acc, 'auc': val_auc})

        print(f"Fold {fold}: Accuracy={val_acc:.4f}, AUC={val_auc:.4f}, Best Iteration={model.best_iteration}")

        models.append(model)

    # OOF全体スコア
    oof_acc = accuracy_score(y, (oof_predictions > 0.5).astype(int))
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

def predict_test(models, test_df, features):
    """
    複数モデルの平均でテストデータを予測
    """
    X_test = test_df[features].copy()

    # カテゴリ変数をカテゴリ型に変換
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
# 3. メイン実行
# ====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Titanic Perished Prediction - 再スタート")
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

    # 使用する特徴量
    features = [
        # 数値
        'Age', 'Fare', 'FarePerPerson',
        'SibSp', 'Parch', 'FamilySize', 'TicketGroupSize',
        # カテゴリ
        'Sex', 'Pclass', 'Embarked', 'Title', 'Deck', 'AgeBin',
        # フラグ
        'IsAlone', 'IsChild', 'IsMother'
    ]

    print(f"\n使用特徴量: {len(features)}個")
    print(features)
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

    submission_file = 'submission_restart.csv'
    submission.to_csv(submission_file, index=False)

    print(f"\n提出ファイル作成: {submission_file}")
    print(f"予測Perished分布:")
    print(submission['Perished'].value_counts().sort_index())
    print(f"予測死亡率: {submission['Perished'].mean():.3f}")
    print()
    print("=" * 60)
    print("完了！")
    print("=" * 60)
