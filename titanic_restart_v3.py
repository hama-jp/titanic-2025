"""
Titanic Perished Prediction v3
交互作用特徴を厳選して追加

新規追加：
1. Sex × Pclass - 救命優先度の直接表現
2. Sex × AgeBin - 年齢層×性別の組み合わせ
3. IsChild × Pclass - 子供×階級
4. Pclass × FarePerPerson - 階級×実質単価のズレ検出

戦略：
- 浅い木（max_depth=4）の弱点を交互作用で補う
- やりすぎない（4つだけ厳選）
- CVスコアで有意な改善を確認
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
    """Ticketを正規化"""
    return ''.join(str(ticket).split()).upper()

def extract_deck(cabin) -> str:
    """Cabinの先頭文字をDeckとして抽出"""
    if pd.isna(cabin):
        return 'U'
    return str(cabin)[0]

def normalize_cabin(cabin) -> str:
    """Cabinを正規化"""
    if pd.isna(cabin):
        return 'Unknown'
    cabins = str(cabin).split()
    if len(cabins) > 0:
        return cabins[0].strip()
    return 'Unknown'

def make_features(train_raw, test_raw):
    """
    v3: 交互作用特徴を追加
    """

    train = train_raw.copy()
    test = test_raw.copy()

    print("=" * 60)
    print("基本特徴量生成")
    print("=" * 60)

    # Title抽出
    print("特徴量: Title")
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

    # Surname抽出
    print("特徴量: Surname")
    for df in [train, test]:
        df['Surname'] = df['Name'].apply(extract_surname)

    # FamilySize, IsAlone
    print("特徴量: FamilySize, IsAlone")
    for df in [train, test]:
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # train+testでグループサイズ計算
    print("特徴量: TicketGroupSize, FamilyGroupSize, CabinGroupSize")
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

    for df in [train, test]:
        df['HasFamilyMatch'] = (df['FamilyGroupSize'] > df['FamilySize']).astype(int)

    # CabinGroupSize
    cabin_group_size = full.groupby('CabinNorm')['PassengerId'].transform('size')
    train['CabinGroupSize'] = cabin_group_size[:len(train)].values
    test['CabinGroupSize'] = cabin_group_size[len(train):].values

    for df in [train, test]:
        df.loc[df['CabinNorm'] == 'Unknown', 'CabinGroupSize'] = 1

    # FarePerPerson
    for df in [train, test]:
        df['FarePerPerson'] = df['Fare'] / df['TicketGroupSize']

    # Deck
    print("特徴量: Deck")
    for df in [train, test]:
        df['Deck'] = df['Cabin'].apply(extract_deck)

    # Age欠損補完
    print("欠損値補完: Age")
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
    print("特徴量: AgeBin")
    age_bins = [0, 12, 18, 35, 55, 80]
    age_labels = ['Child', 'Teen', 'Young', 'Mid', 'Senior']

    for df in [train, test]:
        df['AgeBin'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, include_lowest=True)
        df['AgeBin'] = df['AgeBin'].astype(str)

    # IsChild, IsMother
    print("特徴量: IsChild, IsMother")
    for df in [train, test]:
        df['IsChild'] = (df['Age'] <= 12).astype(int)
        df['IsMother'] = (
            (df['Sex'] == 'female') &
            (df['Age'] >= 18) &
            (df['Parch'] > 0) &
            (df['Title'] == 'Mrs')
        ).astype(int)

    # Fare/Embarked欠損補完
    print("欠損値補完: Fare, Embarked")
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

    # ====================================================================
    # v3: 交互作用特徴（厳選4つ）
    # ====================================================================
    print("\n" + "=" * 60)
    print("v3: 交互作用特徴を追加")
    print("=" * 60)

    # 1. Sex × Pclass - 救命優先度の直接表現
    print("1. Sex × Pclass")
    for df in [train, test]:
        df['Sex_Pclass'] = df['Sex'].astype(str) + '_' + df['Pclass'].astype(str)

    print(f"   カテゴリ数: {train['Sex_Pclass'].nunique()}")
    print(f"   分布: {train['Sex_Pclass'].value_counts().head()}")

    # 2. Sex × AgeBin - 年齢層×性別
    print("2. Sex × AgeBin")
    for df in [train, test]:
        df['Sex_AgeBin'] = df['Sex'].astype(str) + '_' + df['AgeBin'].astype(str)

    print(f"   カテゴリ数: {train['Sex_AgeBin'].nunique()}")

    # 3. IsChild × Pclass - 子供×階級
    print("3. IsChild × Pclass")
    for df in [train, test]:
        df['Child_Pclass'] = df['IsChild'].astype(str) + '_' + df['Pclass'].astype(str)

    print(f"   カテゴリ数: {train['Child_Pclass'].nunique()}")

    # 4. Pclass × FarePerPerson（分位ビン化してから結合）
    print("4. Pclass × FarePerPerson")
    for df in [train, test]:
        # FarePerPersonを4分位でビン化
        df['FarePP_q'] = pd.qcut(df['FarePerPerson'], 4, labels=['Q1','Q2','Q3','Q4'], duplicates='drop')
        df['Pclass_FarePP'] = df['Pclass'].astype(str) + '_' + df['FarePP_q'].astype(str)

    print(f"   カテゴリ数: {train['Pclass_FarePP'].nunique()}")

    print("\n✓ 特徴量生成完了（v3）")
    return train, test

# ====================================================================
# 2. モデル訓練・予測
# ====================================================================

def train_model(train_df, features, target='Perished', n_splits=5, seed=42):
    """Stratified K-FoldでLightGBMを訓練"""

    print("\n" + "=" * 60)
    print("モデル訓練開始")
    print("=" * 60)

    X = train_df[features].copy()
    y = train_df[target]

    # カテゴリ変数を指定
    categorical_features = [
        'Sex', 'Pclass', 'Embarked', 'Title', 'Deck', 'AgeBin',
        # v3: 交互作用（カテゴリ型）
        'Sex_Pclass', 'Sex_AgeBin', 'Child_Pclass', 'Pclass_FarePP'
    ]
    categorical_features = [f for f in categorical_features if f in features]

    for col in categorical_features:
        X[col] = X[col].astype('category')

    # LightGBMパラメータ（浅めの木）
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
    print("\nFeature Importance (上位20):")
    importance = pd.DataFrame()
    for i, model in enumerate(models):
        fold_importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importance(importance_type='gain')
        })
        importance = pd.concat([importance, fold_importance])

    importance = importance.groupby('feature')['importance'].mean().sort_values(ascending=False)
    print(importance.head(20))

    # 交互作用特徴の重要度を抜粋
    print("\n【v3交互作用特徴の重要度】:")
    interaction_features = ['Sex_Pclass', 'Sex_AgeBin', 'Child_Pclass', 'Pclass_FarePP']
    for feat in interaction_features:
        if feat in importance.index:
            print(f"  {feat:20s}: {importance[feat]:8.1f} (全体{importance.rank(ascending=False)[feat]:.0f}位)")

    return models, oof_predictions, fold_scores

def predict_test(models, test_df, features):
    """複数モデルの平均でテストデータを予測"""
    X_test = test_df[features].copy()

    categorical_features = [
        'Sex', 'Pclass', 'Embarked', 'Title', 'Deck', 'AgeBin',
        'Sex_Pclass', 'Sex_AgeBin', 'Child_Pclass', 'Pclass_FarePP'
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
    print("Titanic Perished Prediction v3")
    print("交互作用特徴を厳選追加")
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

    # 使用する特徴量（v3: 交互作用を追加）
    features = [
        # 数値
        'Age', 'Fare', 'FarePerPerson',
        'SibSp', 'Parch', 'FamilySize',
        'TicketGroupSize', 'FamilyGroupSize', 'CabinGroupSize',
        # カテゴリ
        'Sex', 'Pclass', 'Embarked', 'Title', 'Deck', 'AgeBin',
        # フラグ
        'IsAlone', 'IsChild', 'IsMother', 'HasFamilyMatch',
        # v3: 交互作用（厳選4つ）
        'Sex_Pclass',      # 救命優先度
        'Sex_AgeBin',      # 年齢層×性別
        'Child_Pclass',    # 子供×階級
        'Pclass_FarePP'    # 階級×実質単価
    ]

    print(f"\n使用特徴量: {len(features)}個")
    print(f"  基本特徴量: {len(features) - 4}個")
    print(f"  v3交互作用: 4個")
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

    submission_file = 'submission_restart_v3.csv'
    submission.to_csv(submission_file, index=False)

    print(f"\n提出ファイル作成: {submission_file}")
    print(f"予測Perished分布:")
    print(submission['Perished'].value_counts().sort_index())
    print(f"予測死亡率: {submission['Perished'].mean():.3f}")

    print("\n" + "=" * 60)
    print("v3 改善点サマリー")
    print("=" * 60)
    print("✓ 交互作用特徴4つを追加:")
    print("  - Sex_Pclass: 救命優先度の直接表現")
    print("  - Sex_AgeBin: 年齢層×性別の組み合わせ")
    print("  - Child_Pclass: 子供×階級")
    print("  - Pclass_FarePP: 階級×実質単価のズレ")
    print()
    print("次のステップ:")
    print("  1. v2とのCVスコア比較")
    print("  2. 改善が有意なら→スタッキングアンサンブルへ")
    print("=" * 60)
