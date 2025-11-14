"""
Titanic Perished Prediction v2.2
家族関連特徴量を厳選（SibSp, Parch削除）

v2からの変更:
- SibSp, Parch を削除（重要度26, 25）
- 理由: FamilySize = SibSp + Parch + 1 に情報が含まれている
- 特徴量数: 19個 → 17個
- 期待: ノイズ削減、CVスコア改善
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
    """Name列からSurname（姓）を抽出
    例: "Braund, Mr. Owen Harris" -> "Braund"
    """
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
    # 複数のCabinがある場合（例: "C23 C25 C27"）は最初のものを使う
    cabins = str(cabin).split()
    if len(cabins) > 0:
        return cabins[0].strip()
    return 'Unknown'

def make_features(train_raw, test_raw):
    """
    全特徴量を生成する関数

    重要: train+testを結合してグループサイズを計算
    - TicketGroupSize ✓
    - FamilyGroupSize (NEW!)
    - CabinGroupSize (NEW!)
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

    # Titleの統合
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

    # B. Surname抽出（NEW!）
    print("特徴量生成: Surname")
    for df in [train, test]:
        df['Surname'] = df['Name'].apply(extract_surname)

    # C. FamilySize, IsAlone
    print("特徴量生成: FamilySize, IsAlone")
    for df in [train, test]:
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # ====================================================================
    # train+testを結合してグループサイズを計算（重要！）
    # ====================================================================

    print("\n【train+test結合でグループサイズ計算】")

    # D. TicketGroupSize (既存)
    print("1. TicketGroupSize")
    for df in [train, test]:
        df['TicketNorm'] = df['Ticket'].apply(normalize_ticket)

    full = pd.concat([train, test], sort=False)

    # Ticketグループサイズ
    ticket_group_size = full.groupby('TicketNorm')['PassengerId'].transform('size')
    train['TicketGroupSize'] = ticket_group_size[:len(train)].values
    test['TicketGroupSize'] = ticket_group_size[len(train):].values

    print(f"   Train TicketGroupSize範囲: {train['TicketGroupSize'].min()}-{train['TicketGroupSize'].max()}")
    print(f"   Test TicketGroupSize範囲: {test['TicketGroupSize'].min()}-{test['TicketGroupSize'].max()}")

    # E. FamilyGroupSize (NEW! - train+testで計算)
    print("2. FamilyGroupSize (NEW!)")
    # FamilyID = Surname + FamilySize
    # 同じ姓+同じ家族サイズ = 同じ家族の可能性が高い
    for df in [train, test]:
        df['FamilyID'] = df['Surname'] + '_' + df['FamilySize'].astype(str)

    full = pd.concat([train, test], sort=False)
    family_group_size = full.groupby('FamilyID')['PassengerId'].transform('size')

    train['FamilyGroupSize'] = family_group_size[:len(train)].values
    test['FamilyGroupSize'] = family_group_size[len(train):].values

    print(f"   Train FamilyGroupSize範囲: {train['FamilyGroupSize'].min()}-{train['FamilyGroupSize'].max()}")
    print(f"   Test FamilyGroupSize範囲: {test['FamilyGroupSize'].min()}-{test['FamilyGroupSize'].max()}")

    # FamilyGroupSize > FamilySize なら、同じ家族が複数いる可能性
    for df in [train, test]:
        df['HasFamilyMatch'] = (df['FamilyGroupSize'] > df['FamilySize']).astype(int)

    # F. CabinGroupSize (NEW! - train+testで計算)
    print("3. CabinGroupSize (NEW!)")
    for df in [train, test]:
        df['CabinNorm'] = df['Cabin'].apply(normalize_cabin)

    full = pd.concat([train, test], sort=False)
    cabin_group_size = full.groupby('CabinNorm')['PassengerId'].transform('size')

    train['CabinGroupSize'] = cabin_group_size[:len(train)].values
    test['CabinGroupSize'] = cabin_group_size[len(train):].values

    # Cabin='Unknown'の場合はグループサイズを1に（共有なし）
    for df in [train, test]:
        df.loc[df['CabinNorm'] == 'Unknown', 'CabinGroupSize'] = 1

    print(f"   Train CabinGroupSize範囲: {train['CabinGroupSize'].min()}-{train['CabinGroupSize'].max()}")
    print(f"   Cabin共有あり（>1）: Train={sum(train['CabinGroupSize']>1)}, Test={sum(test['CabinGroupSize']>1)}")

    # G. FarePerPerson
    print("4. FarePerPerson")
    for df in [train, test]:
        df['FarePerPerson'] = df['Fare'] / df['TicketGroupSize']

    # H. Deck (Cabinの先頭文字)
    print("\n特徴量生成: Deck")
    for df in [train, test]:
        df['Deck'] = df['Cabin'].apply(extract_deck)

    # ====================================================================
    # A級特徴量
    # ====================================================================

    # I. Age欠損補完（train+testでTitle・Pclass・Sex別の中央値）
    print("欠損値補完: Age (train+test, Title+Pclass+Sex別)")

    # train+testを結合してAge統計を計算（グループサイズと同じ発想）
    full_for_age = pd.concat([train, test], sort=False)

    # Title+Pclass+Sexの組み合わせで中央値を計算
    age_by_group = full_for_age.groupby(['Title', 'Pclass', 'Sex'])['Age'].median()
    age_by_title_pclass = full_for_age.groupby(['Title', 'Pclass'])['Age'].median()
    age_by_title = full_for_age.groupby(['Title'])['Age'].median()

    for df in [train, test]:
        mask_missing = df['Age'].isnull()
        for idx in df[mask_missing].index:
            title = df.loc[idx, 'Title']
            pclass = df.loc[idx, 'Pclass']
            sex = df.loc[idx, 'Sex']

            # 優先度1: Title+Pclass+Sex
            if (title, pclass, sex) in age_by_group.index:
                df.loc[idx, 'Age'] = age_by_group[(title, pclass, sex)]
            # 優先度2: Title+Pclass
            elif (title, pclass) in age_by_title_pclass.index:
                df.loc[idx, 'Age'] = age_by_title_pclass[(title, pclass)]
            # 優先度3: Title
            elif title in age_by_title.index:
                df.loc[idx, 'Age'] = age_by_title[title]
            # 最終: 全体の中央値
            else:
                df.loc[idx, 'Age'] = full_for_age['Age'].median()

        # 念のため残った欠損値も補完
        df['Age'] = df['Age'].fillna(full_for_age['Age'].median())

    # J. AgeBin
    print("特徴量生成: AgeBin")
    age_bins = [0, 12, 18, 35, 55, 80]
    age_labels = ['Child', 'Teen', 'Young', 'Mid', 'Senior']

    for df in [train, test]:
        df['AgeBin'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, include_lowest=True)
        df['AgeBin'] = df['AgeBin'].astype(str)

    # K. IsChild, IsMother
    print("特徴量生成: IsChild, IsMother")
    for df in [train, test]:
        df['IsChild'] = (df['Age'] <= 12).astype(int)
        df['IsMother'] = (
            (df['Sex'] == 'female') &
            (df['Age'] >= 18) &
            (df['Parch'] > 0) &
            (df['Title'] == 'Mrs')
        ).astype(int)

    # L. その他の欠損値補完
    print("欠損値補完: Fare, Embarked")
    fare_by_pclass = train.groupby('Pclass')['Fare'].median()
    for df in [train, test]:
        for pclass in [1, 2, 3]:
            mask = (df['Fare'].isnull()) & (df['Pclass'] == pclass)
            df.loc[mask, 'Fare'] = fare_by_pclass[pclass]

    most_common_embarked = train['Embarked'].mode()[0]
    for df in [train, test]:
        df['Embarked'] = df['Embarked'].fillna(most_common_embarked)

    # FarePerPersonも再計算
    for df in [train, test]:
        df['FarePerPerson'] = df['Fare'] / df['TicketGroupSize']

    print("\n✓ 特徴量生成完了！")
    return train, test

# ====================================================================
# 2. モデル訓練・予測
# ====================================================================

def train_model(train_df, features, target='Perished', n_splits=5, seed=42):
    """Stratified K-FoldでLightGBMを訓練"""

    print("=" * 60)
    print("モデル訓練開始")
    print("=" * 60)

    X = train_df[features].copy()
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
    print(f"カテゴリ変数: {categorical_features}")
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

    # Feature Importance
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
# 3. メイン実行
# ====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Titanic Perished Prediction v2.2")
    print("家族関連特徴量を厳選（SibSp, Parch削除）")
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

    # 使用する特徴量（v2.6: Age削除）
    features = [
        # 数値
        # 'Age',  # v2.6: 削除（AgeBin, IsChild, IsMother, Titleと相関）
        'Fare', 'FarePerPerson',
        # 'SibSp', 'Parch',  # v2.2: 削除（FamilySizeに含まれているため）
        'FamilySize',
        'TicketGroupSize',
        'FamilyGroupSize',
        'CabinGroupSize',
        # カテゴリ
        'Sex', 'Pclass', 'Embarked', 'Title', 'Deck', 'AgeBin',
        # フラグ
        'IsAlone', 'IsChild', 'IsMother',
        'HasFamilyMatch'
    ]

    print(f"\n使用特徴量: {len(features)}個")
    print("v2.8の変更:")
    print("  - v2.6と同じ特徴量（Age削除）")
    print("  - AgeBin作成時のAge補完を改良:")
    print("    * train+testのデータを使用")
    print("    * Title+Pclass+Sex別の中央値で補完")
    print("  - より正確なAgeBinを生成")
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

    submission_file = 'submission_restart_v2_8.csv'
    submission.to_csv(submission_file, index=False)

    print(f"\n提出ファイル作成: {submission_file}")
    print(f"予測Perished分布:")
    print(submission['Perished'].value_counts().sort_index())
    print(f"予測死亡率: {submission['Perished'].mean():.3f}")

    print("\n" + "=" * 60)
    print("v2.8 サマリー")
    print("=" * 60)
    print("✓ v2.6から変更:")
    print("  - 特徴量は同じ（Age削除、AgeBin使用）")
    print("  - AgeBin生成時のAge補完を改良:")
    print("    * train+testデータを使用")
    print("    * Title+Pclass+Sex別の中央値")
    print("  - より正確なAgeBin, IsChild, IsMotherを生成")
    print()
    print("検証ポイント:")
    print("  1. v2.6とのCVスコア比較（85.19% vs ?）")
    print("  2. AgeBinの精度向上でスコア改善するか")
    print("=" * 60)
    print("完了！")
    print("=" * 60)
