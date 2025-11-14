"""
Titanic Perished Prediction - Pseudo-labeling
v2.6をベースにPseudo-labelingを適用

戦略:
1. v2.6モデルでテストデータを予測
2. 高確信度（proba >= 0.95 または proba <= 0.05）のサンプルをpseudo-labelとして選択
3. それらを訓練データに追加して再訓練
4. CVスコアを比較
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
import lightgbm as lgb

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

    # S級特徴量
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

    print("特徴量生成: Surname")
    for df in [train, test]:
        df['Surname'] = df['Name'].apply(extract_surname)

    print("特徴量生成: FamilySize, IsAlone")
    for df in [train, test]:
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # train+testを結合してグループサイズを計算
    print("\n【train+test結合でグループサイズ計算】")

    print("1. TicketGroupSize")
    for df in [train, test]:
        df['TicketNorm'] = df['Ticket'].apply(normalize_ticket)

    full = pd.concat([train, test], sort=False)

    ticket_group_size = full.groupby('TicketNorm')['PassengerId'].transform('size')
    train['TicketGroupSize'] = ticket_group_size[:len(train)].values
    test['TicketGroupSize'] = ticket_group_size[len(train):].values

    print(f"   Train TicketGroupSize範囲: {train['TicketGroupSize'].min()}-{train['TicketGroupSize'].max()}")
    print(f"   Test TicketGroupSize範囲: {test['TicketGroupSize'].min()}-{test['TicketGroupSize'].max()}")

    print("2. FamilyGroupSize")
    for df in [train, test]:
        df['FamilyID'] = df['Surname'] + '_' + df['FamilySize'].astype(str)

    full = pd.concat([train, test], sort=False)
    family_group_size = full.groupby('FamilyID')['PassengerId'].transform('size')

    train['FamilyGroupSize'] = family_group_size[:len(train)].values
    test['FamilyGroupSize'] = family_group_size[len(train):].values

    print(f"   Train FamilyGroupSize範囲: {train['FamilyGroupSize'].min()}-{train['FamilyGroupSize'].max()}")
    print(f"   Test FamilyGroupSize範囲: {test['FamilyGroupSize'].min()}-{test['FamilyGroupSize'].max()}")

    for df in [train, test]:
        df['HasFamilyMatch'] = (df['FamilyGroupSize'] > df['FamilySize']).astype(int)

    print("3. CabinGroupSize")
    for df in [train, test]:
        df['CabinNorm'] = df['Cabin'].apply(normalize_cabin)

    full = pd.concat([train, test], sort=False)
    cabin_group_size = full.groupby('CabinNorm')['PassengerId'].transform('size')

    train['CabinGroupSize'] = cabin_group_size[:len(train)].values
    test['CabinGroupSize'] = cabin_group_size[len(train):].values

    for df in [train, test]:
        df.loc[df['CabinNorm'] == 'Unknown', 'CabinGroupSize'] = 1

    print(f"   Train CabinGroupSize範囲: {train['CabinGroupSize'].min()}-{train['CabinGroupSize'].max()}")
    print(f"   Cabin共有あり（>1）: Train={sum(train['CabinGroupSize']>1)}, Test={sum(test['CabinGroupSize']>1)}")

    print("4. FarePerPerson")
    for df in [train, test]:
        df['FarePerPerson'] = df['Fare'] / df['TicketGroupSize']

    print("\n特徴量生成: Deck")
    for df in [train, test]:
        df['Deck'] = df['Cabin'].apply(extract_deck)

    # A級特徴量
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

    print("特徴量生成: AgeBin")
    age_bins = [0, 12, 18, 35, 55, 80]
    age_labels = ['Child', 'Teen', 'Young', 'Mid', 'Senior']

    for df in [train, test]:
        df['AgeBin'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, include_lowest=True)
        df['AgeBin'] = df['AgeBin'].astype(str)

    print("特徴量生成: IsChild, IsMother")
    for df in [train, test]:
        df['IsChild'] = (df['Age'] <= 12).astype(int)
        df['IsMother'] = (
            (df['Sex'] == 'female') &
            (df['Age'] >= 18) &
            (df['Parch'] > 0) &
            (df['Title'] == 'Mrs')
        ).astype(int)

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
# 3. Pseudo-labeling
# ====================================================================

def apply_pseudo_labeling(train_df, test_df, test_predictions, features, threshold_high=0.95, threshold_low=0.05):
    """
    高確信度のテストサンプルをpseudo-labelとして訓練データに追加

    Args:
        train_df: 訓練データ
        test_df: テストデータ
        test_predictions: テストデータの予測確率
        features: 使用する特徴量
        threshold_high: 死亡と判定する確信度閾値（proba >= threshold_high → Perished=1）
        threshold_low: 生存と判定する確信度閾値（proba <= threshold_low → Perished=0）

    Returns:
        拡張された訓練データ
    """

    print("\n" + "=" * 60)
    print("Pseudo-labeling適用")
    print("=" * 60)

    # 高確信度のサンプルを選択
    high_confidence_perished = test_predictions >= threshold_high
    high_confidence_survived = test_predictions <= threshold_low

    pseudo_mask = high_confidence_perished | high_confidence_survived

    print(f"高確信度サンプル数: {pseudo_mask.sum()}/{len(test_predictions)}")
    print(f"  - 死亡予測 (proba >= {threshold_high}): {high_confidence_perished.sum()}")
    print(f"  - 生存予測 (proba <= {threshold_low}): {high_confidence_survived.sum()}")

    if pseudo_mask.sum() == 0:
        print("⚠ 高確信度サンプルがありません。閾値を調整してください。")
        return train_df

    # Pseudo-labelデータを作成
    pseudo_df = test_df[pseudo_mask].copy()
    pseudo_df['Perished'] = (test_predictions[pseudo_mask] > 0.5).astype(int)

    # 訓練データに追加
    extended_train = pd.concat([train_df, pseudo_df], ignore_index=True)

    print(f"\n拡張後の訓練データサイズ: {len(train_df)} → {len(extended_train)} (+{pseudo_mask.sum()})")
    print(f"Perished分布:")
    print(f"  - 元の訓練データ: {train_df['Perished'].value_counts().to_dict()}")
    print(f"  - 拡張後: {extended_train['Perished'].value_counts().to_dict()}")
    print("=" * 60)

    return extended_train

# ====================================================================
# 4. メイン実行
# ====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Titanic Perished Prediction - Pseudo-labeling")
    print("v2.6ベース + 高確信度テストサンプルを追加学習")
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

    # 使用する特徴量（v2.6と同じ）
    features = [
        # 数値
        'Fare', 'FarePerPerson',
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
    print()

    # ====================================================================
    # ステップ1: ベースラインモデル（v2.6）を訓練
    # ====================================================================
    print("\n" + "=" * 60)
    print("【ステップ1】ベースラインモデル（v2.6）訓練")
    print("=" * 60)

    models_base, oof_base, scores_base = train_model(
        train, features, target='Perished', n_splits=5, seed=42
    )

    # テスト予測
    print("\nテストデータ予測中...")
    test_predictions = predict_test(models_base, test, features)

    print(f"テスト予測分布:")
    print(f"  - Mean: {test_predictions.mean():.4f}")
    print(f"  - Min: {test_predictions.min():.4f}")
    print(f"  - Max: {test_predictions.max():.4f}")

    # ====================================================================
    # ステップ2: Pseudo-labelingを適用
    # ====================================================================
    print("\n" + "=" * 60)
    print("【ステップ2】Pseudo-labeling適用")
    print("=" * 60)

    # 複数の閾値を試す
    threshold_configs = [
        (0.95, 0.05),  # 厳しい閾値
        (0.90, 0.10),  # 中程度
        (0.85, 0.15),  # 緩い閾値
    ]

    best_config = None
    best_score = 0
    results = []

    for threshold_high, threshold_low in threshold_configs:
        print(f"\n{'='*60}")
        print(f"閾値: proba >= {threshold_high} or proba <= {threshold_low}")
        print(f"{'='*60}")

        # Pseudo-labelingを適用
        extended_train = apply_pseudo_labeling(
            train, test, test_predictions, features,
            threshold_high=threshold_high,
            threshold_low=threshold_low
        )

        # 拡張データで再訓練
        print("\n拡張データで再訓練中...")
        models_pseudo, oof_pseudo, scores_pseudo = train_model(
            extended_train, features, target='Perished', n_splits=5, seed=42
        )

        # 結果を記録
        result = {
            'threshold_high': threshold_high,
            'threshold_low': threshold_low,
            'n_pseudo': len(extended_train) - len(train),
            'oof_acc_base': accuracy_score(train['Perished'], (oof_base > 0.5).astype(int)),
            'oof_auc_base': roc_auc_score(train['Perished'], oof_base),
            'oof_acc_pseudo': accuracy_score(extended_train['Perished'], (oof_pseudo > 0.5).astype(int)),
            'oof_auc_pseudo': roc_auc_score(extended_train['Perished'], oof_pseudo),
        }
        results.append(result)

        # ベースラインとの比較（元の訓練データのみでOOFを計算）
        oof_pseudo_train_only = oof_pseudo[:len(train)]
        acc_train_only = accuracy_score(train['Perished'], (oof_pseudo_train_only > 0.5).astype(int))
        auc_train_only = roc_auc_score(train['Perished'], oof_pseudo_train_only)

        result['oof_acc_train_only'] = acc_train_only
        result['oof_auc_train_only'] = auc_train_only

        print(f"\n{'='*60}")
        print(f"結果比較（閾値 {threshold_high}/{threshold_low}）")
        print(f"{'='*60}")
        print(f"Pseudo-labelサンプル数: {result['n_pseudo']}")
        print(f"\nベースライン（v2.6）:")
        print(f"  OOF Accuracy: {result['oof_acc_base']:.4f}")
        print(f"  OOF AUC: {result['oof_auc_base']:.4f}")
        print(f"\nPseudo-labeling（元訓練データのみ）:")
        print(f"  OOF Accuracy: {acc_train_only:.4f} ({acc_train_only - result['oof_acc_base']:+.4f})")
        print(f"  OOF AUC: {auc_train_only:.4f} ({auc_train_only - result['oof_auc_base']:+.4f})")
        print(f"{'='*60}")

        if acc_train_only > best_score:
            best_score = acc_train_only
            best_config = (threshold_high, threshold_low, models_pseudo, extended_train)

    # ====================================================================
    # ステップ3: 最良の設定で最終予測
    # ====================================================================
    print("\n" + "=" * 60)
    print("【ステップ3】最終結果サマリー")
    print("=" * 60)

    # 結果をDataFrameで表示
    results_df = pd.DataFrame(results)
    print("\n全結果:")
    print(results_df.to_string(index=False))

    # 最良の設定を選択
    if best_config is not None:
        threshold_high, threshold_low, models_best, extended_train_best = best_config

        print(f"\n最良設定:")
        print(f"  閾値: proba >= {threshold_high} or proba <= {threshold_low}")
        print(f"  Pseudo-labelサンプル数: {len(extended_train_best) - len(train)}")
        print(f"  OOF Accuracy: {best_score:.4f}")

        # 最終テスト予測
        print("\n最終テストデータ予測中...")
        final_test_predictions = predict_test(models_best, test, features)

        # 提出ファイル作成
        submission = pd.DataFrame({
            'PassengerId': test_raw['PassengerId'],
            'Perished': (final_test_predictions > 0.5).astype(int)
        })

        submission_file = 'submission_pseudo_labeling.csv'
        submission.to_csv(submission_file, index=False)

        print(f"\n提出ファイル作成: {submission_file}")
        print(f"予測Perished分布:")
        print(submission['Perished'].value_counts().sort_index())
        print(f"予測死亡率: {submission['Perished'].mean():.3f}")
    else:
        print("\n⚠ すべての設定でベースラインより悪化しました。")

    print("\n" + "=" * 60)
    print("完了！")
    print("=" * 60)
