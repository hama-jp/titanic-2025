"""
Titanic Perished Prediction - Model Ensemble
v2.6ベース + LightGBM + XGBoost + CatBoost アンサンブル

戦略:
1. v2.6の特徴量セットを使用
2. 3つの異なるGBDTアルゴリズムを訓練
   - LightGBM: 浅めの木で過学習を抑制
   - XGBoost: より強力な正則化
   - CatBoost: カテゴリ変数の自動処理
3. OOFスコアに基づいて重み付き平均
4. CVスコアを比較
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

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

    print("特徴量生成: Title")
    for df in [train, test]:
        df['Title'] = df['Name'].apply(extract_title)

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

    print("\n【train+test結合でグループサイズ計算】")

    print("1. TicketGroupSize")
    for df in [train, test]:
        df['TicketNorm'] = df['Ticket'].apply(normalize_ticket)

    full = pd.concat([train, test], sort=False)
    ticket_group_size = full.groupby('TicketNorm')['PassengerId'].transform('size')
    train['TicketGroupSize'] = ticket_group_size[:len(train)].values
    test['TicketGroupSize'] = ticket_group_size[len(train):].values

    print("2. FamilyGroupSize")
    for df in [train, test]:
        df['FamilyID'] = df['Surname'] + '_' + df['FamilySize'].astype(str)

    full = pd.concat([train, test], sort=False)
    family_group_size = full.groupby('FamilyID')['PassengerId'].transform('size')
    train['FamilyGroupSize'] = family_group_size[:len(train)].values
    test['FamilyGroupSize'] = family_group_size[len(train):].values

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

    print("4. FarePerPerson")
    for df in [train, test]:
        df['FarePerPerson'] = df['Fare'] / df['TicketGroupSize']

    print("\n特徴量生成: Deck")
    for df in [train, test]:
        df['Deck'] = df['Cabin'].apply(extract_deck)

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
# 2. モデル訓練関数
# ====================================================================

def train_lightgbm(X_train, y_train, X_val, y_val, categorical_features):
    """LightGBMモデルを訓練"""
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
        'seed': 42
    }

    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
    val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=categorical_features, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        valid_names=['valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=0)
        ]
    )

    return model

def train_xgboost(X_train, y_train, X_val, y_val):
    """XGBoostモデルを訓練"""
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'learning_rate': 0.03,
        'max_depth': 4,
        'min_child_weight': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'seed': 42,
        'verbosity': 0,
        'enable_categorical': True
    }

    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dval, 'valid')],
        early_stopping_rounds=100,
        verbose_eval=False
    )

    return model

def train_catboost(X_train, y_train, X_val, y_val, categorical_features_idx):
    """CatBoostモデルを訓練"""
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=4,
        l2_leaf_reg=0.1,
        subsample=0.8,
        colsample_bylevel=0.8,
        min_data_in_leaf=20,
        random_seed=42,
        verbose=0,
        early_stopping_rounds=100,
        cat_features=categorical_features_idx
    )

    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        verbose=False
    )

    return model

def train_ensemble(train_df, features, categorical_features, target='Perished', n_splits=5, seed=42):
    """3つのモデルでアンサンブル訓練"""

    print("=" * 60)
    print("アンサンブルモデル訓練開始")
    print("=" * 60)

    X = train_df[features].copy()
    y = train_df[target]

    # カテゴリ変数のインデックスを取得
    categorical_features_idx = [i for i, f in enumerate(features) if f in categorical_features]

    # カテゴリ変数をカテゴリ型に変換
    for col in categorical_features:
        X[col] = X[col].astype('category')

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # 各モデルのOOF予測を保存
    oof_lgb = np.zeros(len(X))
    oof_xgb = np.zeros(len(X))
    oof_cat = np.zeros(len(X))

    models_lgb = []
    models_xgb = []
    models_cat = []

    print(f"特徴量数: {len(features)}")
    print(f"カテゴリ変数: {categorical_features}")
    print(f"\nCV: {n_splits}-Fold Stratified")
    print("-" * 60)

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        print(f"\nFold {fold}:")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # LightGBM
        print("  - LightGBM訓練中...", end=" ")
        model_lgb = train_lightgbm(X_train, y_train, X_val, y_val, categorical_features)
        pred_lgb = model_lgb.predict(X_val, num_iteration=model_lgb.best_iteration)
        oof_lgb[val_idx] = pred_lgb
        models_lgb.append(model_lgb)
        acc_lgb = accuracy_score(y_val, (pred_lgb > 0.5).astype(int))
        auc_lgb = roc_auc_score(y_val, pred_lgb)
        print(f"Acc={acc_lgb:.4f}, AUC={auc_lgb:.4f}")

        # XGBoost
        print("  - XGBoost訓練中...", end=" ")
        model_xgb = train_xgboost(X_train, y_train, X_val, y_val)
        dval = xgb.DMatrix(X_val, enable_categorical=True)
        pred_xgb = model_xgb.predict(dval)
        oof_xgb[val_idx] = pred_xgb
        models_xgb.append(model_xgb)
        acc_xgb = accuracy_score(y_val, (pred_xgb > 0.5).astype(int))
        auc_xgb = roc_auc_score(y_val, pred_xgb)
        print(f"Acc={acc_xgb:.4f}, AUC={auc_xgb:.4f}")

        # CatBoost
        print("  - CatBoost訓練中...", end=" ")
        model_cat = train_catboost(X_train, y_train, X_val, y_val, categorical_features_idx)
        pred_cat = model_cat.predict_proba(X_val)[:, 1]
        oof_cat[val_idx] = pred_cat
        models_cat.append(model_cat)
        acc_cat = accuracy_score(y_val, (pred_cat > 0.5).astype(int))
        auc_cat = roc_auc_score(y_val, pred_cat)
        print(f"Acc={acc_cat:.4f}, AUC={auc_cat:.4f}")

    print("\n" + "=" * 60)
    print("各モデルのOOFスコア")
    print("=" * 60)

    # 各モデルのOOFスコアを計算
    oof_acc_lgb = accuracy_score(y, (oof_lgb > 0.5).astype(int))
    oof_auc_lgb = roc_auc_score(y, oof_lgb)
    print(f"LightGBM: Accuracy={oof_acc_lgb:.4f}, AUC={oof_auc_lgb:.4f}")

    oof_acc_xgb = accuracy_score(y, (oof_xgb > 0.5).astype(int))
    oof_auc_xgb = roc_auc_score(y, oof_xgb)
    print(f"XGBoost:  Accuracy={oof_acc_xgb:.4f}, AUC={oof_auc_xgb:.4f}")

    oof_acc_cat = accuracy_score(y, (oof_cat > 0.5).astype(int))
    oof_auc_cat = roc_auc_score(y, oof_cat)
    print(f"CatBoost: Accuracy={oof_acc_cat:.4f}, AUC={oof_auc_cat:.4f}")

    # 重み付き平均（AUCスコアに基づく）
    print("\n" + "=" * 60)
    print("アンサンブル（重み付き平均）")
    print("=" * 60)

    # AUCスコアに基づいて重みを計算
    total_auc = oof_auc_lgb + oof_auc_xgb + oof_auc_cat
    weight_lgb = oof_auc_lgb / total_auc
    weight_xgb = oof_auc_xgb / total_auc
    weight_cat = oof_auc_cat / total_auc

    print(f"重み: LightGBM={weight_lgb:.3f}, XGBoost={weight_xgb:.3f}, CatBoost={weight_cat:.3f}")

    # アンサンブルOOF予測
    oof_ensemble = weight_lgb * oof_lgb + weight_xgb * oof_xgb + weight_cat * oof_cat
    oof_acc_ensemble = accuracy_score(y, (oof_ensemble > 0.5).astype(int))
    oof_auc_ensemble = roc_auc_score(y, oof_ensemble)

    print(f"\nアンサンブルOOF: Accuracy={oof_acc_ensemble:.4f}, AUC={oof_auc_ensemble:.4f}")
    print("=" * 60)

    return {
        'models_lgb': models_lgb,
        'models_xgb': models_xgb,
        'models_cat': models_cat,
        'weights': (weight_lgb, weight_xgb, weight_cat),
        'oof_predictions': {
            'lgb': oof_lgb,
            'xgb': oof_xgb,
            'cat': oof_cat,
            'ensemble': oof_ensemble
        },
        'oof_scores': {
            'lgb': (oof_acc_lgb, oof_auc_lgb),
            'xgb': (oof_acc_xgb, oof_auc_xgb),
            'cat': (oof_acc_cat, oof_auc_cat),
            'ensemble': (oof_acc_ensemble, oof_auc_ensemble)
        }
    }

def predict_test_ensemble(ensemble_dict, test_df, features, categorical_features):
    """アンサンブルでテストデータを予測"""
    X_test = test_df[features].copy()

    # カテゴリ変数のインデックスを取得
    categorical_features_idx = [i for i, f in enumerate(features) if f in categorical_features]

    # カテゴリ変数をカテゴリ型に変換
    for col in categorical_features:
        X_test[col] = X_test[col].astype('category')

    models_lgb = ensemble_dict['models_lgb']
    models_xgb = ensemble_dict['models_xgb']
    models_cat = ensemble_dict['models_cat']
    weight_lgb, weight_xgb, weight_cat = ensemble_dict['weights']

    # 各モデルの予測
    pred_lgb = np.zeros(len(X_test))
    for model in models_lgb:
        pred_lgb += model.predict(X_test, num_iteration=model.best_iteration)
    pred_lgb /= len(models_lgb)

    pred_xgb = np.zeros(len(X_test))
    dtest = xgb.DMatrix(X_test, enable_categorical=True)
    for model in models_xgb:
        pred_xgb += model.predict(dtest)
    pred_xgb /= len(models_xgb)

    pred_cat = np.zeros(len(X_test))
    for model in models_cat:
        pred_cat += model.predict_proba(X_test)[:, 1]
    pred_cat /= len(models_cat)

    # 重み付き平均
    pred_ensemble = weight_lgb * pred_lgb + weight_xgb * pred_xgb + weight_cat * pred_cat

    return pred_ensemble

# ====================================================================
# 3. メイン実行
# ====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Titanic Perished Prediction - Model Ensemble")
    print("v2.6ベース + LightGBM + XGBoost + CatBoost")
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

    categorical_features = ['Sex', 'Pclass', 'Embarked', 'Title', 'Deck', 'AgeBin']

    print(f"\n使用特徴量: {len(features)}個")
    print()

    # アンサンブルモデル訓練
    ensemble_dict = train_ensemble(
        train, features, categorical_features,
        target='Perished', n_splits=5, seed=42
    )

    # テスト予測
    print("\nテストデータ予測中...")
    test_predictions = predict_test_ensemble(ensemble_dict, test, features, categorical_features)

    # 提出ファイル作成
    submission = pd.DataFrame({
        'PassengerId': test_raw['PassengerId'],
        'Perished': (test_predictions > 0.5).astype(int)
    })

    submission_file = 'submission_ensemble.csv'
    submission.to_csv(submission_file, index=False)

    print(f"\n提出ファイル作成: {submission_file}")
    print(f"予測Perished分布:")
    print(submission['Perished'].value_counts().sort_index())
    print(f"予測死亡率: {submission['Perished'].mean():.3f}")

    print("\n" + "=" * 60)
    print("最終結果サマリー")
    print("=" * 60)
    print("\n各モデルのOOFスコア:")
    for model_name in ['lgb', 'xgb', 'cat', 'ensemble']:
        acc, auc = ensemble_dict['oof_scores'][model_name]
        label = {'lgb': 'LightGBM', 'xgb': 'XGBoost', 'cat': 'CatBoost', 'ensemble': 'アンサンブル'}[model_name]
        print(f"  {label:12s}: Accuracy={acc:.4f}, AUC={auc:.4f}")

    print("\n重み:")
    weight_lgb, weight_xgb, weight_cat = ensemble_dict['weights']
    print(f"  LightGBM: {weight_lgb:.3f}")
    print(f"  XGBoost:  {weight_xgb:.3f}")
    print(f"  CatBoost: {weight_cat:.3f}")

    print("\n" + "=" * 60)
    print("完了！")
    print("=" * 60)
