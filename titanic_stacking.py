"""
Titanic Perished Prediction - Stacking Ensemble
LightGBM + XGBoost + CatBoost + ロジスティック回帰メタモデル

戦略（ご提案の通り）:
1. ベースモデル3つ（LightGBM, XGBoost, CatBoost）
   - 同じ特徴セット（v2）
   - 同じStratified K-Fold（seed=42）
   - 控えめ設定（浅め・正則化）でアンダーフィット寄り
2. OOF予測を生成（N×3行列）
3. ロジスティック回帰メタモデルで結合（L2正則化）

期待効果: v2単体より+0.5〜1.0pt改善
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# ====================================================================
# 特徴量エンジニアリング（v2と同じ）
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
    """v2の特徴量生成（完全に同じ）"""

    train = train_raw.copy()
    test = test_raw.copy()

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

    ticket_group_size = full.groupby('TicketNorm')['PassengerId'].transform('size')
    train['TicketGroupSize'] = ticket_group_size[:len(train)].values
    test['TicketGroupSize'] = ticket_group_size[len(train):].values

    family_group_size = full.groupby('FamilyID')['PassengerId'].transform('size')
    train['FamilyGroupSize'] = family_group_size[:len(train)].values
    test['FamilyGroupSize'] = family_group_size[len(train):].values

    for df in [train, test]:
        df['HasFamilyMatch'] = (df['FamilyGroupSize'] > df['FamilySize']).astype(int)

    cabin_group_size = full.groupby('CabinNorm')['PassengerId'].transform('size')
    train['CabinGroupSize'] = cabin_group_size[:len(train)].values
    test['CabinGroupSize'] = cabin_group_size[len(train):].values

    for df in [train, test]:
        df.loc[df['CabinNorm'] == 'Unknown', 'CabinGroupSize'] = 1

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

    return train, test

# ====================================================================
# ベースモデル定義
# ====================================================================

def train_lightgbm(X_train, y_train, X_val, y_val, categorical_features):
    """LightGBM（v2と同じ設定）"""
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
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=0)
        ]
    )

    return model

def train_xgboost(X_train, y_train, X_val, y_val):
    """XGBoost（控えめ設定）"""
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'learning_rate': 0.05,
        'max_depth': 3,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'seed': 42,
        'verbosity': 0
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dval, 'eval')],
        early_stopping_rounds=100,
        verbose_eval=False
    )

    return model

def train_catboost(X_train, y_train, X_val, y_val, categorical_features):
    """CatBoost（控えめ設定）"""
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=4,
        l2_leaf_reg=3,
        subsample=0.8,
        random_seed=42,
        verbose=False,
        early_stopping_rounds=100
    )

    cat_indices = [i for i, feat in enumerate(X_train.columns) if feat in categorical_features]

    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        cat_features=cat_indices,
        verbose=False
    )

    return model

# ====================================================================
# スタッキング実装
# ====================================================================

def stacking_ensemble(train_df, test_df, features, target='Perished', n_splits=5, seed=42):
    """
    スタッキングアンサンブル

    Level 0: LightGBM, XGBoost, CatBoost
    Level 1: ロジスティック回帰
    """

    print("=" * 60)
    print("スタッキングアンサンブル開始")
    print("=" * 60)

    X = train_df[features].copy()
    y = train_df[target]
    X_test = test_df[features].copy()

    # カテゴリ変数
    categorical_features = ['Sex', 'Pclass', 'Embarked', 'Title', 'Deck', 'AgeBin']
    categorical_features = [f for f in categorical_features if f in features]

    # カテゴリ型に変換
    for col in categorical_features:
        X[col] = X[col].astype('category')
        X_test[col] = X_test[col].astype('category')

    # OOF予測とテスト予測を格納
    oof_lgb = np.zeros(len(X))
    oof_xgb = np.zeros(len(X))
    oof_cat = np.zeros(len(X))

    test_lgb = np.zeros(len(X_test))
    test_xgb = np.zeros(len(X_test))
    test_cat = np.zeros(len(X_test))

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    print(f"特徴量数: {len(features)}")
    print(f"CV: {n_splits}-Fold Stratified")
    print()

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        print(f"Fold {fold}:")

        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # 1. LightGBM
        print("  LightGBM...", end=" ")
        lgb_model = train_lightgbm(X_tr, y_tr, X_val, y_val, categorical_features)
        oof_lgb[val_idx] = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
        test_lgb += lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration) / n_splits
        print("OK")

        # 2. XGBoost（カテゴリをlabel encodingする必要がある）
        print("  XGBoost...", end=" ")
        X_tr_num = X_tr.copy()
        X_val_num = X_val.copy()
        X_test_num = X_test.copy()

        # カテゴリをコード化
        for col in categorical_features:
            X_tr_num[col] = X_tr_num[col].cat.codes
            X_val_num[col] = X_val_num[col].cat.codes
            X_test_num[col] = X_test_num[col].cat.codes

        xgb_model = train_xgboost(X_tr_num, y_tr, X_val_num, y_val)
        dval = xgb.DMatrix(X_val_num)
        dtest = xgb.DMatrix(X_test_num)
        oof_xgb[val_idx] = xgb_model.predict(dval)
        test_xgb += xgb_model.predict(dtest) / n_splits
        print("OK")

        # 3. CatBoost
        print("  CatBoost...", end=" ")
        cat_model = train_catboost(X_tr, y_tr, X_val, y_val, categorical_features)
        oof_cat[val_idx] = cat_model.predict(X_val, prediction_type='Probability')[:, 1]
        test_cat += cat_model.predict(X_test, prediction_type='Probability')[:, 1] / n_splits
        print("OK")

        # Fold結果
        lgb_acc = accuracy_score(y_val, (oof_lgb[val_idx] > 0.5).astype(int))
        xgb_acc = accuracy_score(y_val, (oof_xgb[val_idx] > 0.5).astype(int))
        cat_acc = accuracy_score(y_val, (oof_cat[val_idx] > 0.5).astype(int))

        print(f"  LightGBM Acc: {lgb_acc:.4f}")
        print(f"  XGBoost Acc:  {xgb_acc:.4f}")
        print(f"  CatBoost Acc: {cat_acc:.4f}")
        print()

    # ベースモデルのOOFスコア
    print("=" * 60)
    print("ベースモデルのOOFスコア")
    print("=" * 60)

    lgb_oof_acc = accuracy_score(y, (oof_lgb > 0.5).astype(int))
    lgb_oof_auc = roc_auc_score(y, oof_lgb)
    print(f"LightGBM: Accuracy={lgb_oof_acc:.4f}, AUC={lgb_oof_auc:.4f}")

    xgb_oof_acc = accuracy_score(y, (oof_xgb > 0.5).astype(int))
    xgb_oof_auc = roc_auc_score(y, oof_xgb)
    print(f"XGBoost:  Accuracy={xgb_oof_acc:.4f}, AUC={xgb_oof_auc:.4f}")

    cat_oof_acc = accuracy_score(y, (oof_cat > 0.5).astype(int))
    cat_oof_auc = roc_auc_score(y, oof_cat)
    print(f"CatBoost: Accuracy={cat_oof_acc:.4f}, AUC={cat_oof_auc:.4f}")

    # メタモデル（ロジスティック回帰）
    print("\n" + "=" * 60)
    print("メタモデル: ロジスティック回帰")
    print("=" * 60)

    # OOF予測をスタック
    oof_stack = np.column_stack([oof_lgb, oof_xgb, oof_cat])
    test_stack = np.column_stack([test_lgb, test_xgb, test_cat])

    # ロジスティック回帰メタモデル（L2正則化）
    meta_model = LogisticRegression(C=1.0, random_state=seed, max_iter=1000)
    meta_model.fit(oof_stack, y)

    # メタモデル予測
    meta_oof_pred = meta_model.predict_proba(oof_stack)[:, 1]
    meta_test_pred = meta_model.predict_proba(test_stack)[:, 1]

    # メタモデルスコア
    meta_oof_acc = accuracy_score(y, (meta_oof_pred > 0.5).astype(int))
    meta_oof_auc = roc_auc_score(y, meta_oof_pred)

    print(f"スタッキング: Accuracy={meta_oof_acc:.4f}, AUC={meta_oof_auc:.4f}")
    print(f"\nメタモデル係数: LightGBM={meta_model.coef_[0][0]:.3f}, XGBoost={meta_model.coef_[0][1]:.3f}, CatBoost={meta_model.coef_[0][2]:.3f}")

    print("\n" + "=" * 60)
    print("v2単体との比較")
    print("=" * 60)
    print(f"v2単体 (LightGBM): {lgb_oof_acc:.4f}")
    print(f"スタッキング:      {meta_oof_acc:.4f}")
    print(f"改善:             {(meta_oof_acc - lgb_oof_acc)*100:+.2f}pt")

    return meta_oof_pred, meta_test_pred, oof_stack, test_stack

# ====================================================================
# メイン実行
# ====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Titanic Stacking Ensemble")
    print("LightGBM + XGBoost + CatBoost + ロジスティック回帰")
    print("=" * 60)
    print()

    # データ読み込み
    train_raw = pd.read_csv('train.csv')
    test_raw = pd.read_csv('test.csv')

    # 特徴量生成（v2と同じ）
    train, test = make_features(train_raw, test_raw)

    # v2の特徴量
    features = [
        'Age', 'Fare', 'FarePerPerson',
        'SibSp', 'Parch', 'FamilySize',
        'TicketGroupSize', 'FamilyGroupSize', 'CabinGroupSize',
        'Sex', 'Pclass', 'Embarked', 'Title', 'Deck', 'AgeBin',
        'IsAlone', 'IsChild', 'IsMother', 'HasFamilyMatch'
    ]

    # スタッキング実行
    meta_oof_pred, meta_test_pred, oof_stack, test_stack = stacking_ensemble(
        train, test, features, target='Perished', n_splits=5, seed=42
    )

    # 提出ファイル作成
    submission = pd.DataFrame({
        'PassengerId': test_raw['PassengerId'],
        'Perished': (meta_test_pred > 0.5).astype(int)
    })

    submission_file = 'submission_stacking.csv'
    submission.to_csv(submission_file, index=False)

    print(f"\n提出ファイル作成: {submission_file}")
    print(f"予測Perished分布:")
    print(submission['Perished'].value_counts().sort_index())
    print(f"予測死亡率: {submission['Perished'].mean():.3f}")

    print("\n" + "=" * 60)
    print("完了！")
    print("=" * 60)
