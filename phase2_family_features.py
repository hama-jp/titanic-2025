#!/usr/bin/env python3
"""
Phase 2-1: 家族グループ生存率特徴量

Phase 1の分析でSurnameが最重要特徴だったことを受け、
家族グループ単位での集計特徴量を追加します。

新規特徴量:
- FamilyGroupSurvivalRate: 同じ姓の家族での生存率（リーク込み）
- TicketGroupSurvivalRate: 同じチケットを持つグループでの生存率
- FamilyGroupMeanAge: 家族グループの平均年齢
- FamilyGroupMeanFare: 家族グループの平均運賃
- TicketGroupSize: チケットグループのサイズ
- FamilySurvivalKnownCount: 家族内で既知の生存/死亡数
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("Phase 2-1: 家族グループ生存率特徴量")
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
# 基本特徴量エンジニアリング（v6と同じ）
# =====================================
print("\n⚙️  Basic feature engineering...")

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
# 🆕 Phase 2-1: 家族グループ特徴量
# =====================================
print("\n🆕 Phase 2-1: Creating family group features...")

# ターゲット情報を一時的に追加（trainのみ）
full['Target_tmp'] = np.nan
full.loc[:train_len-1, 'Target_tmp'] = y_train.values

# 1. 姓ベースの家族グループ生存率（リーク込み）
surname_survival = full.groupby('Surname')['Target_tmp'].agg(['mean', 'count', 'std'])
full['FamilyGroupSurvivalRate'] = full['Surname'].map(surname_survival['mean']).fillna(0.5)
full['FamilyGroupSize'] = full['Surname'].map(surname_survival['count'])
full['FamilyGroupSurvivalStd'] = full['Surname'].map(surname_survival['std']).fillna(0)

# 2. チケットベースのグループ生存率
ticket_survival = full.groupby('Ticket')['Target_tmp'].agg(['mean', 'count'])
full['TicketGroupSurvivalRate'] = full['Ticket'].map(ticket_survival['mean']).fillna(0.5)
full['TicketGroupSize'] = full['Ticket'].map(ticket_survival['count'])

# 3. 家族グループの平均年齢・運賃
surname_age = full.groupby('Surname')['Age'].mean()
surname_fare = full.groupby('Surname')['Fare'].mean()
full['FamilyGroupMeanAge'] = full['Surname'].map(surname_age)
full['FamilyGroupMeanFare'] = full['Surname'].map(surname_fare)

# 4. チケットグループの平均年齢・運賃
ticket_age = full.groupby('Ticket')['Age'].mean()
ticket_fare = full.groupby('Ticket')['Fare'].mean()
full['TicketGroupMeanAge'] = full['Ticket'].map(ticket_age)
full['TicketGroupMeanFare'] = full['Ticket'].map(ticket_fare)

# 5. 家族内での年齢順位（若い順）
full['FamilyAgeRank'] = full.groupby('Surname')['Age'].rank(method='dense')

# 6. 家族グループでの性別構成
surname_sex_count = full.groupby('Surname')['Sex'].apply(lambda x: (x == 'female').sum())
full['FamilyFemaleCount'] = full['Surname'].map(surname_sex_count)
surname_male_count = full.groupby('Surname')['Sex'].apply(lambda x: (x == 'male').sum())
full['FamilyMaleCount'] = full['Surname'].map(surname_male_count)

# 7. 年齢と家族平均年齢の差
full['AgeDiffFromFamilyMean'] = full['Age'] - full['FamilyGroupMeanAge']

# 8. 運賃と家族平均運賃の差
full['FareDiffFromFamilyMean'] = full['Fare'] - full['FamilyGroupMeanFare']

print(f"  新規特徴量追加: 14個")

# Target Encoding (既存のカテゴリ特徴量)
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

drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
feature_cols = [col for col in full.columns if col not in drop_cols]

X_full = full[feature_cols]
X_train = X_full[:train_len]
X_test = X_full[train_len:]

print(f"  Total features: {len(feature_cols)} (v6: 49 → Phase2-1: {len(feature_cols)})")

# =====================================
# CV評価
# =====================================
print("\n📊 Cross-validation...")

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

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(lgb_model, X_train, y_train, cv=cv, scoring='accuracy')

print(f"\n  Phase 2-1 CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print(f"  Baseline (v6): 0.8406")
print(f"  Improvement: {cv_scores.mean() - 0.8406:+.4f}")

# =====================================
# 最終予測
# =====================================
print("\n🔮 Training final model and making predictions...")

lgb_model.fit(X_train, y_train)
final_pred = lgb_model.predict(X_test)

# 保存
submission = pd.DataFrame({
    'PassengerId': test_ids,
    'Perished': final_pred
})
submission.to_csv('submission_phase2_family.csv', index=False)

print(f"  Saved: submission_phase2_family.csv")
print(f"\n📈 Prediction statistics:")
print(f"  Perished=0: {(final_pred == 0).sum()} ({(final_pred == 0).mean()*100:.1f}%)")
print(f"  Perished=1: {(final_pred == 1).sum()} ({(final_pred == 1).mean()*100:.1f}%)")

# 特徴量重要度トップ15
print("\n📊 Top 15 Feature Importances:")
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': lgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print(importance_df.head(15).to_string(index=False))

print("\n✅ Phase 2-1 Complete!")
