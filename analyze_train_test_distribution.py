"""
訓練データとテストデータの分布比較分析

目的:
- OOFとPublic Scoreの乖離（-5.5pt以上）の原因を特定
- 訓練データとテストデータの分布の違いを詳細に分析
"""

import pandas as pd
import numpy as np
import re
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ====================================================================
# 特徴量エンジニアリング関数（v2.6と同じ）
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
    """全特徴量を生成する関数"""
    train = train_raw.copy()
    test = test_raw.copy()

    # Add dataset identifier
    train['Dataset'] = 'Train'
    test['Dataset'] = 'Test'

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

    return train, test

# ====================================================================
# 分析関数
# ====================================================================

def analyze_missing_values(train, test):
    """欠損値の比較"""
    print("=" * 80)
    print("欠損値の比較")
    print("=" * 80)

    columns_to_check = ['Age', 'Cabin', 'Embarked', 'Fare']

    print(f"\n{'Feature':<15} {'Train Missing %':<20} {'Test Missing %':<20} {'Difference':<15}")
    print("-" * 80)

    for col in columns_to_check:
        train_missing = train[col].isnull().mean() * 100
        test_missing = test[col].isnull().mean() * 100
        diff = test_missing - train_missing
        print(f"{col:<15} {train_missing:>18.2f}% {test_missing:>19.2f}% {diff:>14.2f}%")
    print()

def analyze_categorical_distribution(train, test):
    """カテゴリ変数の分布比較"""
    print("=" * 80)
    print("カテゴリ変数の分布比較")
    print("=" * 80)

    categorical_cols = ['Sex', 'Pclass', 'Embarked', 'Title', 'Deck', 'AgeBin', 'IsAlone', 'IsChild', 'IsMother']

    for col in categorical_cols:
        print(f"\n{col}:")
        print("-" * 80)

        train_dist = train[col].value_counts(normalize=True).sort_index() * 100
        test_dist = test[col].value_counts(normalize=True).sort_index() * 100

        # Combine to show side by side
        comparison = pd.DataFrame({
            'Train %': train_dist,
            'Test %': test_dist
        })
        comparison['Difference'] = comparison['Test %'] - comparison['Train %']
        comparison = comparison.fillna(0)

        print(comparison.to_string())

        # Chi-square test if applicable
        if len(train[col].unique()) > 1 and len(test[col].unique()) > 1:
            # Create contingency table
            train_counts = train[col].value_counts()
            test_counts = test[col].value_counts()

            # Align indices
            all_categories = sorted(set(train_counts.index) | set(test_counts.index))
            train_aligned = [train_counts.get(cat, 0) for cat in all_categories]
            test_aligned = [test_counts.get(cat, 0) for cat in all_categories]

            contingency = np.array([train_aligned, test_aligned])

            if contingency.sum() > 0 and (contingency > 0).all():
                try:
                    chi2, p_value = stats.chi2_contingency(contingency)[:2]
                    print(f"\nChi-square test: χ² = {chi2:.4f}, p-value = {p_value:.4f}")
                    if p_value < 0.05:
                        print("⚠️  統計的に有意な差あり (p < 0.05)")
                    else:
                        print("✅ 統計的に有意な差なし")
                except:
                    pass
        print()

def analyze_numerical_distribution(train, test):
    """数値変数の分布比較"""
    print("=" * 80)
    print("数値変数の分布比較")
    print("=" * 80)

    numerical_cols = ['Age', 'Fare', 'FarePerPerson', 'SibSp', 'Parch', 'FamilySize',
                      'TicketGroupSize', 'FamilyGroupSize', 'CabinGroupSize']

    print(f"\n{'Feature':<20} {'Train Mean':<15} {'Test Mean':<15} {'Train Median':<15} {'Test Median':<15} {'KS p-value':<15}")
    print("-" * 110)

    for col in numerical_cols:
        train_mean = train[col].mean()
        test_mean = test[col].mean()
        train_median = train[col].median()
        test_median = test[col].median()

        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = stats.ks_2samp(train[col].dropna(), test[col].dropna())

        print(f"{col:<20} {train_mean:>14.2f} {test_mean:>14.2f} {train_median:>14.2f} {test_median:>14.2f} {ks_pvalue:>14.4f}", end='')

        if ks_pvalue < 0.05:
            print(" ⚠️")
        else:
            print(" ✅")

    print("\n⚠️  = 統計的に有意な差あり (p < 0.05)")
    print("✅ = 統計的に有意な差なし")
    print()

def analyze_detailed_statistics(train, test):
    """詳細な統計量の比較"""
    print("=" * 80)
    print("主要特徴量の詳細統計")
    print("=" * 80)

    key_features = ['Age', 'Fare', 'FarePerPerson', 'FamilySize', 'TicketGroupSize']

    for col in key_features:
        print(f"\n{col}:")
        print("-" * 80)

        train_stats = train[col].describe()
        test_stats = test[col].describe()

        comparison = pd.DataFrame({
            'Train': train_stats,
            'Test': test_stats
        })
        comparison['Difference'] = comparison['Test'] - comparison['Train']
        comparison['Diff %'] = (comparison['Difference'] / comparison['Train'] * 100).round(2)

        print(comparison.to_string())
        print()

def analyze_survival_rate_patterns(train):
    """訓練データでの生存率パターン分析"""
    print("=" * 80)
    print("訓練データでの生存率パターン")
    print("=" * 80)

    if 'Perished' in train.columns:
        survival_col = 'Perished'
        train['Survived'] = 1 - train['Perished']
    elif 'Survived' in train.columns:
        survival_col = 'Survived'
    else:
        print("生存データが見つかりません")
        return

    categorical_cols = ['Sex', 'Pclass', 'Embarked', 'Title', 'AgeBin']

    for col in categorical_cols:
        print(f"\n{col}別の生存率:")
        print("-" * 80)

        survival_by_cat = train.groupby(col)['Survived'].agg(['mean', 'count'])
        survival_by_cat.columns = ['Survival Rate', 'Count']
        survival_by_cat['Survival Rate'] = (survival_by_cat['Survival Rate'] * 100).round(2)

        print(survival_by_cat.to_string())
        print()

def main():
    print("=" * 80)
    print("訓練データとテストデータの分布比較分析")
    print("=" * 80)
    print()

    # データ読み込み
    print("データ読み込み中...")
    train_raw = pd.read_csv('train.csv')
    test_raw = pd.read_csv('test.csv')

    print(f"訓練データ: {train_raw.shape}")
    print(f"テストデータ: {test_raw.shape}")
    print()

    # 特徴量生成
    print("特徴量生成中...")
    train, test = make_features(train_raw, test_raw)
    print("✓ 完了")
    print()

    # 1. 欠損値の比較
    analyze_missing_values(train_raw, test_raw)

    # 2. カテゴリ変数の分布比較
    analyze_categorical_distribution(train, test)

    # 3. 数値変数の分布比較
    analyze_numerical_distribution(train, test)

    # 4. 詳細な統計量
    analyze_detailed_statistics(train, test)

    # 5. 生存率パターン（訓練データのみ）
    analyze_survival_rate_patterns(train)

    # まとめ
    print("=" * 80)
    print("分析完了")
    print("=" * 80)
    print("\n重要な発見:")
    print("- ⚠️マークがついた特徴量は訓練とテストで統計的に有意な差がある")
    print("- これらの特徴量が過学習の原因である可能性が高い")
    print("- 分布が似ている特徴量（✅マーク）を重視すべき")
    print()

if __name__ == "__main__":
    main()
