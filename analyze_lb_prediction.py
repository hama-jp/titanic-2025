#!/usr/bin/env python3
"""
LBスコア予測分析

CV 0.9921からLB（リーダーボード）スコアがどれくらい下がるか分析
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("LBスコア予測分析")
print("=" * 70)

# データ読み込み
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f"\n📊 データセット基本情報:")
print(f"  Train: {len(train)}件")
print(f"  Test: {len(test)}件")

# Surname（姓）分析
train['Surname'] = train['Name'].str.split(',').str[0]
test['Surname'] = test['Name'].str.split(',').str[0]

train_surnames = set(train['Surname'].unique())
test_surnames = set(test['Surname'].unique())
common_surnames = train_surnames & test_surnames

print(f"\n🔍 Surname（姓）の重複分析:")
print(f"  Trainユニーク姓: {len(train_surnames)}")
print(f"  Testユニーク姓: {len(test_surnames)}")
print(f"  Train-Test共通姓: {len(common_surnames)}")
print(f"  Train-Test姓重複率: {len(common_surnames)/len(test_surnames)*100:.1f}%")

# 共通姓のカバー率
train_common_count = train[train['Surname'].isin(common_surnames)].shape[0]
test_common_count = test[test['Surname'].isin(common_surnames)].shape[0]

print(f"\n  Train側の共通姓カバー率: {train_common_count/len(train)*100:.1f}% ({train_common_count}件)")
print(f"  Test側の共通姓カバー率: {test_common_count/len(test)*100:.1f}% ({test_common_count}件)")

# Ticket（チケット）分析
train_tickets = set(train['Ticket'].unique())
test_tickets = set(test['Ticket'].unique())
common_tickets = train_tickets & test_tickets

print(f"\n🎫 Ticket（チケット）の重複分析:")
print(f"  Trainユニークチケット: {len(train_tickets)}")
print(f"  Testユニークチケット: {len(test_tickets)}")
print(f"  Train-Test共通チケット: {len(common_tickets)}")
print(f"  Train-Testチケット重複率: {len(common_tickets)/len(test_tickets)*100:.1f}%")

# 共通チケットのカバー率
train_ticket_count = train[train['Ticket'].isin(common_tickets)].shape[0]
test_ticket_count = test[test['Ticket'].isin(common_tickets)].shape[0]

print(f"\n  Train側の共通チケットカバー率: {train_ticket_count/len(train)*100:.1f}% ({train_ticket_count}件)")
print(f"  Test側の共通チケットカバー率: {test_ticket_count/len(test)*100:.1f}% ({test_ticket_count}件)")

# 家族サイズ分析
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1

print(f"\n👨‍👩‍👧‍👦 FamilySize分布:")
print(f"  Train平均家族サイズ: {train['FamilySize'].mean():.2f}")
print(f"  Test平均家族サイズ: {test['FamilySize'].mean():.2f}")

# 単独乗客の割合
train_alone = (train['FamilySize'] == 1).sum()
test_alone = (test['FamilySize'] == 1).sum()

print(f"\n  Train単独乗客: {train_alone}件 ({train_alone/len(train)*100:.1f}%)")
print(f"  Test単独乗客: {test_alone}件 ({test_alone/len(test)*100:.1f}%)")

# 過学習リスク評価
print("\n" + "=" * 70)
print("🔮 LBスコア予測")
print("=" * 70)

# リスク要因
risk_factors = []

# 1. 姓の重複率が低い場合
surname_overlap_rate = test_common_count / len(test)
if surname_overlap_rate < 0.3:
    risk_factors.append(f"姓の重複率が低い ({surname_overlap_rate*100:.1f}%)")
    surname_risk = "高"
elif surname_overlap_rate < 0.5:
    surname_risk = "中"
else:
    surname_risk = "低"

# 2. チケットの重複率が低い場合
ticket_overlap_rate = test_ticket_count / len(test)
if ticket_overlap_rate < 0.1:
    risk_factors.append(f"チケット重複率が低い ({ticket_overlap_rate*100:.1f}%)")
    ticket_risk = "高"
elif ticket_overlap_rate < 0.3:
    ticket_risk = "中"
else:
    ticket_risk = "低"

# 3. CVスコアが異常に高い
cv_score = 0.9921
if cv_score > 0.95:
    risk_factors.append(f"CVスコアが異常に高い ({cv_score:.4f})")
    cv_risk = "高"
elif cv_score > 0.90:
    cv_risk = "中"
else:
    cv_risk = "低"

print(f"\n過学習リスク評価:")
print(f"  姓ベース特徴量リスク: {surname_risk}")
print(f"  チケットベース特徴量リスク: {ticket_risk}")
print(f"  CVスコアリスク: {cv_risk}")

# リスクレベル判定
if risk_factors:
    print(f"\n⚠️  検出されたリスク要因:")
    for i, factor in enumerate(risk_factors, 1):
        print(f"    {i}. {factor}")
    overall_risk = "高"
else:
    overall_risk = "低"

print(f"\n総合リスクレベル: {overall_risk}")

# LBスコア予測
print("\n" + "-" * 70)
print("📉 LBスコア予測シナリオ:")
print("-" * 70)

current_cv = 0.9921

# シナリオ1: 楽観的（家族グループ戦略が有効）
optimistic_lb = current_cv - 0.04
print(f"\n✅ 楽観的シナリオ:")
print(f"   - 仮定: Train-Test間で家族構成が類似")
print(f"   - 仮定: 家族単位の運命共同体仮説が有効")
print(f"   - 予測LB: {optimistic_lb:.4f} (CV比 -0.04)")

# シナリオ2: 中立的（適度な過学習）
neutral_lb = current_cv - 0.08
print(f"\n⚖️  中立的シナリオ (最も現実的):")
print(f"   - 仮定: 一部の家族はTrain-Testで分離")
print(f"   - 仮定: 家族特徴量は有効だが部分的に過学習")
print(f"   - 予測LB: {neutral_lb:.4f} (CV比 -0.08)")

# シナリオ3: 悲観的（強い過学習）
pessimistic_lb = current_cv - 0.13
print(f"\n❌ 悲観的シナリオ:")
print(f"   - 仮定: Train-Testで家族構成が大きく異なる")
print(f"   - 仮定: 家族グループ生存率が強く過学習")
print(f"   - 予測LB: {pessimistic_lb:.4f} (CV比 -0.13)")

# 最も可能性が高いシナリオ
print("\n" + "=" * 70)
print("🎯 推奨予測:")

# データ分析に基づく判定
if surname_overlap_rate > 0.4 and ticket_overlap_rate > 0.2:
    print(f"  最も可能性が高い: 楽観的〜中立的シナリオ")
    print(f"  予測LBスコア範囲: {optimistic_lb:.4f} 〜 {neutral_lb:.4f}")
    print(f"  スコア下落幅: -0.04 〜 -0.08")
elif surname_overlap_rate > 0.2:
    print(f"  最も可能性が高い: 中立的シナリオ")
    print(f"  予測LBスコア範囲: {neutral_lb - 0.02:.4f} 〜 {neutral_lb + 0.02:.4f}")
    print(f"  スコア下落幅: -0.06 〜 -0.10")
else:
    print(f"  最も可能性が高い: 中立的〜悲観的シナリオ")
    print(f"  予測LBスコア範囲: {neutral_lb:.4f} 〜 {pessimistic_lb:.4f}")
    print(f"  スコア下落幅: -0.08 〜 -0.13")

print("\n💡 補足:")
print("  - CVスコアはOOF予測なので、trainデータ内では健全")
print("  - ただし、家族グループ特徴量自体がリーク込みのため")
print("    testデータでの汎化性能は未知数")
print("  - 実際のLBスコアで検証することを推奨")
print("=" * 70)
