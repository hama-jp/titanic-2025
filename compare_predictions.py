#!/usr/bin/env python3
"""v2.6とOptuna浅い木版の予測を詳細比較"""

import pandas as pd
import numpy as np

# 両方の提出ファイルを読み込み
v26 = pd.read_csv('submission_restart_v2_6.csv')
optuna = pd.read_csv('submission_optuna_shallow.csv')

# PassengerIdが一致することを確認
assert (v26['PassengerId'] == optuna['PassengerId']).all(), "PassengerID mismatch!"

# 予測の差分を計算
diff = np.abs(v26['Perished'] - optuna['Perished'])

# 二値化した予測の一致率
v26_binary = (v26['Perished'] > 0.5).astype(int)
optuna_binary = (optuna['Perished'] > 0.5).astype(int)
agreement = (v26_binary == optuna_binary).sum()
disagreement = (v26_binary != optuna_binary).sum()

print("=" * 70)
print("v2.6 vs Optuna浅い木版 予測比較")
print("=" * 70)
print()

print(f"総テストサンプル数: {len(v26)}")
print()

print("【予測確率の差分統計】")
print(f"  平均差分: {diff.mean():.4f}")
print(f"  最大差分: {diff.max():.4f}")
print(f"  標準偏差: {diff.std():.4f}")
print()

print("【二値化予測（閾値=0.5）の一致率】")
print(f"  一致: {agreement} ({agreement/len(v26)*100:.1f}%)")
print(f"  不一致: {disagreement} ({disagreement/len(v26)*100:.1f}%)")
print()

print("【予測死亡率】")
print(f"  v2.6: {v26_binary.sum()}/{len(v26)} = {v26_binary.sum()/len(v26)*100:.1f}%")
print(f"  Optuna: {optuna_binary.sum()}/{len(optuna)} = {optuna_binary.sum()/len(optuna)*100:.1f}%")
print()

# 不一致のケースを詳細分析
if disagreement > 0:
    print("【不一致ケースの詳細分析】")
    disagreement_mask = v26_binary != optuna_binary

    # v2.6が死亡、Optunaが生存と予測
    v26_dead_optuna_alive = (v26_binary == 1) & (optuna_binary == 0)
    print(f"  v2.6=死亡 & Optuna=生存: {v26_dead_optuna_alive.sum()}件")

    # v2.6が生存、Optunaが死亡と予測
    v26_alive_optuna_dead = (v26_binary == 0) & (optuna_binary == 1)
    print(f"  v2.6=生存 & Optuna=死亡: {v26_alive_optuna_dead.sum()}件")
    print()

    # 不一致ケースでの確率差分の平均
    disagreement_diff = diff[disagreement_mask].mean()
    print(f"  不一致ケースの平均確率差: {disagreement_diff:.4f}")
    print()

    # 不一致のPassengerIDを表示（最初の10件）
    disagreement_ids = v26.loc[disagreement_mask, 'PassengerId'].values
    print(f"  不一致PassengerID（最初の10件）: {disagreement_ids[:10].tolist()}")
    print()

print("=" * 70)
print("結論")
print("=" * 70)
print()

if disagreement == 0:
    print("✅ 両モデルの予測は完全に一致しています")
    print("   → どちらを提出しても同じ結果になります")
elif disagreement < len(v26) * 0.05:  # 5%未満
    print(f"⚠️  わずかな差異があります（{disagreement}件、{disagreement/len(v26)*100:.1f}%）")
    print("   → テストセットでは性能差が出る可能性があります")
    print("   → より単純なOptuna版（num_leaves=11）の方が汎化性能が高い可能性")
else:
    print(f"⚠️  有意な差異があります（{disagreement}件、{disagreement/len(v26)*100:.1f}%）")
    print("   → 両方を提出して比較すべきです")

print()
print("【推奨】")
print("  両方をKaggleに提出して、Public Leaderboardスコアを比較")
print("  → より単純なモデル（Optuna num_leaves=11）が勝つ可能性あり 🎯")
print()
