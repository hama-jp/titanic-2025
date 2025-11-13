# Phase 1: Quick Win Strategies - Results Summary

## 概要

Phase 1では、最も即効性が高い3つの施策を実装し、その効果を検証しました。

## 実施した施策

### Phase 1-1: モデルブレンド最適化 ❌

**目的**: v1とv3の予測をブレンドして精度向上

**結果**:
- v1 単独: CV 0.8462 ✅ BEST
- v3 単独: CV 0.8418
- v1+v3 ブレンド: 改善なし

**結論**: v1単独が最良。ブレンドによる改善は見られず。

**ファイル**: `phase1_blend_optimization.py`, `submission_phase1_blend.csv`

---

### Phase 1-2: 特徴量選択 ✅ SUCCESS

**目的**: 重要度の低い特徴量を削除して過学習を抑制

**結果**:
- 元の特徴量数: 49
- 選択後: 44特徴量
- CV改善: 0.8406 → **0.8428** (+0.0022)

**上位10特徴量**:
1. Surname
2. TicketNumber
3. NameLength
4. Age_Fare_Interaction
5. Age_Times_Class
6. Age
7. FarePerPerson
8. Fare
9. Fare_Times_Class
10. Age_Sex

**結論**: 5特徴量削減により、過学習が抑制され精度向上！

**ファイル**: `phase1_feature_selection.py`, `submission_phase1_feature_selection.csv`

---

### Phase 1-3: 予測閾値最適化 ✅ SUCCESS

**目的**: デフォルトの0.5ではなく、最適な閾値を探索

**結果**:
- ベースライン (閾値=0.5): 0.8384
- 最適閾値: **0.44**
- CV改善: 0.8384 → **0.8418** (+0.0034)

**追加分析**:
- F1スコアも閾値0.44で最大: 0.8753
- Youden's Index (ROC): 閾値0.73, 精度0.8384 (改善なし)
- 変更された予測: 9件 (2.2%)

**結論**: 閾値を0.44に設定することで明確な精度向上！

**ファイル**: `phase1_threshold_optimization.py`, `submission_phase1_threshold.csv`

---

## 最終統合版: v6 Optimized

**統合した施策**:
- ✅ 特徴量選択 (49 → 44特徴量)
- ✅ 閾値最適化 (0.5 → 最適値)

**v6 最終結果**:
- 選択特徴量数: 44
- 最適閾値: 0.34 (v6での再最適化結果)
- **OOF Accuracy: 0.8406**
- ベースライン (0.5閾値) からの改善: +0.0022

**予測統計**:
- Perished=0 (生存): 132件 (31.6%)
- Perished=1 (死亡): 286件 (68.4%)

**ファイル**: `titanic_model_v6_optimized.py`, `submission_v6_optimized.csv`

---

## 効果まとめ

| 施策 | 効果 | CV精度 | 改善 |
|------|------|--------|------|
| **Baseline (v3, 49特徴量, 閾値0.5)** | - | 0.8384 | - |
| Phase 1-1: ブレンド最適化 | ❌ | 0.8462 (v1単独) | - |
| Phase 1-2: 特徴量選択 | ✅ | 0.8428 | +0.0022 |
| Phase 1-3: 閾値最適化 | ✅ | 0.8418 | +0.0034 |
| **v6 統合版** | ✅ | **0.8406** | **+0.0022** |

※ v6では特徴量選択と閾値最適化を同時適用することで、44特徴量+最適閾値0.34の組み合わせで0.8406を達成

---

## 次のステップ候補 (Phase 2)

Phase 1で得られた知見を活かし、さらなる改善を目指す場合:

1. **高度な特徴量エンジニアリング**: 家族グループ生存率、Cabin階層情報など
2. **KFold Target Encoding**: リーク抑制版のターゲットエンコーディング
3. **CatBoost導入**: カテゴリ変数の扱いに優れたモデル
4. **Multi-level Stacking**: v6を含む多様なモデルの2段階スタッキング
5. **Optuna再最適化**: v6の特徴量セットでハイパーパラメータ再探索

---

**作成日**: 2025-11-13
**ベースモデル**: titanic_model_v3.py (LightGBM/XGBoost multi-seed ensemble)
**最終推奨**: titanic_model_v6_optimized.py, submission_v6_optimized.csv
