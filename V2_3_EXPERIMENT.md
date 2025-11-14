# v2.3実験: Fare削除の検証（失敗）

## 実験目的

ユーザーからの質問: **「FarePerPersonとFareは両方必要？」**

検証内容: v2.2からFareを削除し、FarePerPerson単独で十分か確認

## 仮説

`FarePerPerson = Fare / TicketGroupSize`なので、FarePerPersonがあればFareは冗長かもしれない

## 実験結果

### CVスコア比較

| バージョン | OOF Accuracy | OOF AUC | 特徴量数 | 変更点 |
|-----------|-------------|---------|---------|--------|
| **v2.2** | **84.29%** | **88.52%** | 17 | 基準 |
| v2.3 | 83.84% | 88.20% | 16 | Fare削除 |
| **差分** | **-0.45pt** ❌ | **-0.32pt** ❌ | -1 | |

### Fold別スコア

```
v2.3結果:
Fold 1: Accuracy=0.8436, AUC=0.8997
Fold 2: Accuracy=0.8708, AUC=0.8850
Fold 3: Accuracy=0.7978, AUC=0.8630
Fold 4: Accuracy=0.8371, AUC=0.8860
Fold 5: Accuracy=0.8427, AUC=0.8905

OOF Accuracy: 0.8384 (83.84%)
OOF AUC: 0.8820 (88.20%)
```

v2.2との差分:
```
Fold 1: 85.48% → 84.36% (-1.12pt)
Fold 2: 87.08% → 87.08% (±0.00pt)
Fold 3: 80.34% → 79.78% (-0.56pt)
Fold 4: 83.71% → 83.71% (±0.00pt)
Fold 5: 84.83% → 84.27% (-0.56pt)
```

### Feature Importance変化

**v2.2 (Fare + FarePerPerson両方あり):**
```
Fare              873.947
FarePerPerson     906.111
合計            ~1780
```

**v2.3 (FarePerPersonのみ):**
```
FarePerPerson     938.959
```

FarePerPersonの重要度は上がったが、情報の総量は減少

## 分析

### ❌ 失敗の理由

**1. FareとFarePerPersonは補完的な情報を持つ**

- **Fare（絶対価格）**: チケットの高額さ = 社会的地位の指標
  - 高額チケット = 上級クラス = 生存率高い
  - 絶対的な価格帯の情報

- **FarePerPerson（一人当たり価格）**: グループ内での公平性
  - 同じチケットを共有する人数で割った値
  - グループサイズを考慮した相対的な価格

- **両者は異なる角度から生存率を予測**

**2. 完全冗長ではない**

- `FarePerPerson = Fare / TicketGroupSize`
- TicketGroupSizeという**第三の変数**が関わるため、完全な冗長性ではない
- FareとFarePerPersonは独立した情報を持つ

**3. SibSp/Parchとの違い**

成功した削除（v2.2）:
```
SibSp + Parch + 1 = FamilySize（完全冗長）
→ SibSp, Parchは完全に冗長なので削除できる
```

失敗した削除（v2.3）:
```
Fare / TicketGroupSize = FarePerPerson（補完的）
→ Fareの絶対値情報も重要なので削除できない
```

### HasFamilyMatch削除失敗との類似性

v2.1でHasFamilyMatch削除も失敗（-0.79pt）した理由と似ている:
- **Feature Importanceが低い ≠ 削除可能**
- **特徴量間の相互作用**が重要
- 単独では弱くても、他の特徴量と組み合わさると効果的

## 結論

### ✅ FareとFarePerPersonは両方必要

**理由:**
1. 異なる情報を提供（絶対価格 vs 相対価格）
2. 完全冗長ではない（TicketGroupSizeが関与）
3. 削除すると-0.45pt Accuracy, -0.32pt AUC悪化

### 🎯 最終推奨

**v2.2（17特徴量）が最適**
- Fare + FarePerPerson 両方を保持
- SibSp, Parchのみ削除（完全冗長性）
- OOF Accuracy: 84.29%, AUC: 88.52%

## 学んだ教訓

### 冗長性の種類を区別する

1. **完全冗長（削除可能）**
   - 例: SibSp, Parch → FamilySize
   - 関係: `FamilySize = SibSp + Parch + 1`（完全な線形関係）
   - ✅ 安全に削除できる

2. **派生だが補完的（削除不可）**
   - 例: Fare → FarePerPerson
   - 関係: `FarePerPerson = Fare / TicketGroupSize`
   - ❌ 両方必要（異なる情報を持つ）

3. **低重要度だが相互作用あり（削除不可）**
   - 例: HasFamilyMatch
   - 重要度: 最下位（5.1）
   - ❌ 他の特徴量との相互作用で貢献

### 検証の重要性

**Feature Importanceだけでは判断できない**
- 必ずCV実験で検証
- 全Foldでの影響を確認
- 派生特徴量でも独立した価値を持つ可能性

---

**結論**: v2.2（17特徴量）を最終推奨版として維持
