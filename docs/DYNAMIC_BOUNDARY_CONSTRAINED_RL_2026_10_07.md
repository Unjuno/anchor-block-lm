# 動的ブロック境界 LoRA + 制約付きRL 実験 — 2026-10-07

## 結論

学習に伴って同じ固定probe文脈の採用長 k が変化することは確認した。64文脈中、最終的に境界が移動した割合は seed 48017/48018/48019 でそれぞれ 10.9% / 6.3% / 14.1%。したがって、今回の実装は固定ブロック長ではなく、Actorの学習状態に応じてブロック境界を再配置する。

ただし、事前に定めた速度・品質条件を通過したseedは 0/3。長く出す方向へ動いたseed 48017 と 48019 では teacher risk と系列KLが悪化し、品質を改善した seed 48018 では平均 k と tokens/call が低下した。現状は速度と分布保持のトレードオフを解消できていない。

## 設計

- evaluator / 通常AR backbone: 固定
- continuation LoRA: 現在Actorが訪問した状態で、teacherの無選別 full-horizon sampleへのKDのみで更新
- length-policy LoRA: k=1..4 を選択し、teacher prefix risk を制約にしたREINFORCEで更新
- confidence/entropy/hidden features: policy入力のみ。報酬にはしない
- 各roundで現在Actorから状態を再収集し、固定dev probeを同じ乱数条件で再評価
- test split: 未使用

## 主結果

| seed | 固定probe mean k 初期→最終 | 境界移動 | probe risk Δ | tokens/call 初期→最終 | 系列KL Δ nat/token | 最終 speed/AR | 事前条件 |
|---|---:|---:|---:|---:|---:|---:|---|
| 48017 | 1.5000 → 1.7031 | 10.9% (7↑/0↓) | +0.3538 | 1.4869 → 1.7038 | +0.1378 | 1.049x | FAIL |
| 48018 | 1.5625 → 1.5312 | 6.2% (1↑/3↓) | -0.0433 | 1.5476 → 1.5178 | -0.0129 | 0.962x | FAIL |
| 48019 | 1.5156 → 1.6406 | 14.1% (7↑/2↓) | +0.1254 | 1.5088 → 1.5081 | +0.0507 | 0.927x | FAIL |

事前条件は、(1) 固定dev probeのmean k増加、(2) selected teacher riskの増加が+0.05 nat/decision以下、(3) free-running系列KLの増加が+0.02 nat/token以下、の3条件を同時に満たすこと。

## 境界移動

- seed 48017: 1->1: 25, 1->2: 1, 1->3: 6, 2->2: 32
- seed 48018: 1->1: 27, 1->2: 1, 2->1: 3, 2->2: 33
- seed 48019: 1->1: 25, 1->2: 5, 1->3: 1, 1->4: 1, 2->1: 1, 2->2: 30, 3->2: 1

seed 48017では7文脈すべてが長くなる方向へ移動し、free-running tokens/callも1.487→1.704へ増え、CPU点推定では純AR比1.049xになった。しかし系列KLは+0.138 nat/token悪化したため、低劣化高速化とは判定しない。

seed 48018では品質側は改善した（系列KL -0.0129 nat/token）が、固定probe mean kは1.5625→1.5313、tokens/callも1.548→1.518へ低下した。

seed 48019では固定probe mean kは1.5156→1.6406へ増えたが、free-running tokens/callはほぼ不変で、系列KLは+0.0507 nat/token悪化した。

## 制約RLの挙動

- seed 48017: train-only risk target=0.5488, 最終train selected risk=0.6434, 最終dual λ=0.7139
- seed 48018: train-only risk target=0.3898, 最終train selected risk=0.4519, 最終dual λ=3.6836
- seed 48019: train-only risk target=0.5318, 最終train selected risk=0.6224, 最終dual λ=3.0518

3seedすべてで最終train selected riskがtrain-only targetを上回った。したがって、この8-round設定では制約器は収束していない。これは次の診断対象であり、devを見て品質閾値を変更する根拠にはしない。

代理報酬はdual λ自体がroundごとに変化するため、異なるround間の絶対値をそのまま比較してreward hackingと断定できない。今回確認できたのは、動的境界が学習されたことと、現設定では速度とteacher分布保持を同時には満たしていないこと。

## 検証

- GitHub Actions run 37494321837 / commit `605292b7a64e8ef38e2b74243ec65db77e952ec1`
- 109 tests passed, 0 skipped; Python compile pass
- evaluator / AR backbone unchanged; continuation and gate changed
- artifact ZIP SHA-256 matched GitHub-reported digests
- `evaluation_raw.npz` から initial/final 系列KLを独立再計算し `result.json` と一致
- test split未使用

Artifact SHA-256:
- 48017: `413b172f6d0f3246ddea55fa51e751a3ab8bdc242d19d60ff5b1a97a0a3afb48`
- 48018: `154eec61f9c4c6706b7531281544839dc3ace2fe33055262d2a435a21066b37b`
- 48019: `0649dad1d65da078077e745d87d1a307970afea5fec5ca026ccd1f1929075a12`

## 限界

tiny nanoGPT・単一コーパス・CPU/no-KV-cacheでの開発実験。速度比は本番推論性能を示さない。KLは分布指標であり、人間評価の文章品質率ではない。
