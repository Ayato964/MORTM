# 方向タグ仕様（研究設計書 v1.3 付録B / §9.2 の凍結仕様）

**設計原則**: 真実源は `pattern_key`（残存ブロックの順序列, lossless）。主ラベルは
**ダッシュボード用の派生ビュー**であり、H1〜H5 の判定には用いない。判定に用いる
「方向別 loss」は **セグメント別損失**（B.3）で計算する。理由: 系列全体 AR CE の下では
1 サンプルが複数の条件付き方向に同時に勾配を与える（例 [PAST, CONST, META] は
continuation と analysis の両方）ため、サンプル単位の単一ラベルでは計測が壊れる。

実装対応:
- `pattern_key` = `convert_foundation.py FoundationDataMaker.convert()` が生成する
  残存ブロック名の順序列（例 `"PAST_M,SYSTEM,CONST_M,FUTURE_M"`）。META ドロップ時は
  `SYSTEM` を含まない（例 `"FUTURE_M,CONST_M"`）。EOS は系列先頭に常在し pattern_key には含めない。
- npz に per-sample の `pattern_keys`（array{i} と同順の文字列配列）として保存する。
- B.1 主ラベル・B.3 セグメント損失は pattern_key + サンプル内のブロックマーカー
  (`<SYSTEM>/<PAST_M>/<CONST_M>/<FUTURE_M>/<TAG_END>`) から決定論的に復元する。

## B.1 主ラベル（優先順・先勝ち、決定論的）

| 優先 | ラベル | 条件（present=残存、A<B = A が B より前） |
|---|---|---|
| 1 | `analysis` | META present ∧ 残存音楽ブロックの少なくとも 1 つ < META |
| 2 | `infill` | CONST present ∧ PAST < CONST ∧ FUTURE < CONST |
| 3 | `anticipation` | CONST present ∧ FUTURE < CONST（制約族より PAST は不在） |
| 4 | `continuation` | CONST present ∧ PAST < CONST ∧ FUTURE は不在または > CONST |
| 5 | `meta_gen` | CONST present ∧ CONST の前に音楽ブロックなし ∧ META < CONST |
| 6 | `uncond` | CONST present ∧ CONST の前に何もない（真の p(CONST) prefix） |
| 7 | `no_target` | CONST 削除 |
| 8 | `other` | 到達不能。E0 の網羅 assert で 1 件でも出たら仕様バグとして停止 |

直交フラグも併存保存: `meta_pos ∈ {before_const, after_some_music, absent, na}`,
`music_ctx_before_const ∈ {none, P, F, PF, na}`。

## B.2 裁定の記録（P-1〜P-3）

- **P-1**: `uncond` は「CONST の前に何もない」場合に予約（分岐 A の採用により初めて発生）。
  META のみが前にある場合は `meta_gen` と改名 — p(CONST|META) は条件付き生成であり、
  これを uncond と呼ぶ誤記を恒久的に排除する。
- **P-2**: FUTURE のみが前 = `anticipation` として独立ラベル化。infill に混ぜない
  （損失曲線の汚染防止）、other に落とさない（音楽的に固有の方向 p(過去|未来)=導入部生成、
  質量 ~5% と無視できない）。**ただし §6.1 の評価タスクには追加しない**（評価スイートは
  5 タスクで凍結済み。anticipation はセグメント損失での監視のみ）。
- **P-3**: CONST 削除は一律 `no_target`。META 位置での細分は行わない — 当該サンプルの
  META 末尾区間は B.3 により analysis 損失へ自動算入されるため、ラベル細分は不要。

## B.3 セグメント別損失（H 判定の正式な計測器）

pattern_key から各ブロック区間のトークン範囲を復元し、以下を集計する:

| 損失名 | 区間 | 算入条件（prefix = 当該区間より前の残存ブロック集合） |
|---|---|---|
| analysis 損失 | META 区間 | prefix に音楽ブロック ≥1（r_meta の新定義と一致） |
| infill 損失 | CONST 区間 | prefix ⊇ {PAST, FUTURE} |
| continuation 損失 | CONST 区間 | PAST ∈ prefix ∧ FUTURE ∉ prefix |
| anticipation 損失 | CONST 区間 | FUTURE ∈ prefix ∧ PAST ∉ prefix |
| meta_gen 損失 | CONST 区間 | prefix = {META} |
| uncond 損失 | CONST 区間 | prefix = ∅ |

E5 の損失重み λ は「analysis 損失の算入条件を満たす META 区間トークン」に適用する。
§4.3 の方向別 val loss ログはこの表の 6 系列 + 全体損失で構成する。

**注記（v1.3）**: 本表の算入条件はすべて **prefix 基準**（META が区間より後にあるサンプルは
当該区間の prefix に META を含まない）。よって `uncond 損失`・メタなし方向の各損失は
p_drop=0 のデータでも空にならない。系列レベルの META 有無で層別したい場合は
直交フラグ `meta_pos=absent` を使う。

*(v1.2 凍結 / v1.3 注記。変更時は必ず版を上げる)*
