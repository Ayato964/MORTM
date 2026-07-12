# MORTM 論文プログラム 進捗レポート（設計書 v1.3）

branch: `paper/blockorder-prereg` / 最終更新: 2026-07-11

## ★v1.4 裁定（META ドロップ降格・A1再生成不要）
CFG 同型の根拠は数学的誤り(撤回): META 後置サンプルでも CONST 区間は全確率の法則で
Σ_m p(CONST|prefix,m)p(m|prefix)=真の meta-free 条件付き/周辺に厳密一致し、位置的自由だけで
meta-free 方向は被覆済み。ドロップは分布情報を足さない。→ **A1 は p_drop=0(元実装=既存v5と同一分布)に復帰**。
META ドロップは **E5 の ablation 変数**に降格(config フラグのみ)。**A1 データ全再生成は不要**。
§3.4 の系「META の位置的自由が分析方向と meta-free 方向を同時被覆」を理論節に追加。

実行順(§12): `9.6→9.12 → 9.0/9.2/付録B/9.5(データ影響)→ 全再生成 → 9.1/9.3/9.4 → E0 → 凍結 → E1…`

## 完了 ✅

| 項目 | 内容 | 成果物 |
|---|---|---|
| 9.6 | 再現性基盤。requirements固定(+cu128注記)、set_seed(検証済) | `requirements.txt`, `mortm/utils/repro.py` |
| — | **重大修正**: site-packagesの古い非editable `mortm4.7` コピーがリポ外実行時に現行コードをshadow → `pip install -e .`(4.9)で恒久排除 | (環境) |
| 9.12 | 往復テスト。**pitch完全可逆(0/26256)**、shift/durationは仕様の量子化±1丸め。粗い情報破壊なし | `tests/test_tokenizer_roundtrip.py`, `docs/E0_roundtrip_note.md` |
| 9.0 | META独立ドロップ config フラグ。**v1.4: default p_drop=0(A1元分布)、E5専用に降格** | `convert_foundation.py` |
| 9.2 | pattern_keys を新規生成npzに保存(index揃え)。**既存v5はマーカーからロード時復元可(検証済)** | `convert_foundation.py` |
| 付録B | 方向タグ仕様を凍結コミット | `docs/direction_tag_spec.md` |
| 9.5 | 拍子検査(4/4以外除外、無記載は4/4扱い)。**既存v5生成時に既適用済み**。両経路対応に補強 | `convert.py` |

## データ状況(v1.4後・訂正版)★両アーム完全準備済み=データ生成不要
- **A1** = `/media/.../MORTM/pre_train/ver5/music` (2,660,508 npz, 4/4済, p_drop=0=元分布)。pattern_keyはarray列から復元可。
- **A2** = `/home/takaaki-nagoshi/data/scaling/ver5_noaug/music` (**2,604,288 npz**, 固定順[SYSTEM,PAST,CONST,FUTURE]=disable_block_augment 検証済)。※先に見た外部38npzは残骸で誤認だった。
- **JSON予算索引**: A1=`json_v5/{200M,400M,800M,1.6B,3.2B}` / A2=`json_v5_noaug/同`。eval.jsonも両方あり。全て存在。
- → **E1の両アームはデータ・索引とも全予算で準備完了。基盤データ生成は一切不要。**

## E0 監査 (進行中, `experiments/e0_audit.py` → `docs/E0_audit_report.md`)
- ✅ **E0-2 Π/r_meta/網羅**: A1 r_meta=0.638・33パターン・other=0 / A2 r_meta=0.000・1パターン・other=0。統制前提を実データで確認。
- ✅ **E0-1 拍子除外率**: rawコーパス=`GMD/training`(680,244 MIDI)。非4/4除外≈33%・無記載7.5%(4/4扱い)。サンプル推定(4000)、フル走査で確定可。
- ✅ **E0-3 勾配水没**: META分析位置トークン率=0.615%(>0.5%閾値→E5損失重み既定OFF可、僅差)。
- ✅ **E0-4 反復会計**: 3.2Bで A1=724k系列/214,807曲(3.37/曲) vs A2=483k系列/196,769曲(2.46/曲)。§8.1交絡を定量化。
- ✅ **E0-5 出所**: key=music21 KrumhanslSchmuckler(循環→パイプライン一致率呼称) / density=決定論。⏳キー人手検証200曲=ユーザ作業。
- ✅ **E0-6 タグ/終端**: ブロックタグはスパン/ギャップ非保持→CONST削除の偽隣接リスクはA3bで監視。
- ✅ **E0-7 META目録(v1.6)**: 基盤META = instrument/density/key のみ100%。**length(GMC)・genre・chord は0%＝不在**。
  → 有効分析属性は **key(主)+density(従)** の2つのみ。chord/genre/**length**は除外(E0-7恒久ルール:目録外禁止)。
  ★v1.6§E3は小節数を従指標に残すが、E0-7実測で不在→要裁定(length除外が筋)。

## v1.6 反映事項
- genre: E3/H1/H3から除外→探索的E3-Xに分離(GMD基盤にラベル無し)。§6.1条件付き生成・§6.2遵守・§7 H1・Fig.3から除去。
- ★E0-7で length(小節数) も不在判明 → **裁定A: length除外**確定(2026-07-12)。**分析スイート=key(主)+density(従)の2つ**。

## E1に向けた残コード(自走中)
- ✅ **B.3 方向別セグメント損失** (`mortm/eval/direction_loss.py`): マーカーから区間抽出→付録B.3バケツ別マスク。
  実データ検証で **A1=6方向全被覆(analysis1.7/infill14.5/continuation25.1/anticipation6.6/meta_gen18.4/uncond33.7%) vs A2=continuation100%(他0)** を確認＝テーゼを損失被覆レベルで裏付け。E1でval loss層別に使用。
- ✅ **9.3 protocol_builder** (`mortm/eval/protocol_builder.py`): 5タスクP形式プロンプトを決定論構築。検証済(全タスク正しい打ち切り・0混入なし・condgen=key+densityのみ=裁定A準拠)。G形式(SFT用)は E2/E3 時に追加予定。
- ✅ **9.4 metrics** (`mortm/eval/metrics.py`): 形式妥当性(文法違反/horizon)、seam(S1境界音程/S2 IOI/S4密度段差; S3=chord不在でN/A)、分布(pitchclass JS/音価JS/3-gram重複/8-gram剽窃)、ブートCI。numpy自前JS・W1(検証: self=0/直交=1)。実データ検証済。
- ✅ **9.1 fullFT(E2)** (`train_fullft.py` + `train.py`/`config.py`): E2=既存`train_mortm`+`load_model_directory`(A2起点)+`checkpoint_percents`。LoRA不使用(§4.4-1)。%途中保存はopt-inフック(E1無影響, 検証済)。
- ⏳ protocol_builder G形式(E2/E3) / 評価ドライバ(要学習済A1/A2) / train_head(9.8,E3) / T-noBar(9.7,E4)
## 凍結ゲート(進行)
- ✅ **§7閾値凍結(概念的に確定)**: 各閾値の論理/文献背景(H2=Bavarian FIM, 他=効果量/等価性の標準)をユーザ承認→凍結。※git tag `prereg-v1` は現作業のcommit要(ユーザ判断待ち, 自動commitしない)。
- ✅ **曲単位98/1/1 split構築** `scaling/make_paper_split.py` → `docs/splits/{train,val,test}_songs.txt`(git管理)。train217,222/val2,199/test2,288、重複0・決定論・曲単位(リーク防止)。
- 🚩 **要対応(E1前)**: 既存`json_v5`/`json_v5_noaug`索引は旧split基準→**新splitのtrain曲のみで学習索引を作り直す**必要(test/valリーク防止)。npz再生成不要、索引再構築のみ。
- ⏳ TEST-SEQ(test曲・時系列全ブロック) / TEST-TASK(test曲・protocol_builder窓 各N=1000) 構築 → 凍結`testset-v1`。
- ⏳ **キー200曲検証**: サンプル`docs/splits/key_verify_200_hashes.txt`確定。pipeline key付与TSVをバックグラウンド生成中→`docs/splits/key_verify_200.tsv`。ユーザは human_key/match 列を記入。
- ~~cb57431~~ = 無視でOK(ユーザ確定)。

## 次 ⏭️
- 9.1 フルFTトレーナ(E2用) / 9.3 protocol_builder / 9.4 metrics / B.3セグメント損失(pattern_key復元器, §4.3方向別val lossに必須)
- §7凍結(tag `prereg-v1`) → TEST-SEQ/TEST-TASK凍結(tag `testset-v1`) → E1(§4.3 LRスイープ→S×3/M×1)

## 未解決
- 設計書基準 `cb57431` vs ローカル `3113ff3` のコミット不一致(prereg-v1凍結前に要解消)

## 規約(要遵守)
- E2はLoRA禁止=フルFT必須。テスト確定後の指標差替禁止。チェリーピッキング禁止。
- A1/A2差は順序・部分集合分布のみ(タグ・META語彙は同一提示)。方向別lossは付録B.3セグメント損失で計測。
