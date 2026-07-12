# TEST セット凍結 manifest (testset-v1, 研究設計書 v1.6 §4.2/§6.1)

2026-07-13 凍結。データ本体は git 外(data/paper/)のため、ビルダースクリプト(git)+本 md5 で固定。
再現: `make_paper_split.py`→`make_paper_indices.py`→`make_paper_testtask.py`(全て seed=42, 決定論)。

## split (docs/splits/, git管理)
- train: 217,222 songs  md5=6fc92cc456ab7b3ea0165fdf069de501
- val: 2,199 songs  md5=9c6f93acc24076dc852a3ab2869b2a81
- test: 2,288 songs  md5=3cd6ee5e5e8eb1097a19f42ce47fbc86

## リークフリー学習索引 (data/paper/{A1,A2})
- A1/200M/train.json: 45,323 seq  md5=c81d78184536462c323a6fcc353b0889
- A1/400M/train.json: 89,978 seq  md5=381436a74e0372583b30704acc604fad
- A1/800M/train.json: 180,359 seq  md5=6d4b6f36858a8fc5440393d091c6b0f5
- A1/1.6B/train.json: 361,652 seq  md5=5f289e480cb0cd9caf0ed544c2a7450c
- A1/3.2B/train.json: 722,514 seq  md5=ff84a4da4d9f0eea56ce42a4f0975cee
- A1/val.json: 26,388 seq  md5=82af5f143ef9268eb46641f2e94c012a  (A2/test=TEST-SEQ)
- A1/test.json: 27,456 seq  md5=0ec83508617f01c2c13c5ae56d1be755  (A2/test=TEST-SEQ)
- A2/200M/train.json: 30,325 seq  md5=6dd10ae080e6e0156ebd1ed4e6406dbd
- A2/400M/train.json: 60,079 seq  md5=d2e4d9a4d1801b8b6a468aa40ac0e5f6
- A2/800M/train.json: 121,213 seq  md5=74127f2dac8d561649ddfe701dd04ec6
- A2/1.6B/train.json: 242,120 seq  md5=af9430ae2063e709de3fcc7e735e5270
- A2/3.2B/train.json: 484,887 seq  md5=2f38dcf15ba9654952971415b71fbca1
- A2/val.json: 25,884 seq  md5=503cd2f59468988235e3503e57330f04  (A2/test=TEST-SEQ)
- A2/test.json: 26,880 seq  md5=5e2650a35cc12b2c892970c79ab7493e  (A2/test=TEST-SEQ)

## TEST-TASK (data/paper/TEST-TASK/, 各N=1000窓, 曲重複なし)
- infill.jsonl: 1,000 窓  md5=9fd2a6befe528e5a1a6c7bb6c72c11d8
- continuation.jsonl: 1,000 窓  md5=30d70c1ef913e18842d1d233e7c0cd0f
- condgen.jsonl: 1,000 窓  md5=634b58355db048edd5faf82b5dab4668
- uncond.jsonl: 1,000 窓  md5=1badc74416d9969d21427d6648137c88
- analysis_key.jsonl: 1,000 窓  md5=053147c90e1b822ac5a5279a8d3744cf
- analysis_dense.jsonl: 1,000 窓  md5=053147c90e1b822ac5a5279a8d3744cf
