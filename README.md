# Metric-Oriented Rhythmic Transformer for melodic generation!!(MORTM)

## Welcome
これはTransformerを用いた**旋律を自動生成**するモデルである。\
様々なTransformerのモデルが存在するが、このモデルの最大の特徴は**拍節を考慮し、旋律を生成する** ところである。\
使い方は「学習」と「前処理」によって異なるので、以下に例を示す。


## Preprocessing (MORT)
拍節を考慮した前処理、またそのトークナイザーを**Metric-Oriented Rhythmic Tokenizer**と命名した。\
使い方は、mortmライブラリのconvertというモジュールをimportする。