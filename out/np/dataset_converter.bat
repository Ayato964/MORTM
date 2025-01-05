@echo off
setlocal enabledelayedexpansion

:: 現在のディレクトリに'turing'フォルダを作成
mkdir turing

:: 初期設定
set source_dir=datasets
set target_dir=turing
set count=0
set folder_index=1

:: 'datasets'内のファイルをループ処理
for %%f in (%source_dir%\*) do (
    set /a count+=1

    :: 5000個ごとに新しいフォルダを作成
    if !count! equ 1 (
        mkdir %target_dir%\part!folder_index!
    )

    :: ファイルを対応するフォルダに移動
    move "%%f" %target_dir%\part!folder_index!\

    :: 5000個移動したらリセット
    if !count! equ 5000 (
        set count=0
        set /a folder_index+=1
    )
)

echo 分割完了！
pause
