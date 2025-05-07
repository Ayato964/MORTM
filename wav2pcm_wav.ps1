# --------------------------------------------------
# float WAV → PCM16 WAV 一括変換 & 元ファイル削除
# --------------------------------------------------

# 対象ディレクトリを指定
$inputDir  = ".\out\audio\datasets_small"
# 出力サンプルレート・チャンネル
$sampleRate = 44100
$channels   = 1

# ディレクトリ内の .wav ファイルをすべて処理
Get-ChildItem -Path $inputDir -Filter "*.wav" | ForEach-Object {
    $origPath = $_.FullName
    # 一時出力ファイル名（元ファイル名 + "_tmp"）
    $tmpPath  = Join-Path $inputDir ($_.BaseName + "_tmp.wav")

    Write-Host "Converting: $($_.Name) → PCM16 ..."

    # ffmpeg で PCM16 に再エンコード
    ffmpeg -y `
        -i $origPath `
        -ar $sampleRate `
        -ac $channels `
        -c:a pcm_s16le `
        $tmpPath

    # 変換が成功したら元を削除してリネーム
    if (Test-Path $tmpPath) {
        Remove-Item $origPath
        Rename-Item   $tmpPath -NewName $_.Name
        Write-Host " Done: $($_.Name)"
    }
    else {
        Write-Warning "  Failed to convert $($_.Name)"
    }
}
