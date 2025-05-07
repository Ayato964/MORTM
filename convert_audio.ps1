# --------------------------------------------------
# MIDI→WAV（FluidSynth）→PCM16 WAV（ffmpeg）一括変換スクリプト
# --------------------------------------------------

# SoundFont と出力ディレクトリを指定
$sf2        = ".\data\sf\gu.sf2"
$inMidiDir  = ".\out\midi\datasets_small"
$outWavDir  = ".\out\audio\datasets_small"
$sampleRate = 44100          # 出力サンプルレート
$channels   = 1              # モノラル

# 必要なら出力フォルダ作成
if (-not (Test-Path $outWavDir)) {
    New-Item -ItemType Directory -Path $outWavDir | Out-Null
}

Get-ChildItem -Path $inMidiDir -Filter *.mid | ForEach-Object {
    $baseName   = $_.BaseName
    $midiPath   = $_.FullName
    $tmpWavPath = Join-Path $outWavDir ($baseName + ".wav")
    $outWavPath = Join-Path $outWavDir ($baseName + "_pcm16.wav")

    # 1) FluidSynth で 32bit float WAV を生成
    Write-Host "Synthesizing MIDI → WAV: $baseName.mid"
    fluidsynth -F $tmpWavPath -r $sampleRate -ni $sf2 $midiPath

    # 2) ffmpeg で 16bit PCM WAV に再エンコード
    Write-Host "Re-encoding to PCM16: $baseName_pcm16.wav"
    ffmpeg -y `
        -i $tmpWavPath `
        -ar $sampleRate `
        -ac $channels `
        -c:a pcm_s16le `
        $outWavPath

    # 3) （任意）元の float WAV を削除
    Remove-Item $tmpWavPath
}
