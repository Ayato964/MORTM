import soundfile as sf

data, sr = sf.read("out/audio/datasets_small/alone.mid.wav", always_2d=True)
print(f"Sample rate: {sr}")
print(f"Shape: {data.shape}")
