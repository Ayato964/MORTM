# 🎵 MORTM: Metric-Oriented Rhythmic Transformer for Music Generation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**MORTM (Metric-Oriented Rhythmic Transformer for Music Generation)** is a Transformer-based melody generation model that focuses on the **metric structure** of music. It generates musical sequences autoregressively, one bar at a time, while preserving rhythmic consistency. MORTM also includes `V_MORTM` for audio-based generation and `BERTM` for music classification tasks.

## ✨ Features

* **Bar-level Autoregressive Generation**: Normalizes each bar to 64 ticks and generates one bar at a time.
* **High-Quality Music Generation**: Uses a custom tokenizer to capture musical structure (pitch, duration, relative timing, bars), leading to coherent outputs.
* **Efficient Transformer Architecture**:

  * **Decoder-Only (GPT-style)**: Optimized for autoregressive generation.
  * **FlashAttention2 & ALiBi**: Memory-efficient, high-speed attention with excellent long-sequence generalization.
  * **Mixture of Experts (MoE)**: Sparsely activated FFN layers for higher capacity.
* **Structured Tokenization**: Tokens for `Pitch`, `Duration`, `Position` and structural tokens like `<SME>`, `<TS>`, `<TE>`. Normalizes to 96 ticks per bar.
* **Multimodal (`V_MORTM`)**: Processes audio features (Mel spectrograms) directly.
* **Classification (`BERTM`)**: A BERT-like encoder for music classification tasks.
* **Versatile Applications**: Melody generation, improvisation assistance, education, human-AI co-creation, audio style transfer.

## 🚀 Why MORTM?

* **State-of-the-Art**: Combines FlashAttention2, MoE, and ALiBi.
* **Musical Understanding**: Custom tokenizer captures core musical elements.
* **Scalability**: Supports diverse styles and long sequences.
* **Audio Domain**: `V_MORTM` for richer audio-based generation.
* **Modular**: Easy prototyping and comparative experiments.

## 🛠️ Installation

### Prerequisites

* Python 3.8+
* NVIDIA GPU (for FlashAttention2)
* CUDA Toolkit (compatible with PyTorch)

### 1. Install PyTorch

Follow instructions at [pytorch.org](https://pytorch.org/get-started/locally/). Example:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Install FlashAttention2

```bash
pip install flash-attn --no-build-isolation
```

### 3. Install Other Dependencies

```bash
pip install numpy einops pretty_midi midi2audio soundfile torchaudio PyYAML
```

* `midi2audio` requires FluidSynth and a soundfont (e.g., `.sf2`).

### 4. Optional: Gmail Notifications

```bash
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
```

Requires OAuth2 setup (`client_secret.json`).

## ⚡ Quick Start

### Data Preparation

Convert MIDI to tokenized `.npz`:

```python
from mortm.train.tokenizer import Tokenizer, get_token_converter, TO_TOKEN
from mortm.convert import MIDI2Seq

tokenizer = Tokenizer(music_token=get_token_converter(TO_TOKEN))
converter = MIDI2Seq(tokenizer, "midi_dir", "your_midi.mid", program_list=[0], split_measure=12)
converter.convert()
converter.save("output_npz_dir")

# Save tokenizer vocabulary
tokenizer.save("vocab_output_dir")
```

## Inference

### MORTM: Melody Generation

```python
import torch
import numpy as np
from mortm.models.mortm import MORTM, MORTMArgs
from mortm.train.tokenizer import Tokenizer, get_token_converter, TO_MUSIC
from mortm.de_convert import ct_token_to_midi
from mortm.models.modules.progress import _DefaultLearningProgress

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = Tokenizer(music_token=get_token_converter(TO_MUSIC), load_data="vocab_list.json")
args = MORTMArgs("configs/models/mortm/A.json")
model = MORTM(progress=_DefaultLearningProgress(), args=args)
model.load_state_dict(torch.load("trained_mortm.pth", map_location=DEVICE))
model.to(DEVICE).eval()

seed_ids = torch.tensor([tokenizer.get("<GEN>"), tokenizer.get("<TS>")], device=DEVICE)
with torch.no_grad():
    _, full_seq = model.top_p_sampling_measure(seed_ids, p=0.95, max_measure=8, temperature=0.7)

ct_token_to_midi(tokenizer, full_seq, "generated_melody.mid", program=0, tempo=120)
```

### BERTM: Music Classification

```python
import torch
import numpy as np
import torch.nn.functional as F
from mortm.models.bertm import BERTM, MORTMArgs as BERTMArgs
from mortm.train.tokenizer import Tokenizer, get_token_converter, TO_MUSIC
from mortm.models.modules.progress import _DefaultLearningProgress

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = Tokenizer(music_token=get_token_converter(TO_MUSIC), load_data="vocab_list.json")
args = BERTMArgs("configs/models/bertm/class_file.json")
model = BERTM(progress=_DefaultLearningProgress(), args=args)
model.load_state_dict(torch.load("trained_bertm.pth", map_location=DEVICE))
model.to(DEVICE).eval()

input_npz = np.load("input_music.npz")['array1']
input_ids = torch.tensor(input_npz, dtype=torch.long, device=DEVICE).unsqueeze(0)
with torch.no_grad():
    logits = model(input_ids)
    probs = F.softmax(logits, dim=-1)
pred = "Human" if probs.argmax() == 0 else "AI"
print(f"Prediction: {pred}, Probabilities: {probs.squeeze().tolist()}")
```

## Training

### Train MORTM

```bash
python train_mortm_example.py --model_config configs/models/mortm/A.json \
    --train_config configs/train/pre_training.json \
    --root_directory path/to/npz_dataset \
    --save_directory out/models_mortm \
    --version MyMORTM_v1
```

### Train V\_MORTM

```bash
python train_v_mortm_example.py --model_config configs/models/v_mortm/A.json \
    --train_config configs/train/pre_training.json \
    --root_directory path/to/wav_dataset \
    --save_directory out/models_v_mortm \
    --version MyV_MORTM_v1
```

### Train BERTM

```bash
python train_bertm_example.py --model_config configs/models/bertm/class_file.json \
    --train_config configs/train/pre_training.json \
    --human_dir path/to/human_npz \
    --ai_dir path/to/ai_npz \
    --save_directory out/models_bertm \
    --version MyBERTM_v1
```

## Troubleshooting

* **load\_state\_dict errors**: Check config and map\_location.
* **Inference errors**: Ensure correct tensor shapes and vocab.
* **CUDA OOM**: Reduce batch size or use smaller model.
* **FlashAttention2 issues**: Verify CUDA and compiler compatibility.

## Token Format (Example)

```plaintext
<Gen> <TS>
Pitch=64 Duration=8 Position=0
Pitch=66 Duration=8 Position=8
...
<TE> <SME>
```

* `Pitch`: MIDI note number (e.g., 64 = E4)
* `Duration`: Length in ticks (8 ticks = eighth note)
* `Position`: Start position within the bar (0–95)
* `<SME>`: End of bar
* `<TS>/<TE>`: Track start/end tokens

## Training & Generation Workflow

1. Convert MIDI data into normalized token sequences.
2. Input one bar of tokens to the decoder.
3. Autoregressively generate the next bar.
4. Evaluation Metrics:

  * Does each bar sum to 64 ticks?
  * Frequency of out-of-scale notes.
  * Musicality and phrasing smoothness.

## Model Variants

Defined in JSON configs (`configs/models/...`). Example:

| Parameter        | Value | Description              |
| ---------------- | ----- | ------------------------ |
| d\_model         | 512   | Embedding dimension      |
| num\_heads       | 8     | Attention heads          |
| num\_layers      | 12    | Decoder layers           |
| dim\_feedforward | 2048  | FFN dimension            |
| num\_experts     | 16    | MoE experts              |
| topk\_experts    | 2     | Active experts per token |
| vocab\_size      | ...   | From vocab\_list.json    |


| Model     | Layers | Experts | Shared Experts | Embedding Dim | Heads |
|-----------|--------|---------|----------------|----------------|--------|
| MORTM-C   | 12     | 6       | 1              | 512            | 8      |
| MORTM-B   | 12     | 12      | 1              | 512            | 8      |
| MORTM-A   | 12     | 16      | 1              | 512            | 8      |
| MORTM-S   | 12     | 24      | 1              | 512            | 8      |
| MORTM-SS  | 12     | 64      | 1              | 512            | 8      |


## 🛠️ Techniques

* **FlashAttention2**: Efficient exact attention.
* **ALiBi**: Linear biases for relative positions.
* **MoE**: Sparse mixture-of-experts.
* **Absolute PE**: Optional for phrase preservation.

## License

MIT License

## References

* Vaswani et al., "Attention is All You Need"
* Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention"
* Press et al., "Train Short, Test Long: Attention with Linear Biases"
* Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"
* Huang et al., "Music Transformer"

## Author

Takaaki Nagoshi
Graduate School of Integrated Basic Sciences, Nihon University
[cs23033@g.nihon-u.ac.jp](mailto:cs23033@g.nihon-u.ac.jp)
