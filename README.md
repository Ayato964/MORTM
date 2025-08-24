Welcome to the MORTM Library!

I am your personal guide to MORTM (Metric-Oriented Rhythmic Transformer for Music Generation). This document provides a beginner-friendly overview of MORTM, its main features, installation instructions, and basic usage. MORTM is a Transformer-based melody generation model that focuses on the metric structure of music.

---

### 🎵 MORTM: Metric-Oriented Rhythmic Transformer for Music Generation

MORTM (Metric-Oriented Rhythmic Transformer for Music Generation) is a Transformer-based melody generation model that focuses on the **metric structure** of music. It generates musical sequences autoregressively, one bar at a time, while preserving rhythmic consistency. MORTM also includes V_MORTM for audio-based generation and BERTM for music classification tasks.

#### ✨ Key Features

*   **Bar-level Autoregressive Generation**: Each bar is normalized to 96 ticks (or 64 ticks in some contexts) and generated one bar at a time. It sequentially predicts one bar and uses it as the next input.
*   **High-Quality Music Generation**: Utilizes a custom tokenizer to capture musical structure, including pitch, duration, relative timing, and bars, leading to coherent outputs.
*   **Efficient Transformer Architecture**:
    *   **Decoder-Only (GPT-style)**: Optimized for autoregressive generation.
    *   **FlashAttention2 & ALiBi**: Offers memory-efficient, high-speed attention with excellent long-sequence generalization. FlashAttention2 resolves computational bottlenecks, allowing deeper models to be trained. ALiBi (Attention with Linear Biases) adds linear biases for relative positions to handle long-range dependencies and is compatible with FlashAttention2 as an alternative to Relative Positional Encoding (RPE).
    *   **Mixture of Experts (MoE)**: Employs sparsely activated Feed-Forward Network (FFN) layers, typically with Top-2 routing, to significantly increase model capacity while maintaining computational efficiency.
*   **Structured Tokenization**: Uses tokens for Pitch, Duration, and Position, along with structural tokens like `<SME>` (End of Bar), `<TS>` (Track Start), and `<TE>` (Track End). Position tokens represent the start position within a bar (0-95 ticks).
*   **Multimodal Support (V_MORTM)**: Can directly process audio features such as Mel spectrograms.
*   **Classification (BERTM)**: Features a BERT-like encoder for music classification tasks.
*   **Versatile Applications**: Applicable for melody generation, improvisation assistance, education, human-AI co-creation, and audio style transfer.

#### 🚀 Why MORTM?

*   **State-of-the-Art**: Combines advanced techniques such as FlashAttention2, MoE, and ALiBi.
*   **Musical Understanding**: Its custom tokenizer effectively captures core musical elements.
*   **Scalability**: Supports diverse styles and long musical sequences.
*   **Audio Domain**: V_MORTM enables richer audio-based generation.
*   **Modular**: Facilitates easy prototyping and comparative experiments.

---

### 🛠️ Installation

To set up your environment for MORTM, follow these steps:

#### Prerequisites

*   Python 3.8+
*   NVIDIA GPU (for FlashAttention2)
*   CUDA Toolkit (compatible with PyTorch)

#### 1. Install PyTorch

Follow the instructions at [pytorch.org](https://pytorch.org). For example:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 2. Install FlashAttention2

```bash
pip install flash-attn --no-build-isolation
```

#### 3. Install Other Dependencies

```bash
pip install numpy einops pretty_midi midi2audio soundfile torchaudio PyYAML
```
**Note**: `midi2audio` requires FluidSynth and a soundfont (e.g., `.sf2` file).

#### 4. Optional: Gmail Notifications

If you wish to receive training progress updates via Gmail:
```bash
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
```
This requires OAuth2 setup (`client_secret.json`).

---

### ⚡ Quick Start

#### Data Preparation

Convert MIDI files into tokenized `.npz` format:

```python
from mortm.train.tokenizer import Tokenizer, get_token_converter, TO_TOKEN
from mortm.convert import MIDI2Seq

# Initialize tokenizer
tokenizer = Tokenizer(music_token=get_token_converter(TO_TOKEN)) #
# Convert MIDI to sequence
converter = MIDI2Seq(tokenizer, "midi_dir", "your_midi.mid", program_list=, split_measure=12) #
converter.convert() #
# Save converted data
converter.save("output_npz_dir") #
# Save tokenizer vocabulary
tokenizer.save("vocab_output_dir") #
```

#### Inference

##### MORTM: Melody Generation

```python
import torch
import numpy as np
from mortm.models.mortm import MORTM, MORTMArgs
from mortm.train.tokenizer import Tokenizer, get_token_converter, TO_MUSIC
from mortm.de_convert import ct_token_to_midi
from mortm.models.modules.progress import _DefaultLearningProgress

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #
tokenizer = Tokenizer(music_token=get_token_converter(TO_MUSIC), load_data="vocab_list.json") #
args = MORTMArgs("configs/models/mortm/A.json") #
model = MORTM(progress=_DefaultLearningProgress(), args=args) #
model.load_state_dict(torch.load("trained_mortm.pth", map_location=DEVICE)) #
model.to(DEVICE).eval() #

seed_ids = torch.tensor([tokenizer.get("<MGEN>"), tokenizer.get("<TS>")], device=DEVICE) #
with torch.no_grad(): #
    _, full_seq = model.top_p_sampling_measure_kv_cache(seed_ids, p=0.95, max_measure=8, temperature=0.7) #
ct_token_to_midi(tokenizer, full_seq, "generated_melody.mid", program=0, tempo=120) #
```

##### BERTM: Music Classification

```python
import torch
import numpy as np
import torch.nn.functional as F
from mortm.models.bertm import BERTM, MORTMArgs as BERTMArgs
from mortm.train.tokenizer import Tokenizer, get_token_converter, TO_MUSIC
from mortm.models.modules.progress import _DefaultLearningProgress

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #
tokenizer = Tokenizer(music_token=get_token_converter(TO_MUSIC), load_data="vocab_list.json") #
args = BERTMArgs("configs/models/bertm/class_file.json") #
model = BERTM(progress=_DefaultLearningProgress(), args=args) #
model.load_state_dict(torch.load("trained_bertm.pth", map_location=DEVICE)) #
model.to(DEVICE).eval() #

input_npz = np.load("input_music.npz")['array1'] #
input_ids = torch.tensor(input_npz, dtype=torch.long, device=DEVICE).unsqueeze(0) #

with torch.no_grad(): #
    logits = model(input_ids) #
    probs = F.softmax(logits, dim=-1) #
    pred = "Human" if probs.argmax() == 0 else "AI" #
    print(f"Prediction: {pred}, Probabilities: {probs.squeeze().tolist()}") #
```

#### Training

##### Train MORTM
```bash
python run_train.py --model_config configs/models/mortm/A.json \
--train_config configs/train/pre_training.json \
--root_directory path/to/npz_dataset \
--save_directory out/models_mortm \
--version MyMORTM_v1
```

##### Train V_MORTM
```bash
python run_v_train.py --model_config configs/models/v_mortm/A.json \
--train_config configs/train/pre_training.json \
--root_directory path/to/wav_dataset \
--save_directory out/models_v_mortm \
--version MyV_MORTM_v1
```

##### Train BERTM
```bash
python class_train.py --model_config configs/models/bertm/class_file.json \
--train_config configs/train/pre_training.json \
--human_dir path/to/human_npz \
--ai_dir path/to/ai_npz \
--save_directory out/models_bertm \
--version MyBERTM_v1
```

---

### Token Format (Example)

MORTM represents musical events as structured tokens. For example:
`<MGEN> <TS> Pitch=64 Duration=8 Position=0 Pitch=66 Duration=8 Position=8 ... <TE> <SME>`

*   `Pitch`: MIDI note number (e.g., 64 = E4)
*   `Duration`: Length in ticks (8 ticks = eighth note)
*   `Position`: Start position within the bar (0–95 ticks)
*   `<SME>`: Special token indicating the end of a bar
*   `<TS>` / `<TE>`: Track start/end tokens
*   `<MGEN>`: Generation start token
*   `<ESEQ>`: Sequence end token
*   `<BLANK>`: Blank token
*   `<CLS>`: Classification token
*   `<Query_M>` / `</Query_M>`: Query melody start/end tokens
*   `<Query_C>` / `</Query_C>`: Query chord start/end tokens

---

### Troubleshooting

*   **`load_state_dict` errors**: Check configuration and `map_location`.
*   **Inference errors**: Ensure correct tensor shapes and vocabulary are used.
*   **CUDA OOM (Out of Memory)**: Reduce batch size or use a smaller model.
*   **FlashAttention2 issues**: Verify CUDA and compiler compatibility.

---

### Model Variants

Model parameters are defined in JSON configuration files (e.g., `configs/models/...`). Key parameters and model types include:

| Parameter         | Value | Description           |
| :---------------- | :---- | :-------------------- |
| `d_model`         | 512   | Embedding dimension   |
| `num_heads`       | 8     | Number of attention heads |
| `num_layers`      | 12    | Number of decoder layers |
| `dim_feedforward` | 2048  | FFN dimension         |
| `num_experts`     | 16    | Number of MoE experts |
| `topk_experts`    | 2     | Number of active experts per token |
| `vocab_size`      | ...   | Obtained from `vocab_list.json` |

Since MORTM 3.0, models are provided based on the number of experts.

| Model   | Layers | Experts | Shared Experts | Embedding Dim | Heads |
| :------ | :----- | :------ | :------------- | :------------ | :---- |
| MORTM-C | 12     | 6       | 1              | 512           | 8     |
| MORTM-B | 12     | 12      | 1              | 512           | 8     |
| MORTM-A | 12     | 16      | 1              | 512           | 8     |
| MORTM-S | 12     | 24      | 1              | 512           | 8     |
| MORTM-SS| 12     | 64      | 1              | 512           | 8     |

---

### License

MIT License

### Author

Takaaki Nagoshi
Graduate School of Integrated Basic Sciences, Nihon University
cs23033@g.nihon-u.ac.jp
