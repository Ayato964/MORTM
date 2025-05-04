# 🎵 MORTM: Metric-Oriented Rhythmic Transformer for Melodic Generation

**MORTM** is a Transformer-based melody generation model that focuses on the **metric structure** of music. It generates melodies in an autoregressive manner, one bar at a time, while preserving rhythmic consistency.

---

##  Features

- **Bar-level Autoregressive Generation**: Normalizes each bar to 64 ticks and generates one bar at a time.
- **MIDI-Like Token Structure**: Uses `Pitch`, `Duration`, and `Position` tokens, enhanced with <SME>, <TS>, and <TE> for structure awareness.
- **Decoder-Only Transformer**: GPT-style architecture with FlashAttention2 and ALiBi support.
- **Efficient FeedForward with MoE**: Incorporates Mixture of Experts (MoE) layers to increase capacity without increasing compute cost.
- **Applications**: Improvisation support, music education, co-creation with humans, and pop/jazz melody generation.

---

##  Model Variants (v3.0)

| Model     | Layers | Experts | Shared Experts | Embedding Dim | Heads |
|-----------|--------|---------|----------------|----------------|--------|
| MORTM-C   | 12     | 6       | 1              | 512            | 8      |
| MORTM-B   | 12     | 12      | 1              | 512            | 8      |
| MORTM-A   | 12     | 16      | 1              | 512            | 8      |
| MORTM-S   | 12     | 24      | 1              | 512            | 8      |
| MORTM-SS  | 12     | 64      | 1              | 512            | 8      |

---

## 🛠️ Techniques

- **FlashAttention2**: Memory-efficient high-speed attention.
- **ALiBi (Attention with Linear Biases)**: Enables relative position bias compatible with FlashAttention.
- **MoE (Mixture of Experts)**: Sparse feedforward network using top-k routing.
- **Optional Absolute Positional Encoding**: To preserve natural phrasing in generated melodies.

---

##  Token Format (Example)

<Gen> <TS>
Pitch=64 Duration=8 Position=0
Pitch=66 Duration=8 Position=8
...
<TE> <SME>

- `Pitch`: MIDI note number (e.g., 64 = E4)
- `Duration`: Length in ticks (8 ticks = eighth note)
- `Position`: Start position within the bar (0–95)
- `<SME>`: End of bar
- `<TS>/<TE>`: Track start/end tokens

---

##  Training & Generation Workflow

1. Convert MIDI data into normalized token sequences.
2. Input one bar of tokens to the decoder.
3. Autoregressively generate the next bar.
4. Evaluation Metrics:
   - Does each bar sum to 64 ticks?
   - Frequency of out-of-scale notes.
   - Musicality and phrasing smoothness.

---

##  Dependencies

```bash
pip install torch flash-attn==2.x numpy miditok
📚 References
Vaswani et al., "Attention is All You Need"

Huang et al., "Music Transformer"

Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"

Press et al., "Train Short, Test Long: Attention with Linear Biases"

👤 Author
Takaaki Nagoshi

Graduate School of Integrated Basic Sciences, Nihon University

Contact: t.nagoshi@example.com (replace with real email)