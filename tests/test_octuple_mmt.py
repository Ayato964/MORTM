import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from ConvertSymbolicTokens import SymbolicCorpusConfig, SymbolicCorpusConverter
from mortm.models.modules.attention import RotaryEmbedding
from mortm.models.modules.config import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    INST_RVQ_TOKEN_OFFSET,
    MORTM5Args,
    NUM_SLOT_TOKENS,
    OCTUPLE_DEFAULT_ATTR_SIZES,
    PAD_TOKEN_ID,
    SLOT_CODE_DRUMS,
    SLOT_CODE_EMPTY,
)
from mortm.models.mortm5 import OctupleTransformer, build_mortm5_ablation_model
from mortm.train.config import TrainArgs
from mortm.train.datasets import (
    DynamicStepBatchPacker,
    MORTM5Collate,
    MORTM5MemmapDataset,
    MORTM5SymbolicMemmapDataset,
    mortm5_collate_fn,
)
from mortm.train.muon import build_muon_optimizer
from mortm.utils.convert_pianoroll import (
    CHUNK_STEPS,
    STEPS_PER_BAR,
    chunks_to_midi,
    rasterize_notes_to_chunks_u8,
    slot_codes_to_roll_meta,
)
from mortm.utils.symbolic_tokenizers import (
    DEFAULT_GRID_SPEC,
    OctupleMMTTokenizer,
    extract_segment_canonical_notes,
)


def test_dynamic_rope_arbitrary_length_and_bf16_precision():
    dim = 64
    rope = RotaryEmbedding(dim=dim, max_position_embeddings=64, base=10000.0)
    assert len(rope.state_dict()) == 0, "RotaryEmbedding must not add persistent state_dict keys"

    # Simulate model.to(bfloat16) and query far beyond initial max_position_embeddings (and >256 bf16 int limit)
    rope = rope.to(dtype=torch.bfloat16)
    x_zero = torch.empty((0, 4, dim), dtype=torch.bfloat16)
    cos_z, sin_z = rope(x_zero, seq_len=0)
    assert cos_z.shape == (0, dim) and sin_z.shape == (0, dim)

    pos_ids = torch.tensor([0, 257, 1025, 4099, 5123], dtype=torch.long)
    x_dummy = torch.zeros((len(pos_ids), 4, dim), dtype=torch.bfloat16)
    cos_table, sin_table = rope(x_dummy, seq_len=int(pos_ids.max().item()) + 1)
    assert rope.max_position_embeddings >= 5124
    cos_bf16 = cos_table[pos_ids]
    sin_bf16 = sin_table[pos_ids]
    assert cos_bf16.dtype == torch.bfloat16 and sin_bf16.dtype == torch.bfloat16

    # Exact float64 reference check at pos=5123 (catches bfloat16 arange integer truncation)
    inv_freq_f64 = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    freqs_f64 = torch.outer(pos_ids.to(torch.float64), inv_freq_f64)
    emb_f64 = torch.cat((freqs_f64, freqs_f64), dim=-1)
    ref_cos = emb_f64.cos().to(torch.bfloat16)
    ref_sin = emb_f64.sin().to(torch.bfloat16)
    assert torch.allclose(cos_bf16.float(), ref_cos.float(), atol=1e-2)
    assert torch.allclose(sin_bf16.float(), ref_sin.float(), atol=1e-2)


def test_octuple_tokenizer_quantization_clipping_and_roundtrip():
    tok = OctupleMMTTokenizer()
    assert tok.vocab_size == 817
    assert tok.step_unit_width == 6

    # 1. 312-bin duration & 32-bin velocity idempotence
    lut = DEFAULT_GRID_SPEC.build_dur_lut()
    assert len(lut) == 312
    assert np.array_equal(lut[:192], np.arange(1, 193, dtype=np.int32))
    assert int(lut[-1]) == 3072
    re_bins = DEFAULT_GRID_SPEC.quantize_dur_np(lut)
    assert np.array_equal(re_bins, np.arange(312, dtype=np.int32))

    vel_bins = np.arange(32, dtype=np.int32)
    deq_vel = DEFAULT_GRID_SPEC.dequantize_vel_np(vel_bins)
    assert np.array_equal(DEFAULT_GRID_SPEC.quantize_vel_np(deq_vel), vel_bins)

    # 2. Cross-segment sustain clipping & collision merging parity vs rasterize_notes_to_chunks_u8
    # Segment 1 covers chunks [128, 256) => steps [3072, 6144)
    s0 = 128 * CHUNK_STEPS  # 3072
    notes_arr = np.asarray(
        [
            # Sustained across boundary s0=3072 -> clipped to local_step=0 (bar=0,pos=0), dur=48
            [1, s0 - 96, s0 + 48, 60, 100],
            # Duplicate collision at same (slot=2, start=s0+24, pitch=67) -> merged with max_dur=72, max_vel=118
            [2, s0 + 24, s0 + 48, 67, 80],
            [2, s0 + 24, s0 + 96, 67, 118],
            # Drum hit in bar 15
            [0, s0 + 15 * STEPS_PER_BAR + 12, s0 + 15 * STEPS_PER_BAR + 18, 36, 110],
            # Outside segment (before s0) -> excluded
            [1, 0, 96, 48, 90],
        ],
        dtype=np.int32,
    )
    slot_codes = np.asarray([SLOT_CODE_DRUMS, 0, 24, SLOT_CODE_EMPTY], dtype=np.int16)

    canon = extract_segment_canonical_notes(notes_arr, chunk_start=128, chunk_end=256, max_bars=16)
    assert len(canon) == 3
    # First note is clipped cross-boundary sustain at (bar=0, pos=0, slot=1, pitch=60, dur=48, vel=100)
    assert tuple(canon[0].tolist()) == (0, 0, 1, 60, 48, 100)
    # Second note is merged collision (bar=0, pos=24, slot=2, pitch=67, dur=72, vel=118)
    assert tuple(canon[1].tolist()) == (0, 24, 2, 67, 72, 118)

    body_u16 = tok.encode_song_segment(notes_arr, chunk_start=128, chunk_end=256)
    full_seq = torch.from_numpy(tok.build_full_sequence(body_u16, slot_codes=slot_codes))
    assert full_seq.numel() == (1 + NUM_SLOT_TOKENS + 3 + 1) * 6

    dec_notes, dec_slots = tok.decode_tokens_to_notes(full_seq, num_chunks=128)
    assert np.array_equal(dec_slots, slot_codes)
    assert len(dec_notes) == 3
    # Exact start/end/pitch/slot preservation for durations <= 192
    assert tuple(dec_notes[0, :4].tolist()) == (1, 0, 48, 60)
    assert tuple(dec_notes[1, :4].tolist()) == (2, 24, 96, 67)

    # Compare active sustain mask of decoded chunks_u8 vs ground-truth sliced chunks_u8
    gt_full_u8 = rasterize_notes_to_chunks_u8(256, notes_arr)[128:256]
    dec_u8, _ = tok.decode_tokens_to_chunks_u8(full_seq, num_chunks=128)
    # Note: cross-boundary note has sustain in gt_full_u8[1::2] and identical pitch/duration coverage
    assert np.array_equal(dec_u8[:, 1::2] > 0, gt_full_u8[:, 1::2] > 0)

    # 3. Silent (0-note) segment round-trip
    empty_body = np.zeros((0,), dtype=np.uint16)
    empty_seq = torch.from_numpy(tok.build_full_sequence(empty_body, slot_codes=slot_codes))
    assert empty_seq.numel() == 6 * 6
    empty_notes, empty_slots = tok.decode_tokens_to_notes(empty_seq, num_chunks=16)
    assert len(empty_notes) == 0
    assert np.array_equal(empty_slots, slot_codes)


def test_octuple_160m_exact_parameter_counts_and_muon_routing():
    oct_args = MORTM5Args("configs/models/mortm5/symic_main/octuple_160m.json")
    sym_args = MORTM5Args("configs/models/mortm5/symic_main/symic_g4l4_160m.json")

    assert oct_args.model_mode == "octuple"
    assert oct_args.step_unit_width == 6
    assert oct_args.token_offset == INST_RVQ_TOKEN_OFFSET
    assert oct_args.vocab_size == 817
    assert oct_args.recompute_vocab() == 817

    with torch.device("meta"):
        oct_meta = build_mortm5_ablation_model(oct_args)
        sym_meta = build_mortm5_ablation_model(sym_args)
    assert isinstance(oct_meta, OctupleTransformer)

    oct_non_emb = oct_meta.get_non_embedding_param_count()
    sym_non_emb = sym_meta.get_non_embedding_param_count()
    assert oct_non_emb == 163_603_456
    assert oct_non_emb == sym_non_emb

    oct_total = sum(p.numel() for p in oct_meta.parameters() if p.requires_grad)
    assert oct_total == 165_276_672
    assert oct_total - oct_non_emb == 1_673_216  # 817 * 1024 * 2

    # Verify HybridMuonOptimizer parameter group routing on a compact CPU model
    small_cfg = oct_args.to_dict()
    small_cfg.update({"d_model": 64, "d_layer": 2, "num_heads": 4, "dim_feedforward": 192})
    small_model = OctupleTransformer(MORTM5Args(small_cfg)).cpu()
    t_args = TrainArgs("configs/train/mortm5/octuple_main_160m.json")
    t_args_4ep = TrainArgs("configs/train/mortm5/octuple_main_160m_4ep.json")
    assert t_args.max_micro_steps == 65536
    assert t_args.batch_size == 128 and t_args.accumulation_steps == 2 and t_args.big_batch_size == 16384
    assert t_args_4ep.max_micro_steps == 65536 and t_args_4ep.num_epochs == 4
    assert t_args_4ep.scheduler["total_steps"] == 58440
    opt = build_muon_optimizer(
        small_model,
        lr=t_args.lr_param,
        weight_decay=t_args.weight_decay,
        verbose=False,
    )

    muon_ids = {id(p) for g in opt.muon.param_groups for p in g["params"]}
    adamw_ids = {id(p) for g in opt.adamw.param_groups for p in g["params"]}
    assert id(small_model.embedding.weight) in adamw_ids
    assert id(small_model.Wout.proj.weight) in adamw_ids
    assert id(small_model.embedding.weight) not in muon_ids
    assert id(small_model.Wout.proj.weight) not in muon_ids
    # Every 2D decoder matrix is routed to Muon
    assert len(muon_ids) == 2 * 5  # d_layer=2 * (qkv_weight, W_o, w1, w2, w3)
    assert "attr_offsets" not in small_model.state_dict()
    assert "attr_limits" not in small_model.state_dict()


def test_decode_octuple_head_loss_grad_equivalence_and_kv_cache_parity():
    torch.manual_seed(123)
    cfg = {
        "name": "OctupleTiny",
        "tokenizer_key": "octuple_mmt",
        "model_mode": "octuple",
        "symbolic_vocab": True,
        "tuple_size": 6,
        "octuple_attr_sizes": list(OCTUPLE_DEFAULT_ATTR_SIZES),
        "use_instrument_tokens": True,
        "token_offset": 133,
        "vocab_size": 817,
        "d_model": 64,
        "d_layer": 2,
        "num_heads": 4,
        "dim_feedforward": 192,
        "dropout": 0.0,
        "position_length": 16,  # Intentionally small to exercise dynamic RoPE & KV-cache growth (>16 steps)
        "norm_type": "rmsnorm",
        "use_rope": True,
        "use_silu": True,
        "use_bias": False,
        "use_moe_decoder": False,
    }
    args = MORTM5Args(cfg)
    model = OctupleTransformer(args).cpu().eval()
    tok = OctupleMMTTokenizer()

    # Build 2 sequences of lengths > position_length (22 steps and 12 steps)
    rng = np.random.RandomState(7)
    seqs = []
    for n_notes in (16, 6):
        canon = np.zeros((n_notes, 6), dtype=np.int32)
        canon[:, 0] = np.sort(rng.randint(0, 16, size=n_notes))
        canon[:, 1] = np.sort(rng.randint(0, 192, size=n_notes))
        canon[:, 2] = rng.randint(0, 4, size=n_notes)
        canon[:, 3] = rng.randint(24, 96, size=n_notes)
        canon[:, 4] = rng.randint(1, 240, size=n_notes)
        canon[:, 5] = rng.randint(20, 120, size=n_notes)
        body = tok.encode_canonical_segment_notes(canon)
        seqs.append(torch.from_numpy(tok.build_full_sequence(body, [SLOT_CODE_DRUMS, 0, 32, SLOT_CODE_EMPTY])))

    in_ids, tgts, pad_mask = mortm5_collate_fn(seqs, mode="octuple", vq_groups=6)

    # 1. Exact loss & gradient equivalence vs 6 independent exact-range CrossEntropy heads
    model.zero_grad(set_to_none=True)
    logits = model(in_ids, padding_mask=pad_mask, is_causal=True)
    local_tgts = model.to_local_targets(tgts, ignore_index=PAD_TOKEN_ID)
    seq0_3d = seqs[0].reshape(1, -1, 6)
    assert torch.equal(model.local_to_global_ids(model.to_local_targets(seq0_3d)), seq0_3d)

    fused_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        local_tgts.reshape(-1),
        ignore_index=PAD_TOKEN_ID,
    )
    fused_loss.backward()
    fused_proj_grad = model.Wout.proj.weight.grad.detach().clone()
    fused_emb_grad = model.embedding.weight.grad.detach().clone()
    assert not torch.isnan(fused_proj_grad).any()
    assert float(fused_emb_grad[PAD_TOKEN_ID].abs().max().item()) == 0.0

    model.zero_grad(set_to_none=True)
    logits_ref = model(in_ids, padding_mask=pad_mask, is_causal=True)
    total_sum = torch.zeros((), dtype=torch.float32)
    total_cnt = 0
    for m, sz in enumerate(OCTUPLE_DEFAULT_ATTR_SIZES):
        valid_cols = 133 + sz
        logits_m = logits_ref[:, :, m, :valid_cols].reshape(-1, valid_cols)
        tgt_m = local_tgts[:, :, m].reshape(-1)
        total_sum = total_sum + F.cross_entropy(logits_m, tgt_m, ignore_index=PAD_TOKEN_ID, reduction="sum")
        total_cnt += int((tgt_m != PAD_TOKEN_ID).sum().item())
    ref_loss = total_sum / float(total_cnt)
    ref_loss.backward()

    assert torch.allclose(fused_loss, ref_loss, atol=1e-6)
    assert torch.allclose(fused_proj_grad, model.Wout.proj.weight.grad, atol=1e-6)
    assert torch.allclose(fused_emb_grad, model.embedding.weight.grad, atol=1e-6)

    # 2. KV-cache step-by-step decoding parity beyond initial position_length=16
    seq_3d = seqs[0].reshape(1, -1, 6)  # [1, 22, 6] > 16
    with torch.no_grad():
        full_logits = model(seq_3d, is_causal=True, is_save_cache=False)
        model.reset_kv_cache()
        prefill_logits = model(seq_3d[:, :5, :], is_causal=True, is_save_cache=True)
        step_outs = [prefill_logits]
        for s_i in range(5, seq_3d.size(1)):
            step_outs.append(model(seq_3d[:, s_i : s_i + 1, :], is_causal=True, is_save_cache=True))
        kv_logits = torch.cat(step_outs, dim=1)
        model.reset_kv_cache()

    finite_mask = torch.isfinite(full_logits)
    assert torch.equal(finite_mask, torch.isfinite(kv_logits))
    assert torch.allclose(full_logits[finite_mask], kv_logits[finite_mask], atol=2e-2, rtol=2e-2)

    # 3. Autoregressive generate() grammar & lexicographical monotonic (bar, pos) check
    custom_slots = [SLOT_CODE_DRUMS, 25, 48, SLOT_CODE_EMPTY]
    gen = model.generate(
        max_notes=24,
        max_bars=16,
        min_notes=4,
        slot_codes=custom_slots,
        batch_size=2,
        temperature=1.0,
        top_k=16,
        monotonic_time=True,
    )
    assert gen.dim() == 3 and gen.size(0) == 2 and gen.size(2) == 6
    for b_i in range(2):
        ext_slots = model.extract_slot_codes(gen[b_i])
        assert ext_slots is not None and list(ext_slots) == custom_slots
        dec_notes, _ = model.decode_to_notes(gen[b_i])
        assert len(dec_notes) >= 4
        # Starts must be non-decreasing because (bar, pos) is constrained monotonically
        assert np.all(np.diff(dec_notes[:, 1]) >= 0)


def test_dynamic_step_batch_packer_exact_gradient_invariance():
    torch.manual_seed(99)
    cfg = {
        "name": "OctupleGradCheck",
        "tokenizer_key": "octuple_mmt",
        "model_mode": "octuple",
        "symbolic_vocab": True,
        "tuple_size": 6,
        "use_instrument_tokens": True,
        "token_offset": 133,
        "vocab_size": 817,
        "d_model": 48,
        "d_layer": 2,
        "num_heads": 4,
        "dim_feedforward": 128,
        "dropout": 0.0,
        "position_length": 64,
    }
    args = MORTM5Args(cfg)
    model_single = OctupleTransformer(args).cpu().train()
    model_packed = OctupleTransformer(args).cpu().train()
    model_packed.load_state_dict(model_single.state_dict())

    tok = OctupleMMTTokenizer()
    rng = np.random.RandomState(42)
    # Include 0-note segment, medium segments, and an outlier > max_micro_steps (e.g. 40 steps > max_micro_steps=28)
    note_counts = [0, 3, 18, 2, 35, 7, 11, 1]
    sequences = []
    for cnt in note_counts:
        if cnt == 0:
            body = np.zeros((0,), dtype=np.uint16)
        else:
            canon = np.zeros((cnt, 6), dtype=np.int32)
            canon[:, 0] = np.sort(rng.randint(0, 16, size=cnt))
            canon[:, 1] = np.sort(rng.randint(0, 192, size=cnt))
            canon[:, 2] = rng.randint(0, 4, size=cnt)
            canon[:, 3] = rng.randint(36, 84, size=cnt)
            canon[:, 4] = rng.randint(1, 96, size=cnt)
            canon[:, 5] = rng.randint(30, 110, size=cnt)
            body = tok.encode_canonical_segment_notes(canon)
        sequences.append(torch.from_numpy(tok.build_full_sequence(body, [SLOT_CODE_DRUMS, 0, -1, -1])))

    collate = MORTM5Collate(mode="octuple", vq_groups=6, pad_token_id=PAD_TOKEN_ID)

    # A. Reference single-batch forward/backward
    in_all, tgt_all, mask_all = collate(sequences)
    logits_all = model_single(in_all, padding_mask=mask_all, is_causal=True)
    tgt_flat_all = model_single.format_training_targets(tgt_all, ignore_index=PAD_TOKEN_ID)
    loss_single = F.cross_entropy(logits_all.reshape(-1, logits_all.size(-1)), tgt_flat_all, ignore_index=PAD_TOKEN_ID)
    loss_single.backward()
    tokens_single = int((in_all != PAD_TOKEN_ID).sum().item())

    # B. DynamicStepBatchPacker split into multiple variable-size sub-batches
    packer = DynamicStepBatchPacker(max_micro_steps=28, max_batch_size=3, unit_width=6, pad_token_id=PAD_TOKEN_ID)
    packed_step = packer.pack_sequences(sequences, collate_fn=collate)
    assert len(packed_step.sub_batches) >= 4
    assert packed_step.total_valid_targets == int((tgt_flat_all != PAD_TOKEN_ID).sum().item())

    denom = float(packed_step.total_valid_targets)
    loss_accum = 0.0
    tokens_packed = 0
    for sub_in, sub_tgt, sub_mask in packed_step.sub_batches:
        tokens_packed += int((sub_in != PAD_TOKEN_ID).sum().item())
        sub_logits = model_packed(sub_in, padding_mask=sub_mask, is_causal=True)
        sub_tgt_flat = model_packed.format_training_targets(sub_tgt, ignore_index=PAD_TOKEN_ID)
        sub_sum = F.cross_entropy(
            sub_logits.reshape(-1, sub_logits.size(-1)),
            sub_tgt_flat,
            ignore_index=PAD_TOKEN_ID,
            reduction="sum",
        )
        (sub_sum / denom).backward()
        loss_accum += float(sub_sum.item()) / denom

    assert tokens_packed == tokens_single
    assert abs(loss_accum - float(loss_single.item())) < 1e-5
    for (n1, p1), (n2, p2) in zip(model_single.named_parameters(), model_packed.named_parameters()):
        assert n1 == n2
        assert torch.allclose(p1.grad, p2.grad, atol=5e-4, rtol=1e-3), f"Grad mismatch on {n1}"


def test_converter_and_colab_worker_octuple_e2e():
    import colab_gpu_worker as worker

    with tempfile.TemporaryDirectory() as tmpdir:
        midi_dir = Path(tmpdir) / "midis"
        midi_dir.mkdir(parents=True, exist_ok=True)
        midi_paths = []

        # Create 3 synthetic MIDIs: 1 short (<8 chunks, skipped), 1 normal (16 chunks), 1 multi-segment (140 chunks)
        for idx, n_chunks in enumerate([4, 16, 140]):
            roll = np.zeros((n_chunks, 8, 24, 128), dtype=np.float32)
            roll[0, 0, 0, 36] = 0.9
            roll[0, 1, 0:4, 36] = 1.0
            if n_chunks >= 16:
                roll[12, 2, 0, 60] = 0.85
                roll[12, 3, 0:12, 60] = 1.0
            if n_chunks >= 140:
                # Sustain crossing chunk 128 boundary (127 -> 128) + note at chunk 138 so trailing segment >= 8 chunks
                roll[127, 2, 12, 64] = 0.8
                roll[127, 3, 12:24, 64] = 1.0
                roll[128, 3, 0:12, 64] = 1.0
                roll[138, 2, 0, 67] = 0.85
                roll[138, 3, 0:12, 67] = 1.0
            mp_str = str(midi_dir / f"song_{idx}.mid")
            chunks_to_midi(
                roll,
                meta=slot_codes_to_roll_meta([SLOT_CODE_DRUMS, 0, SLOT_CODE_EMPTY, SLOT_CODE_EMPTY]),
                output_path=mp_str,
            )
            midi_paths.append(mp_str)

        split_json = Path(tmpdir) / "mini.json"
        split_json.write_text(json.dumps(midi_paths), encoding="utf-8")

        out_root = Path(tmpdir) / "tokens"
        conv = SymbolicCorpusConverter(
            SymbolicCorpusConfig(
                tokenizer_key="octuple_mmt",
                out_dir=str(out_root),
                num_workers=0,
                checkpoint_every=1,
                verify_m4_dir=None,
            )
        )
        for sp in ("train", "val"):
            st = conv.convert_split(split=sp, json_path=str(split_json))
            assert st["valid_songs"] == 2
            assert st["skipped_songs"] == 1
            assert st["total_segments"] == 3  # 16 chunks -> 1 seg; 140 chunks -> 2 segs (128 + 12)

        oct_dir = str(out_root / "octuple_mmt")
        m_cfg = {
            "name": "MORTM5_Octuple_Test",
            "tokenizer_key": "octuple_mmt",
            "model_mode": "octuple",
            "symbolic_vocab": True,
            "tuple_size": 6,
            "use_instrument_tokens": True,
            "token_offset": 133,
            "vocab_size": 817,
            "d_model": 64,
            "d_layer": 2,
            "num_heads": 4,
            "dim_feedforward": 128,
            "dropout": 0.0,
            "position_length": 64,
        }
        t_cfg = {
            "optimizer": "muon",
            "lr_param": 0.001,
            "big_batch_size": 8,
            "batch_size": 2,
            "accumulation_steps": 1,
            "max_micro_steps": 64,
            "num_epochs": 1,
            "scheduler": {"type": "wsd", "warmup_steps": 1, "stable_ratio": 0.5, "total_steps": 2, "auto_total_steps": True},
        }

        ds = MORTM5MemmapDataset.from_split_dir(oct_dir, "train", MORTM5Args(m_cfg))
        assert isinstance(ds, MORTM5SymbolicMemmapDataset)
        assert len(ds) == 3

        # Test colab_gpu_worker lifecycle with OctupleTransformer + DynamicStepBatchPacker
        old_drive = os.environ.get("MORTM5_DRIVE_CKPT_DIR")
        old_worker_dir = os.environ.get("MORTM5_WORKER_DIR")
        os.environ["MORTM5_DRIVE_CKPT_DIR"] = str(Path(tmpdir) / "ckpts")
        os.environ["MORTM5_WORKER_DIR"] = str(Path(tmpdir) / "worker")

        class _DummyReq:
            def __init__(self, payload=None, query=None):
                self._payload = payload or {}
                self.query_params = query or {}
                self.headers = {"content-type": "application/json"}

            async def json(self):
                return self._payload

        try:
            init_res = asyncio.run(
                worker.mortm5_init(
                    _DummyReq(
                        payload={
                            "model_config": m_cfg,
                            "train_config": t_cfg,
                            "colab_split_dir": oct_dir,
                            "stage_to_local": False,
                            "version": "oct_e2e",
                        }
                    )
                )
            )
            assert init_res["model_class"] == "OctupleTransformer"
            assert init_res["train_windows"] == 3
            assert init_res["total_steps"] == 2

            asyncio.run(worker.mortm5_start_outer_batch(_DummyReq(payload={"outer_idx": 0, "epoch": 0})))
            for _ in range(100):
                poll = asyncio.run(worker.mortm5_poll_outer_batch(_DummyReq(query={"cursor": "0"})))
                if poll["done"]:
                    break
                import time as _t
                _t.sleep(0.02)
            assert poll["error"] is None
            assert poll["step"] == 2
            assert len(poll["results"]) == 2

            val_res = asyncio.run(worker.mortm5_validate(_DummyReq(query={"max_batches": "0"})))
            assert val_res["val_tokens"] > 0
            assert np.isfinite(val_res["val_loss"]) and val_res["val_loss"] > 0.0

            # Also verify Mode B: Mac -> Colab CSR variable-length sequence upload
            from run_remote_mortm5 import _pack_csr_sequences_zlib

            csr_bytes = _pack_csr_sequences_zlib([ds[i] for i in range(len(ds))])
            init_stream = asyncio.run(
                worker.mortm5_init(
                    _DummyReq(
                        payload={
                            "model_config": m_cfg,
                            "train_config": t_cfg,
                            "colab_split_dir": None,
                            "train_windows": len(ds),
                            "stage_to_local": False,
                            "version": "oct_csr_stream",
                        }
                    )
                )
            )
            assert init_stream["total_steps"] == 2
            worker.STATE["upload_parts"] = [csr_bytes]
            asyncio.run(worker.mortm5_start_outer_batch(_DummyReq(payload={"outer_idx": 0, "epoch": 0})))
            for _ in range(100):
                poll_csr = asyncio.run(worker.mortm5_poll_outer_batch(_DummyReq(query={"cursor": "0"})))
                if poll_csr["done"]:
                    break
                import time as _t
                _t.sleep(0.02)
            assert poll_csr["error"] is None
            assert poll_csr["step"] == 2

            worker.STATE["upload_parts"] = [csr_bytes]
            val_csr = asyncio.run(worker.mortm5_validate(_DummyReq(query={"max_batches": "0"})))
            assert val_csr["val_tokens"] == val_res["val_tokens"]
            assert np.isfinite(val_csr["val_loss"]) and val_csr["val_loss"] > 0.0
        finally:
            if old_drive is None:
                os.environ.pop("MORTM5_DRIVE_CKPT_DIR", None)
            else:
                os.environ["MORTM5_DRIVE_CKPT_DIR"] = old_drive
            if old_worker_dir is None:
                os.environ.pop("MORTM5_WORKER_DIR", None)
            else:
                os.environ["MORTM5_WORKER_DIR"] = old_worker_dir


def test_real_gmd_val_parity_if_present():
    oct_dir = ROOT_DIR / "out" / "mortm5_tokens" / "octuple_mmt"
    m4_dir = ROOT_DIR / "out" / "mortm5_tokens" / "m4_g4l4"
    if not (oct_dir / "val_state.json").exists() or not (m4_dir / "val_state.json").exists():
        pytest.skip("Real GMD val tokens not present locally")

    oct_st = json.loads((oct_dir / "val_state.json").read_text(encoding="utf-8"))
    m4_st = json.loads((m4_dir / "val_state.json").read_text(encoding="utf-8"))
    assert oct_st["valid_songs"] == m4_st["valid_songs"] == 4950
    assert oct_st["skipped_songs"] == m4_st["skipped_songs"] == 50
    assert oct_st["total_chunks"] == m4_st["total_chunks"] == 2_578_538
    assert np.array_equal(np.load(oct_dir / "val_slots.npy"), np.load(m4_dir / "val_slots.npy"))

    oct_args = MORTM5Args("configs/models/mortm5/symic_main/octuple_160m.json")
    ds = MORTM5MemmapDataset.from_split_dir(str(oct_dir), "val", oct_args)
    assert len(ds) == oct_st["total_segments"] == 22_607
    sample0 = ds[0]
    assert sample0.dim() == 1 and sample0.numel() >= 36 and sample0.numel() % 6 == 0


def test_octuple_generate_heterogeneous_prompt_min_bar_empty_slots_and_nan_safety():
    # 1. _sample_from_logits all -inf / NaN guard AND mixed grammar-mask (-inf) + valid NaN / negative finite guard
    bad_logits = torch.full((3, 16), float("-inf"))
    bad_logits[1, :] = float("nan")
    sampled = OctupleTransformer._sample_from_logits(bad_logits, temperature=1.0, top_k=4, top_p=0.9)
    assert sampled.shape == (3,)
    assert torch.all((sampled >= 0) & (sampled < 16))

    # Velocity-style masked row: legal local classes 133..164 are NaN, while 0..132 (<PAD>/prefix) and 165..444 are -inf
    vel_nan_logits = torch.full((64, 445), float("-inf"), dtype=torch.float32)
    vel_nan_logits[:, 133:165] = float("nan")
    greedy_vel = OctupleTransformer._sample_from_logits(vel_nan_logits, temperature=0.0)
    assert torch.all((greedy_vel >= 133) & (greedy_vel < 165))
    stoch_vel = OctupleTransformer._sample_from_logits(vel_nan_logits, temperature=1.0, top_k=16, top_p=0.9)
    assert torch.all((stoch_vel >= 133) & (stoch_vel < 165))

    # Negative finite valid logits [-2.0, -5.0] + -inf grammar masks must NEVER select -inf masked classes
    neg_masked = torch.full((32, 445), float("-inf"), dtype=torch.bfloat16)
    neg_masked[:, 133] = -2.0
    neg_masked[:, 134] = -5.0
    assert torch.all(OctupleTransformer._sample_from_logits(neg_masked, temperature=0.0) == 133)
    assert torch.all(OctupleTransformer._sample_from_logits(neg_masked, temperature=1e-20) == 133)
    neg_stoch = OctupleTransformer._sample_from_logits(neg_masked, temperature=1.0, top_k=8, top_p=0.95)
    assert torch.all((neg_stoch == 133) | (neg_stoch == 134))

    # 2. Build tiny Octuple model
    torch.manual_seed(2026)
    args = MORTM5Args(
        {
            "name": "OctupleGenEdge",
            "tokenizer_key": "octuple_mmt",
            "model_mode": "octuple",
            "symbolic_vocab": True,
            "tuple_size": 6,
            "use_instrument_tokens": True,
            "token_offset": 133,
            "vocab_size": 817,
            "d_model": 48,
            "d_layer": 2,
            "num_heads": 4,
            "dim_feedforward": 128,
            "dropout": 0.0,
            "position_length": 32,
        }
    )
    model = OctupleTransformer(args).cpu().eval()
    tok = OctupleMMTTokenizer()

    # 3. Heterogeneous right-padded prompt batch + min_bar=4 + min_notes=3 + active slot [DRUMS, EMPTY, EMPTY, EMPTY]
    slots_one_active = [SLOT_CODE_DRUMS, SLOT_CODE_EMPTY, SLOT_CODE_EMPTY, SLOT_CODE_EMPTY]
    # Prompt 0: 2 notes in bar 0..1 (trailing silence in bars 2..3)
    canon0 = np.asarray([[0, 0, 0, 36, 12, 100], [1, 48, 0, 38, 12, 95]], dtype=np.int32)
    p0 = torch.from_numpy(
        tok.build_full_sequence(tok.encode_canonical_segment_notes(canon0), slot_codes=slots_one_active, add_eos=False)
    ).reshape(-1, 6)  # [7, 6]
    # Prompt 1: 5 notes in bar 0..2
    canon1 = np.asarray(
        [
            [0, 0, 0, 36, 12, 100],
            [0, 48, 0, 42, 12, 90],
            [1, 0, 0, 38, 12, 100],
            [1, 96, 0, 46, 24, 95],
            [2, 24, 0, 36, 12, 105],
        ],
        dtype=np.int32,
    )
    p1 = torch.from_numpy(
        tok.build_full_sequence(tok.encode_canonical_segment_notes(canon1), slot_codes=slots_one_active, add_eos=False)
    ).reshape(-1, 6)  # [10, 6]

    padded_prompt = torch.full((2, 10, 6), PAD_TOKEN_ID, dtype=torch.long)
    padded_prompt[0, :7, :] = p0
    padded_prompt[1, :10, :] = p1

    gen = model.generate(
        prompt=padded_prompt,
        max_notes=8,
        max_bars=16,
        min_bar=4,
        min_notes=3,
        temperature=1.0,
        top_k=16,
        monotonic_time=True,
    )
    assert gen.dim() == 3 and gen.size(0) == 2 and gen.size(2) == 6
    for b_i, orig_notes in enumerate([2, 5]):
        seq_steps = gen[b_i]
        # Verify no mid-sequence PAD holes before EOS
        non_pad = (seq_steps != PAD_TOKEN_ID).any(dim=-1).nonzero(as_tuple=False).squeeze(-1)
        assert torch.equal(non_pad, torch.arange(len(non_pad), dtype=torch.long))
        dec_notes, dec_slots = model.decode_to_notes(seq_steps, num_chunks=128)
        assert list(dec_slots) == slots_one_active
        # min_notes=3 newly generated notes on top of orig_notes
        assert len(dec_notes) >= orig_notes + 3
        # Every newly generated note must start at bar >= 4 (step >= 4 * 192 = 768) and use only active slot 0
        new_notes = dec_notes[orig_notes:]
        assert np.all(new_notes[:, 1] >= 4 * STEPS_PER_BAR)
        assert np.all(new_notes[:, 0] == 0)

    # 4. All-empty slots must emit immediate <EOS> without crashing even when min_notes > 0
    all_empty_slots = [SLOT_CODE_EMPTY] * 4
    gen_empty = model.generate(
        slot_codes=all_empty_slots,
        max_notes=8,
        min_notes=5,
        temperature=1.0,
    )
    assert gen_empty.shape == (1, 6, 6)  # 1 BOS + 4 INST + 1 EOS
    assert torch.all(gen_empty[0, -1, :] == EOS_TOKEN_ID)

    # 5. Decoder strips any note referencing an inactive SLOT_CODE_EMPTY slot identically in notes & chunks_u8
    spurious_canon = np.asarray([[0, 0, 3, 60, 24, 100]], dtype=np.int32)  # slot=3 is EMPTY in slots_one_active
    spurious_seq = torch.from_numpy(
        tok.build_full_sequence(tok.encode_canonical_segment_notes(spurious_canon), slot_codes=slots_one_active)
    )
    d_notes, _ = tok.decode_tokens_to_notes(spurious_seq, num_chunks=8)
    d_u8, _ = tok.decode_tokens_to_chunks_u8(spurious_seq, num_chunks=8)
    m_u8, _ = model.decode_to_chunks_u8(spurious_seq, num_chunks=8, slot_codes=slots_one_active)
    m_f32, _ = model.decode_to_chunks_float(spurious_seq, num_chunks=8, slot_codes=slots_one_active)
    assert len(d_notes) == 0
    assert int(d_u8.max()) == 0
    assert int(m_u8.max()) == 0
    assert float(m_f32.max()) == 0.0


def test_mortm5_midi_generator_octuple_without_vq_and_4bar_continuation(tmp_path: Path):
    from mortm.utils.mortm5_infer import Mortm5MidiGenerator, Mortm5SamplingConfig

    # Verify preset string resolution for "octuple_160m" and "symic_g4l4_160m"
    assert MORTM5Args("octuple_160m").model_mode == "octuple"
    assert MORTM5Args("symic_g4l4_160m").model_mode == "grouped"

    cfg = MORTM5Args("octuple_160m").to_dict()
    cfg.update({"d_model": 48, "d_layer": 2, "num_heads": 4, "dim_feedforward": 128, "position_length": 64})
    cfg_path = tmp_path / "oct_infer.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    model = build_mortm5_ablation_model(MORTM5Args(cfg)).cpu().eval()
    ckpt_path = tmp_path / "oct_infer.pt"
    torch.save({"model_state_dict": model.state_dict()}, str(ckpt_path))

    gen_engine = Mortm5MidiGenerator.from_pretrained(
        ablation=str(cfg_path),
        ckpt_path=str(ckpt_path),
        vq_ckpt_path=None,
        device=torch.device("cpu"),
    )
    assert gen_engine.is_symbolic is True
    assert gen_engine.vq_model is None

    # Unprompted MIDI generation
    out_midi = tmp_path / "unprompted_oct.mid"
    sampling = Mortm5SamplingConfig(temperature=1.0, top_k=16, min_notes=4, max_notes=12)
    ret_path = gen_engine.generate_to_midi(
        output_path=str(out_midi),
        slot_codes=[SLOT_CODE_DRUMS, 0, SLOT_CODE_EMPTY, SLOT_CODE_EMPTY],
        num_chunks=64,
        sampling=sampling,
    )
    assert ret_path == out_midi and out_midi.exists() and out_midi.stat().st_size > 0

    # 4-bar (32-chunk) prompt -> 16-bar (128-chunk, i.e. +96 continuation chunks) via in-memory float32 roll
    # Put 1 note in bar 0, silence in bars 1..3 -> continuation must only add notes at bar >= 4 (chunks >= 32)
    prompt_roll = np.zeros((32, 8, 24, 128), dtype=np.float32)
    prompt_roll[0, 2, 0, 60] = 0.85
    prompt_roll[0, 3, 0:12, 60] = 1.0
    cont_probs = gen_engine.generate_chunks(
        slot_codes=[SLOT_CODE_DRUMS, 0, SLOT_CODE_EMPTY, SLOT_CODE_EMPTY],
        num_chunks=96,
        sampling=Mortm5SamplingConfig(temperature=1.0, top_k=16, min_notes=3, max_notes=10),
        prompt_chunks_f32=prompt_roll,
    )
    assert cont_probs.shape == (96, 8, 128, 24)
    assert float(cont_probs.max()) > 0.0
    # Inactive slots 2 and 3 (channels 4..7) must be strictly zeroed
    assert float(cont_probs[:, 4:8, :, :].max()) == 0.0

    # Symbolic generate_to_midi with [N, 8, 24, 128] prompt_chunks_f32 AND prompt_midi_path
    from mortm.utils.convert_pianoroll import extract_quantized_notes_and_slots

    cont_midi_f32 = tmp_path / "cont_oct_f32.mid"
    gen_engine.generate_to_midi(
        output_path=str(cont_midi_f32),
        slot_codes=[SLOT_CODE_DRUMS, 0, SLOT_CODE_EMPTY, SLOT_CODE_EMPTY],
        num_chunks=96,
        sampling=Mortm5SamplingConfig(temperature=1.0, top_k=16, min_notes=3, max_notes=10),
        prompt_chunks_f32=prompt_roll,
    )
    _, notes_f32, _ = extract_quantized_notes_and_slots(cont_midi_f32)
    # Prompt bar 0 note (start=0, pitch=60) preserved, bars 1..3 (steps 192..767) remain silent, new notes at step >= 768
    assert np.any((notes_f32[:, 1] == 0) & (notes_f32[:, 3] == 60))
    assert not np.any((notes_f32[:, 1] >= STEPS_PER_BAR) & (notes_f32[:, 1] < 4 * STEPS_PER_BAR))
    assert np.any(notes_f32[:, 1] >= 4 * STEPS_PER_BAR)

    prompt_midi_file = tmp_path / "prompt_src.mid"
    prompt_roll_6bar = np.zeros((48, 8, 24, 128), dtype=np.float32)
    prompt_roll_6bar[0, 2, 0, 62] = 0.85
    prompt_roll_6bar[0, 3, 0:12, 62] = 1.0
    prompt_roll_6bar[47, 0, 0, 36] = 0.9
    prompt_roll_6bar[47, 1, 0:24, 36] = 1.0
    chunks_to_midi(
        prompt_roll_6bar,
        meta=slot_codes_to_roll_meta([SLOT_CODE_DRUMS, 24, SLOT_CODE_EMPTY, SLOT_CODE_EMPTY]),
        output_path=str(prompt_midi_file),
    )

    cont_midi_path = tmp_path / "cont_oct_midi.mid"
    gen_engine.generate_to_midi(
        output_path=str(cont_midi_path),
        slot_codes=[SLOT_CODE_EMPTY] * 4,  # Overridden by prompt_midi_file's slot_codes [DRUMS, 24, -1, -1]
        num_chunks=96,
        sampling=Mortm5SamplingConfig(temperature=1.0, top_k=16, min_notes=3, max_notes=10),
        prompt_midi_path=str(prompt_midi_file),
        prompt_chunks_limit=32,  # 4 bars = 32 chunks
    )
    _, notes_pm, slots_pm = extract_quantized_notes_and_slots(cont_midi_path)
    assert int(slots_pm[1]) == 24
    assert np.any((notes_pm[:, 1] == 0) & (notes_pm[:, 3] == 62))
    assert np.any(notes_pm[:, 1] >= 4 * STEPS_PER_BAR)

    # 8-bar (64-chunk) prompt -> 24-bar (192-chunk, i.e. +128 continuation chunks) sliding-window rollout (§5.2)
    prompt_8bar = np.zeros((64, 8, 24, 128), dtype=np.float32)
    prompt_8bar[0, 2, 0, 64] = 0.85
    prompt_8bar[0, 3, 0:12, 64] = 1.0
    cont_24bar_probs = gen_engine.generate_chunks(
        slot_codes=[SLOT_CODE_DRUMS, 0, SLOT_CODE_EMPTY, SLOT_CODE_EMPTY],
        num_chunks=128,
        sampling=Mortm5SamplingConfig(temperature=1.0, top_k=16, min_notes=6, max_notes=20),
        prompt_chunks_f32=prompt_8bar,
    )
    assert cont_24bar_probs.shape == (128, 8, 128, 24)
    # Both window 1 (bars 8..15 => probs[:64]) AND window 2 beyond bar 16 (bars 16..23 => probs[64:]) must be active
    assert float(cont_24bar_probs[:64].max()) > 0.0
    assert float(cont_24bar_probs[64:].max()) > 0.0

    cont_24bar_midi = tmp_path / "cont_oct_24bar.mid"
    gen_engine.generate_to_midi(
        output_path=str(cont_24bar_midi),
        slot_codes=[SLOT_CODE_DRUMS, 0, SLOT_CODE_EMPTY, SLOT_CODE_EMPTY],
        num_chunks=128,
        sampling=Mortm5SamplingConfig(temperature=1.0, top_k=16, min_notes=6, max_notes=20),
        prompt_chunks_f32=prompt_8bar,
    )
    _, notes_24b, _ = extract_quantized_notes_and_slots(cont_24bar_midi)
    assert np.any((notes_24b[:, 1] == 0) & (notes_24b[:, 3] == 64))
    assert np.any(notes_24b[:, 1] >= 16 * STEPS_PER_BAR)

    # 16-bar (128-chunk) prompt -> 24-bar (+64 continuation chunks) must never clamp into bar 15 of the prompt
    prompt_16bar = np.zeros((128, 8, 24, 128), dtype=np.float32)
    prompt_16bar[0, 2, 0, 67] = 0.85
    prompt_16bar[0, 3, 0:12, 67] = 1.0
    probs_after_16bar = gen_engine.generate_chunks(
        slot_codes=[SLOT_CODE_DRUMS, 0, SLOT_CODE_EMPTY, SLOT_CODE_EMPTY],
        num_chunks=64,
        sampling=Mortm5SamplingConfig(temperature=1.0, top_k=16, min_notes=3, max_notes=12),
        prompt_chunks_f32=prompt_16bar,
    )
    assert probs_after_16bar.shape == (64, 8, 128, 24)
    assert float(probs_after_16bar.max()) > 0.0

    # Non-bar-aligned half-beat prompt (prompt_len=5 chunks = 120 steps, num_chunks=1 => steps 120..143, §5.3)
    hb_prompt = np.zeros((5, 8, 24, 128), dtype=np.float32)
    hb_prompt[0, 2, 0, 60] = 0.85
    hb_prompt[0, 3, 0:12, 60] = 1.0  # Note at step 0, chunks 1..4 (steps 24..119) silent
    hb_probs = gen_engine.generate_chunks(
        slot_codes=[SLOT_CODE_DRUMS, 0, SLOT_CODE_EMPTY, SLOT_CODE_EMPTY],
        num_chunks=1,
        sampling=Mortm5SamplingConfig(temperature=1.0, top_k=16, min_notes=2, max_notes=6),
        prompt_chunks_f32=hb_prompt,
    )
    assert hb_probs.shape == (1, 8, 128, 24)
    assert float(hb_probs.max()) > 0.0

    hb_midi = tmp_path / "cont_oct_halfbeat.mid"
    gen_engine.generate_to_midi(
        output_path=str(hb_midi),
        slot_codes=[SLOT_CODE_DRUMS, 0, SLOT_CODE_EMPTY, SLOT_CODE_EMPTY],
        num_chunks=1,
        sampling=Mortm5SamplingConfig(temperature=1.0, top_k=16, min_notes=2, max_notes=6),
        prompt_chunks_f32=hb_prompt,
    )
    _, notes_hb, _ = extract_quantized_notes_and_slots(hb_midi)
    assert np.any((notes_hb[:, 1] == 0) & (notes_hb[:, 3] == 60))
    # Prompt's silent steps 24..119 must remain 100% untouched, and new notes must start strictly inside [120, 144)
    assert not np.any((notes_hb[:, 1] >= CHUNK_STEPS) & (notes_hb[:, 1] < 5 * CHUNK_STEPS))
    assert np.any((notes_hb[:, 1] >= 5 * CHUNK_STEPS) & (notes_hb[:, 1] < 6 * CHUNK_STEPS))

    # Polymorphic RVQ (MidiVision + AblationTransformer2) check for [N, 8, 24, 128] chunks & prompt_midi_path
    from mortm.models.mortm5 import MidiVision
    from mortm.utils.mortm5_infer import Mortm5PromptEncoder

    vq_cfg = MORTM5Args("configs/models/mortm5/m4_deeprvq_g4l4_d128_k2048.json").to_dict()
    vq_cfg.update({"first_channel": 8, "max_channel": 16, "bottle_channel": 8, "encoder_wout": 32, "codebook_size": 32})
    vq_model = MidiVision(MORTM5Args(vq_cfg), sync_codebook=False).cpu().eval()

    rvq_cfg = MORTM5Args("symic_g4l4_160m").to_dict()
    rvq_cfg.update(
        {
            "d_model": 32,
            "d_layer": 1,
            "num_heads": 4,
            "dim_feedforward": 64,
            "position_length": 128,
            "codebook_size": 32,
            "vocab_size": 133 + 4 * 4 * 32,
        }
    )
    rvq_model = build_mortm5_ablation_model(MORTM5Args(rvq_cfg)).cpu().eval()
    rvq_gen = Mortm5MidiGenerator(vq_model=vq_model, model=rvq_model, device=torch.device("cpu"))
    assert rvq_gen.is_symbolic is False

    # 1) Mortm5PromptEncoder accepts [N, 8, 24, 128] directly without shape mismatch
    enc_p = Mortm5PromptEncoder.encode_chunks_to_prompt(
        prompt_roll[:8],
        vq_model=vq_model,
        model=rvq_model,
        slot_codes=[SLOT_CODE_DRUMS, 0, SLOT_CODE_EMPTY, SLOT_CODE_EMPTY],
    )
    assert enc_p.shape == (1, 5 + 8 * 4, 4)

    # 2) RVQ generate_to_midi accepts [N, 8, 24, 128] prompt_chunks_f32 AND prompt_midi_path + prompt_chunks_limit
    rvq_out_f32 = tmp_path / "rvq_cont_f32.mid"
    rvq_gen.generate_to_midi(
        output_path=str(rvq_out_f32),
        slot_codes=[SLOT_CODE_DRUMS, 0, SLOT_CODE_EMPTY, SLOT_CODE_EMPTY],
        num_chunks=8,
        prompt_chunks_f32=prompt_roll[:8],
    )
    assert rvq_out_f32.exists() and rvq_out_f32.stat().st_size > 0

    rvq_out_pm = tmp_path / "rvq_cont_pm.mid"
    rvq_gen.generate_to_midi(
        output_path=str(rvq_out_pm),
        slot_codes=[SLOT_CODE_EMPTY] * 4,
        num_chunks=8,
        prompt_midi_path=str(prompt_midi_file),
        prompt_chunks_limit=8,
    )
    assert rvq_out_pm.exists() and rvq_out_pm.stat().st_size > 0

    # 3) RVQ sliding-window continuation beyond window horizon
    rvq_sw_probs = rvq_gen.generate_chunks(
        slot_codes=[SLOT_CODE_DRUMS, 0, SLOT_CODE_EMPTY, SLOT_CODE_EMPTY],
        num_chunks=16,
        sampling=Mortm5SamplingConfig(temperature=1.0, top_k=8, sliding_window_chunks=16),
        prompt_chunks_f32=prompt_roll[:8],
    )
    assert rvq_sw_probs.shape == (16, 8, 128, 24)


def test_local_train_mortm5_octuple_dynamic_packer_and_non_divisible_steps(tmp_path: Path):
    from mortm.train.train import train_mortm5

    midi_dir = tmp_path / "midis"
    midi_dir.mkdir(parents=True, exist_ok=True)
    midi_paths = []
    # 5 valid 16-chunk songs -> 5 segments (not divisible by batch_size=2 * accum=2 = 4!)
    for idx in range(5):
        roll = np.zeros((16, 8, 24, 128), dtype=np.float32)
        roll[0, 0, 0, 36] = 0.9
        roll[0, 1, 0:6, 36] = 1.0
        roll[15, 2, 0, 60 + idx] = 0.8
        roll[15, 3, 0:24, 60 + idx] = 1.0
        mp_str = str(midi_dir / f"s_{idx}.mid")
        chunks_to_midi(
            roll,
            meta=slot_codes_to_roll_meta([SLOT_CODE_DRUMS, 0, SLOT_CODE_EMPTY, SLOT_CODE_EMPTY]),
            output_path=mp_str,
        )
        midi_paths.append(mp_str)

    split_json = tmp_path / "split.json"
    split_json.write_text(json.dumps(midi_paths), encoding="utf-8")
    tok_root = tmp_path / "tokens"
    conv = SymbolicCorpusConverter(
        SymbolicCorpusConfig(
            tokenizer_key="octuple_mmt",
            out_dir=str(tok_root),
            num_workers=0,
            verify_m4_dir=None,
        )
    )
    for sp in ("train", "val"):
        conv.convert_split(sp, str(split_json))

    oct_tok_dir = tok_root / "octuple_mmt"
    m_cfg = MORTM5Args("octuple_160m").to_dict()
    m_cfg.update({"d_model": 32, "d_layer": 1, "num_heads": 4, "dim_feedforward": 64, "position_length": 32})
    t_cfg = {
        "optimizer": "muon",
        "lr_param": 0.001,
        "big_batch_size": 8,
        "batch_size": 2,
        "accumulation_steps": 2,
        "max_micro_steps": 48,
        "num_epochs": 1,
        "val_total_tokens": 10,
        "scheduler": {"type": "wsd", "warmup_steps": 1, "stable_ratio": 0.5, "total_steps": 0, "auto_total_steps": True},
    }
    save_dir = tmp_path / "save"
    train_mortm5(
        model_config=m_cfg,
        train_config=t_cfg,
        token_directory=str(oct_tok_dir),
        save_directory=str(save_dir),
        version="local_oct_v1",
        project_name=None,
        resume=True,
    )
    from mortm.models.modules.config import resolve_ablation_subdir

    sub = resolve_ablation_subdir(m_cfg["name"])
    ckpt_path = save_dir / sub / f"{m_cfg['name']}.local_oct_v1.checkpoint.pt"
    assert ckpt_path.exists()
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    # ceil(5 / (2 * 2)) = 2 optimizer steps (trailing 5th segment NOT dropped!)
    assert ckpt["optimizer_step"] == 2
    assert ckpt["verification_loss"] > 0.0


def test_remote_hybrid_trainer_fresh_val_loss_and_converter_segment_parity_and_heal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import colab_gpu_worker as worker
    from ConvertSymbolicTokens import SymbolicSplitWriter
    from run_remote_mortm5 import RemoteMORTM5HybridTrainer, _pack_csr_sequences_zlib

    # 1. Build synthetic MIDI split + mock m4_g4l4 reference WITH offsets.npy to test 1-to-1 segment parity & resume heal
    midi_dir = tmp_path / "midis"
    midi_dir.mkdir(parents=True, exist_ok=True)
    midi_paths = []
    chunk_lens = [16, 140, 24]  # 1 + 2 + 1 = 4 segments total (> big_batch_size=2 for Mode-B val truncation check!)
    for idx, n_c in enumerate(chunk_lens):
        roll = np.zeros((n_c, 8, 24, 128), dtype=np.float32)
        roll[0, 0, 0, 36] = 0.9
        roll[0, 1, 0:6, 36] = 1.0
        # Sustain to end of chunk n_c - 1 so MIDI duration equals exactly n_c chunks
        roll[n_c - 1, 2, 0, 60] = 0.85
        roll[n_c - 1, 3, 0:24, 60] = 1.0
        mp_str = str(midi_dir / f"m_{idx}.mid")
        chunks_to_midi(
            roll,
            meta=slot_codes_to_roll_meta([SLOT_CODE_DRUMS, 0, SLOT_CODE_EMPTY, SLOT_CODE_EMPTY]),
            output_path=mp_str,
        )
        midi_paths.append(mp_str)

    split_json = tmp_path / "split.json"
    split_json.write_text(json.dumps(midi_paths), encoding="utf-8")

    ref_m4 = tmp_path / "m4_ref"
    ref_m4.mkdir(parents=True, exist_ok=True)
    for sp in ("train", "val"):
        offs = []
        cur = 0
        for n_c in chunk_lens:
            offs.append([cur, n_c * 16])
            cur += n_c * 16
        np.save(str(ref_m4 / f"{sp}_offsets.npy"), np.asarray(offs, dtype=np.int64))
        np.save(
            str(ref_m4 / f"{sp}_slots.npy"),
            np.tile(np.asarray([SLOT_CODE_DRUMS, 0, SLOT_CODE_EMPTY, SLOT_CODE_EMPTY], dtype=np.int16), (3, 1)),
        )
        (ref_m4 / f"{sp}_state.json").write_text(
            json.dumps(
                {
                    "processed_midi_count": 3,
                    "valid_songs": 3,
                    "skipped_songs": 0,
                    "total_chunks": sum(chunk_lens),
                    "tokens_per_chunk": 16,
                }
            ),
            encoding="utf-8",
        )

    out_root = tmp_path / "tokens"
    conv = SymbolicCorpusConverter(
        SymbolicCorpusConfig(
            tokenizer_key="octuple_mmt",
            out_dir=str(out_root),
            num_workers=0,
            verify_m4_dir=str(ref_m4),
        )
    )
    for sp in ("train", "val"):
        conv.convert_split(sp, str(split_json))

    # Corrupt val_tokens.bin with trailing garbage + oversized val_offsets.npy + stray .tmp file, then verify heal on resume
    oct_dir = out_root / "octuple_mmt"
    orig_bin_size = (oct_dir / "val_tokens.bin").stat().st_size
    with open(oct_dir / "val_tokens.bin", "ab") as bf:
        bf.write(b"\xff\xff\xff\xff")
    orig_offs = np.load(oct_dir / "val_offsets.npy")
    np.save(oct_dir / "val_offsets.npy", np.vstack([orig_offs, [[99999, 12]]]))
    stray_tmp = oct_dir / "val_state.json.tmp"
    stray_tmp.write_text("partial", encoding="utf-8")

    healed_writer = SymbolicSplitWriter(
        target_dir=oct_dir,
        split="val",
        tokenizer=conv.tokenizer,
        window_spec=conv.config.window_spec,
        resume=True,
    )
    assert not stray_tmp.exists()
    assert (oct_dir / "val_tokens.bin").stat().st_size == orig_bin_size
    assert len(np.load(oct_dir / "val_offsets.npy")) == 3
    assert healed_writer.valid_songs == 3

    # 2. Mode-B RemoteMORTM5HybridTrainer test: big_batch_size=2 < len(val_ds)=4
    drive_ckpt_root = tmp_path / "drive_ckpts"
    worker_tmp = tmp_path / "worker_tmp"
    monkeypatch.setenv("MORTM5_DRIVE_CKPT_DIR", str(drive_ckpt_root))
    monkeypatch.setenv("MORTM5_WORKER_DIR", str(worker_tmp))

    class _DummyReq:
        def __init__(self, json_data=None, query_params=None, body_bytes=b"", headers=None):
            self._json = json_data or {}
            self.query_params = query_params or {}
            self._body = body_bytes
            self.headers = headers or {"content-type": "application/json"}

        async def json(self):
            return self._json

        async def body(self):
            return self._body

    saved_val_losses = []
    validated_token_counts = []

    class _ModeBInProcessClient:
        def health(self):
            return asyncio.run(worker.health())

        def init_mortm5(self, payload):
            return asyncio.run(worker.mortm5_init(_DummyReq(json_data=payload)))

        def upload_variable_sequences(self, sequences):
            raw = _pack_csr_sequences_zlib(sequences)
            asyncio.run(worker.clear_upload())
            asyncio.run(
                worker.append_upload(_DummyReq(body_bytes=raw, headers={"content-type": "application/octet-stream"}))
            )
            return len(raw)

        def start_outer_batch(self, outer_idx, epoch, window_indices=None):
            return asyncio.run(
                worker.mortm5_start_outer_batch(
                    _DummyReq(json_data={"outer_idx": outer_idx, "epoch": epoch, "window_indices": window_indices})
                )
            )

        def poll_outer_batch(self, cursor=0):
            return asyncio.run(worker.mortm5_poll_outer_batch(_DummyReq(query_params={"cursor": str(cursor)})))

        def validate(self, max_batches=0):
            res = asyncio.run(worker.mortm5_validate(_DummyReq(query_params={"max_batches": str(max_batches)})))
            validated_token_counts.append(int(res["val_tokens"]))
            return res

        def save_checkpoint(
            self,
            outer_idx,
            epoch,
            total_outer,
            val_loss,
            version,
            ablation_id="",
            wandb_id="",
            prepare_download=False,
            milestone="",
            sync_drive=False,
        ):
            saved_val_losses.append(float(val_loss))
            return asyncio.run(
                worker.mortm5_save_ckpt(
                    _DummyReq(
                        query_params={
                            "outer": str(outer_idx),
                            "epoch": str(epoch),
                            "total_outer": str(total_outer),
                            "val_loss": f"{val_loss:.6f}",
                            "version": version,
                            "ablation_id": ablation_id,
                            "wandb_id": wandb_id,
                            "prepare_download": "0",
                            "milestone": milestone,
                            "sync_drive": "1" if sync_drive else "0",
                        }
                    )
                )
            )

    m_cfg = MORTM5Args("octuple_160m").to_dict()
    m_cfg.update({"d_model": 32, "d_layer": 1, "num_heads": 4, "dim_feedforward": 64, "position_length": 32})
    t_cfg = {
        "optimizer": "muon",
        "lr_param": 0.001,
        "big_batch_size": 2,  # Intentionally smaller than len(val_ds)=4!
        "batch_size": 2,
        "accumulation_steps": 1,
        "max_micro_steps": 48,
        "num_epochs": 1,
        "val_total_tokens": 1,
        "checkpoint_percents": [50, 100],
        "scheduler": {"type": "wsd", "warmup_steps": 1, "stable_ratio": 0.5, "total_steps": 2},
    }

    hybrid_trainer = RemoteMORTM5HybridTrainer(
        client=_ModeBInProcessClient(),
        m_args=MORTM5Args(m_cfg),
        t_args=TrainArgs(t_cfg),
        version="oct_mode_b",
        save_dir=str(tmp_path / "mac_save"),
        colab_split_dir=None,
        mac_split_dir=str(oct_dir),
        stage_to_local=False,
        use_torch_compile=False,
        ablation_id="octuple_160m",
        download_ckpt_to_mac=False,
        val_max_batches=0,
    )
    run_res = hybrid_trainer.run(project_name=None)
    assert run_res["status"] == "completed"
    assert run_res["final_step"] == 2
    # Every single checkpoint save must receive a fresh, positive validation loss (never stale 0.0!)
    assert len(saved_val_losses) >= 2
    assert all(v > 0.0 and np.isfinite(v) for v in saved_val_losses)
    # Full 4-segment validation must match expected total valid target tokens across ALL 4 val segments
    val_ds = MORTM5MemmapDataset.from_split_dir(str(oct_dir), "val", MORTM5Args(m_cfg))
    expected_val_targets = sum((int(val_ds[i].numel()) - 6) for i in range(len(val_ds)))
    assert all(tc == expected_val_targets for tc in validated_token_counts)


