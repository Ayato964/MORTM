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

from ConvertSymbolicTokens import (
    SymbolicCorpusConfig,
    SymbolicCorpusConverter,
    transcode_octuple_split_to_remi,
)
from mortm.models.modules.config import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    INST_EMPTY_TOKEN_ID,
    INST_RVQ_TOKEN_OFFSET,
    MORTM5Args,
    NUM_SLOT_TOKENS,
    PAD_TOKEN_ID,
    REMI_DEFAULT_ATTR_SIZES,
    SLOT_CODE_DRUMS,
    SLOT_CODE_EMPTY,
    resolve_ablation_subdir,
)
from mortm.models.mortm5 import REMITransformer, build_mortm5_ablation_model
from mortm.train.config import TrainArgs
from mortm.train.datasets import (
    DynamicStepBatchPacker,
    MORTM5Collate,
    MORTM5MemmapDataset,
    MORTM5SymbolicMemmapDataset,
    mortm5_collate_fn,
)
from mortm.train.muon import build_muon_optimizer
from mortm.train.noam import get_cosine_schedule_with_warmup
from mortm.utils.convert_pianoroll import (
    CHUNK_STEPS,
    STEPS_PER_BAR,
    chunks_to_midi,
    slot_codes_to_roll_meta,
)
from mortm.utils.mortm5_infer import Mortm5MidiGenerator, Mortm5SamplingConfig
from mortm.utils.symbolic_tokenizers import (
    DEFAULT_REMI_SPEC,
    MultitrackREMITokenizer,
    OctupleMMTTokenizer,
    SYMBOLIC_TOKENIZER_REGISTRY,
    build_symbolic_tokenizer,
    extract_segment_canonical_notes,
)


def _make_tiny_remi_args(**overrides) -> MORTM5Args:
    cfg = {
        "name": "MORTM5_REMI_Tiny",
        "tokenizer_key": "multitrack_remi",
        "model_mode": "remi",
        "symbolic_vocab": True,
        "remi_attr_sizes": list(REMI_DEFAULT_ATTR_SIZES),
        "use_instrument_tokens": True,
        "token_offset": 133,
        "vocab_size": 802,
        "d_model": 64,
        "d_layer": 2,
        "num_heads": 4,
        "dim_feedforward": 192,
        "dropout": 0.0,
        "position_length": 32,  # Intentionally small to exercise dynamic RoPE & KV-cache growth
        "grad_checkpoint_tokens": 16384,
        "norm_type": "rmsnorm",
        "use_rope": True,
        "use_silu": True,
        "use_bias": False,
        "use_moe_decoder": False,
    }
    cfg.update(overrides)
    return MORTM5Args(cfg)


def _assert_valid_remi_grammar(
    seq_1d: torch.Tensor,
    model: REMITransformer,
    max_bars: int = 16,
    active_slots: list[bool] | None = None,
) -> tuple[int, int]:
    """Strictly validate that a 1D REMI token sequence obeys the Multitrack-REMI FSM grammar."""
    toks = [int(t) for t in seq_1d.detach().cpu().tolist() if int(t) != PAD_TOKEN_ID]
    assert len(toks) >= 6, f"Sequence too short: {toks}"
    assert toks[0] == BOS_TOKEN_ID
    inst_toks = toks[1:5]
    derived_active = [t != INST_EMPTY_TOKEN_ID for t in inst_toks]
    if active_slots is not None:
        assert derived_active == active_slots

    body = toks[5:]
    state = "BEFORE_BAR"
    cur_bar = -1
    last_pos = -1
    note_count = 0

    for idx, t in enumerate(body):
        if t == EOS_TOKEN_ID:
            assert state in ("AFTER_BAR", "AFTER_VEL"), (
                f"Illegal <EOS> before first <BAR> or inside incomplete note tuple at idx={idx}, state={state}"
            )
            assert idx == len(body) - 1, f"Tokens found after <EOS> at idx={idx}: {body[idx:]}"
            return cur_bar + 1, note_count

        if state == "BEFORE_BAR":
            assert t == model.bar_token_id, f"Expected <BAR> at body start, got {t}"
            cur_bar += 1
            last_pos = -1
            state = "AFTER_BAR"
        elif state == "AFTER_BAR":
            if t == model.bar_token_id:
                cur_bar += 1
                last_pos = -1
                state = "AFTER_BAR"
            else:
                assert model.pos_offset <= t < model.slot_offset, f"Expected <BAR> or <POS>, got {t}"
                last_pos = t - model.pos_offset
                state = "AFTER_POS"
        elif state == "AFTER_POS":
            assert model.slot_offset <= t < model.pitch_offset, f"Expected <SLOT>, got {t}"
            slot_i = t - model.slot_offset
            if any(derived_active):
                assert derived_active[slot_i], f"Inactive slot {slot_i} emitted when active={derived_active}"
            state = "AFTER_SLOT"
        elif state == "AFTER_SLOT":
            assert model.pitch_offset <= t < model.dur_offset, f"Expected <PITCH>, got {t}"
            state = "AFTER_PITCH"
        elif state == "AFTER_PITCH":
            assert model.dur_offset <= t < model.vel_offset, f"Expected <DUR>, got {t}"
            state = "AFTER_DUR"
        elif state == "AFTER_DUR":
            assert model.vel_offset <= t < model.vel_offset + model.vel_size, f"Expected <VEL>, got {t}"
            note_count += 1
            state = "AFTER_VEL"
        elif state == "AFTER_VEL":
            if t == model.bar_token_id:
                cur_bar += 1
                last_pos = -1
                state = "AFTER_BAR"
            elif model.pos_offset <= t < model.slot_offset:
                new_pos = t - model.pos_offset
                assert new_pos > last_pos, f"Non-monotonic <POS>: {new_pos} <= {last_pos} in bar {cur_bar}"
                last_pos = new_pos
                state = "AFTER_POS"
            elif model.slot_offset <= t < model.pitch_offset:
                slot_i = t - model.slot_offset
                if any(derived_active):
                    assert derived_active[slot_i], f"Inactive slot {slot_i} emitted when active={derived_active}"
                state = "AFTER_SLOT"
            else:
                raise AssertionError(f"Illegal token {t} after <VEL> at idx={idx}")

        assert cur_bar < max_bars, f"Bar count exceeded max_bars={max_bars} (cur_bar={cur_bar})"

    assert state in ("AFTER_BAR", "AFTER_VEL"), f"Ended before <BAR> or mid-note in state={state}"
    return cur_bar + 1, note_count


def test_remi_vocab_spec_roundtrip_and_octuple_bijection():
    assert DEFAULT_REMI_SPEC.vocab_size == 802
    assert DEFAULT_REMI_SPEC.attr_offsets == (133, 134, 326, 330, 458, 770)
    assert DEFAULT_REMI_SPEC.attr_limits == (134, 326, 330, 458, 770, 802)
    assert "remi" in SYMBOLIC_TOKENIZER_REGISTRY
    assert "multitrack_remi" in SYMBOLIC_TOKENIZER_REGISTRY

    remi_tok = build_symbolic_tokenizer("multitrack_remi")
    oct_tok = OctupleMMTTokenizer()
    assert isinstance(remi_tok, MultitrackREMITokenizer)
    assert remi_tok.vocab_size == 802
    assert remi_tok.step_unit_width == 1

    # Construct multi-slot notes with shared (bar, pos) chords, empty middle bars, and cross-segment sustain
    s0 = 128 * CHUNK_STEPS
    notes_arr = np.asarray(
        [
            # Cross-boundary sustain clipped to (bar=0, pos=0)
            [1, s0 - 48, s0 + 48, 60, 100],
            # Simultaneous chord note at same (bar=0, pos=0) -> shares single <POS_0>!
            [2, s0, s0 + 96, 64, 92],
            # Duplicate collision at (bar=0, pos=24, slot=1, pitch=67) -> merged
            [1, s0 + 24, s0 + 48, 67, 70],
            [1, s0 + 24, s0 + 72, 67, 116],
            # Empty bars 1 and 2, then drum + bass notes in bar 3 (num_chunks=32 => 4 bars)
            [0, s0 + 3 * STEPS_PER_BAR + 48, s0 + 3 * STEPS_PER_BAR + 54, 36, 120],
            [2, s0 + 3 * STEPS_PER_BAR + 48, s0 + 3 * STEPS_PER_BAR + 144, 43, 88],
        ],
        dtype=np.int32,
    )
    slot_codes = np.asarray([SLOT_CODE_DRUMS, 0, 33, SLOT_CODE_EMPTY], dtype=np.int16)

    canon = extract_segment_canonical_notes(notes_arr, chunk_start=128, chunk_end=160, max_bars=16)
    assert len(canon) == 5  # 2 at (0,0) + 1 merged at (0,24) + 2 at (3,48)

    remi_body = remi_tok.encode_canonical_segment_notes(canon, num_chunks=32)
    oct_body = oct_tok.encode_canonical_segment_notes(canon, num_chunks=32)
    transcoded_body = remi_tok.transcode_octuple_body_to_remi(oct_body, num_chunks=32, out_dtype=np.uint16)

    # Exact body formula: 4 <BAR> + 3 unique (bar, pos) (<POS> at (0,0), (0,24), (3,48)) + 4 * 5 notes = 27 tokens
    assert remi_body.size == 4 + 3 + 4 * 5
    assert np.array_equal(remi_body, transcoded_body)

    remi_full = torch.from_numpy(remi_tok.build_full_sequence(remi_body, slot_codes=slot_codes))
    oct_full = torch.from_numpy(oct_tok.build_full_sequence(oct_body, slot_codes=slot_codes))
    assert remi_full.numel() == 1 + NUM_SLOT_TOKENS + 27 + 1

    remi_notes, remi_slots = remi_tok.decode_tokens_to_notes(remi_full, num_chunks=32)
    oct_notes, oct_slots = oct_tok.decode_tokens_to_notes(oct_full, num_chunks=32)
    assert np.array_equal(remi_slots, slot_codes)
    assert np.array_equal(remi_notes, oct_notes)

    remi_u8, _ = remi_tok.decode_tokens_to_chunks_u8(remi_full, num_chunks=32)
    oct_u8, _ = oct_tok.decode_tokens_to_chunks_u8(oct_full, num_chunks=32)
    assert np.array_equal(remi_u8, oct_u8)

    # Silent segment (0 notes, 16 chunks = 2 bars) -> 2 <BAR> tokens -> 0 decoded notes
    empty_remi = remi_tok.encode_canonical_segment_notes(np.zeros((0, 6), dtype=np.int32), num_chunks=16)
    empty_trans = remi_tok.transcode_octuple_body_to_remi(np.zeros((0,), dtype=np.uint16), num_chunks=16, out_dtype=np.uint16)
    assert empty_remi.size == 2 and np.all(empty_remi == DEFAULT_REMI_SPEC.bar_token_id)
    assert np.array_equal(empty_remi, empty_trans)
    empty_full = torch.from_numpy(remi_tok.build_full_sequence(empty_remi, slot_codes=slot_codes))
    dec_empty_notes, dec_empty_slots = remi_tok.decode_tokens_to_notes(empty_full, num_chunks=16)
    assert len(dec_empty_notes) == 0
    assert np.array_equal(dec_empty_slots, slot_codes)


def test_remi_160m_exact_parameter_counts_muon_routing_and_isoflops_schedule():
    remi_args = MORTM5Args("configs/models/mortm5/symic_main/remi_160m.json")
    oct_args = MORTM5Args("configs/models/mortm5/symic_main/octuple_160m.json")
    sym_args = MORTM5Args("configs/models/mortm5/symic_main/symic_g4l4_160m.json")

    for alias in ("remi", "multitrack_remi", "remi_160m"):
        alias_args = MORTM5Args(alias)
        assert alias_args.model_mode == "remi"
        assert alias_args.vocab_size == 802
        assert alias_args.step_unit_width == 1
        assert resolve_ablation_subdir(model_name=alias_args.name, ablation_id=alias) == "remi_160m"

    assert remi_args.recompute_vocab() == 802
    assert remi_args.grad_checkpoint_tokens == 16384
    assert oct_args.grad_checkpoint_tokens == 16384

    with torch.device("meta"):
        remi_meta = build_mortm5_ablation_model(remi_args)
        oct_meta = build_mortm5_ablation_model(oct_args)
        sym_meta = build_mortm5_ablation_model(sym_args)

    assert isinstance(remi_meta, REMITransformer)
    assert remi_meta.Wout.bias is None

    remi_non_emb = remi_meta.get_non_embedding_param_count()
    oct_non_emb = oct_meta.get_non_embedding_param_count()
    sym_non_emb = sym_meta.get_non_embedding_param_count()
    assert remi_non_emb == 163_603_456
    assert remi_non_emb == oct_non_emb == sym_non_emb

    remi_total = sum(p.numel() for p in remi_meta.parameters() if p.requires_grad)
    assert remi_total == 165_245_952
    assert remi_total - remi_non_emb == 1_642_496  # 2 * 802 * 1024

    # Verify Muon vs AdamW parameter routing
    tiny_model = REMITransformer(_make_tiny_remi_args()).cpu()
    t_main = TrainArgs("configs/train/mortm5/remi_main_160m.json")
    t_iso = TrainArgs("configs/train/mortm5/remi_main_160m_isoflops.json")

    assert t_main.max_micro_steps == 65536 and t_main.max_steps == 0
    assert t_main.checkpoint_percents == [24, 50, 75, 100]
    assert t_iso.max_micro_steps == 65536 and t_iso.max_steps == 3865
    assert t_iso.scheduler["auto_total_steps"] is False
    assert t_iso.scheduler["total_steps"] == 3865

    # Verify exact LR trajectory match between 1-Epoch WSD and Iso-FLOPs WSD across steps 0..3594 (covering pct24=3520)
    fn_main = get_cosine_schedule_with_warmup(
        warmup_steps=int(0.05 * 14610),
        total_steps=14610,
        min_lr_ratio=0.05,
        stable_steps=int(0.88 * 14610),
    )
    fn_iso = get_cosine_schedule_with_warmup(
        warmup_steps=t_iso.scheduler["warmup_steps"],
        total_steps=t_iso.scheduler["total_steps"],
        min_lr_ratio=t_iso.scheduler["min_lr_ratio"],
        stable_steps=t_iso.scheduler["stable_steps"],
    )
    for step in (0, 100, 730, 2000, 3520, 3594):
        assert abs(fn_main(step) - fn_iso(step)) < 1e-12, f"LR mismatch at step={step}"
    assert abs(fn_iso(3865) - 0.05) < 1e-9

    opt = build_muon_optimizer(tiny_model, lr=t_main.lr_param, weight_decay=t_main.weight_decay, verbose=False)
    muon_ids = {id(p) for g in opt.muon.param_groups for p in g["params"]}
    adamw_ids = {id(p) for g in opt.adamw.param_groups for p in g["params"]}
    assert id(tiny_model.embedding.weight) in adamw_ids
    assert id(tiny_model.Wout.weight) in adamw_ids
    assert id(tiny_model.embedding.weight) not in muon_ids
    assert id(tiny_model.Wout.weight) not in muon_ids
    assert len(muon_ids) == 2 * 5


def test_real_gmd_val_on_the_fly_and_offline_transcoding_parity():
    oct_dir = ROOT_DIR / "out" / "mortm5_tokens" / "octuple_mmt"
    if not (oct_dir / "val_tokens.bin").exists():
        pytest.skip("out/mortm5_tokens/octuple_mmt/val_tokens.bin not available")

    remi_args = MORTM5Args("remi_160m")
    oct_args = MORTM5Args("octuple_160m")

    # 1. Sibling directory auto-fallback (`multitrack_remi` -> `octuple_mmt`) + On-the-Fly transcoding
    non_existent_remi_dir = str(ROOT_DIR / "out" / "mortm5_tokens" / "multitrack_remi")
    otf_ds = MORTM5MemmapDataset.from_split_dir(non_existent_remi_dir, "val", remi_args)
    oct_ds = MORTM5SymbolicMemmapDataset.from_split_dir(str(oct_dir), "val", oct_args)

    assert isinstance(otf_ds, MORTM5SymbolicMemmapDataset)
    assert otf_ds.transcode_from_octuple is True
    assert otf_ds.source_format == "octuple_mmt"
    assert len(otf_ds) == len(oct_ds) == 22_607

    # Check sampled segments across full val split for exact note round-trip identity
    remi_tok = MultitrackREMITokenizer()
    oct_tok = OctupleMMTTokenizer()
    sample_indices = [0, 1, 17, 250, 1024, 5000, 12345, 22606]
    # Also include any 0-note segment if present in val
    zero_note_idxs = np.nonzero(oct_ds.segments[:, 1] == 0)[0]
    if len(zero_note_idxs) > 0:
        sample_indices.append(int(zero_note_idxs[0]))

    for idx in sample_indices:
        r_seq = otf_ds[idx]
        o_seq = oct_ds[idx]
        n_chunks = otf_ds.get_segment_num_chunks(idx)
        r_notes, r_slots = remi_tok.decode_tokens_to_notes(r_seq, num_chunks=n_chunks)
        o_notes, o_slots = oct_tok.decode_tokens_to_notes(o_seq, num_chunks=n_chunks)
        assert np.array_equal(r_slots, o_slots)
        assert np.array_equal(r_notes, o_notes), f"Note mismatch at val segment idx={idx}"

    # 2. Offline binary transcoder (`transcode_octuple_split_to_remi`) bit-exact equality vs On-the-Fly
    with tempfile.TemporaryDirectory() as tmp:
        st = transcode_octuple_split_to_remi(
            octuple_dir=oct_dir,
            out_dir=tmp,
            split="val",
            tokenizer_key="multitrack_remi",
            limit_segments=64,
        )
        assert st["total_segments"] == 64
        offline_ds = MORTM5SymbolicMemmapDataset.from_split_dir(
            os.path.join(tmp, "multitrack_remi"), "val", remi_args
        )
        assert offline_ds.source_format == "multitrack_remi"
        assert offline_ds.transcode_from_octuple is False
        assert len(offline_ds) == 64
        for idx in range(64):
            assert torch.equal(offline_ds[idx], otf_ds[idx]), f"Offline vs On-the-Fly mismatch at seg {idx}"

    # 3. Direct MIDI -> multitrack_remi conversion via SymbolicCorpusConverter vs MIDI -> octuple_mmt -> remi
    eval_json = ROOT_DIR / "data" / "gmd" / "eval.json"
    if eval_json.exists():
        with open(eval_json, "r", encoding="utf-8") as f:
            eval_paths = [p for p in json.load(f)[:8] if os.path.exists(p)][:4]
        if eval_paths:
            with tempfile.TemporaryDirectory() as tmp:
                mini_json = os.path.join(tmp, "mini_eval.json")
                with open(mini_json, "w", encoding="utf-8") as mf:
                    json.dump(eval_paths, mf)
                for tk in ("octuple_mmt", "multitrack_remi"):
                    conv = SymbolicCorpusConverter(
                        SymbolicCorpusConfig(
                            tokenizer_key=tk,
                            out_dir=tmp,
                            num_workers=0,
                            resume=False,
                            verify_m4_dir=None,
                        )
                    )
                    conv.convert_split("val", mini_json)

                ds_direct_remi = MORTM5SymbolicMemmapDataset.from_split_dir(
                    os.path.join(tmp, "multitrack_remi"), "val", remi_args
                )
                ds_otf_from_oct = MORTM5SymbolicMemmapDataset.from_split_dir(
                    os.path.join(tmp, "octuple_mmt"), "val", remi_args
                )
                assert len(ds_direct_remi) == len(ds_otf_from_oct) and len(ds_direct_remi) > 0
                for idx in range(len(ds_direct_remi)):
                    assert torch.equal(ds_direct_remi[idx], ds_otf_from_oct[idx])


def test_remi_packer_gradient_invariance_and_grad_checkpoint_equivalence():
    torch.manual_seed(42)
    remi_tok = MultitrackREMITokenizer()
    rng = np.random.RandomState(19)
    seqs = []
    for n_notes, n_chunks in ((18, 128), (5, 64), (11, 96), (0, 32), (24, 128)):
        canon = np.zeros((n_notes, 6), dtype=np.int32)
        if n_notes > 0:
            canon[:, 0] = np.sort(rng.randint(0, max(1, n_chunks // 8), size=n_notes))
            canon[:, 1] = np.sort(rng.randint(0, 192, size=n_notes))
            canon[:, 2] = rng.randint(0, 3, size=n_notes)
            canon[:, 3] = rng.randint(36, 84, size=n_notes)
            canon[:, 4] = rng.randint(6, 96, size=n_notes)
            canon[:, 5] = rng.randint(32, 120, size=n_notes)
        body = remi_tok.encode_canonical_segment_notes(canon, num_chunks=n_chunks)
        full = remi_tok.build_full_sequence(body, slot_codes=[SLOT_CODE_DRUMS, 0, 33, SLOT_CODE_EMPTY])
        seqs.append(torch.from_numpy(full))

    collate = MORTM5Collate(mode="remi", vq_groups=1, pad_token_id=PAD_TOKEN_ID)
    packer = DynamicStepBatchPacker(max_micro_steps=96, max_batch_size=2, unit_width=1, pad_token_id=PAD_TOKEN_ID)
    packed = packer.pack_sequences(seqs, collate_fn=collate)
    assert len(packed.sub_batches) >= 3

    # 1. Exact gradient equivalence: packed sub-batches vs full unsplit batch
    m_ref = REMITransformer(_make_tiny_remi_args()).cpu().train()
    m_pack = REMITransformer(_make_tiny_remi_args()).cpu().train()
    m_pack.load_state_dict(m_ref.state_dict())

    full_in, full_tgt, full_mask = collate(seqs)
    ref_logits = m_ref(full_in, padding_mask=full_mask, is_causal=True)
    ref_loss = F.cross_entropy(
        ref_logits.float().reshape(-1, ref_logits.size(-1)),
        full_tgt.reshape(-1),
        ignore_index=PAD_TOKEN_ID,
        reduction="mean",
    )
    ref_loss.backward()

    denom = float(packed.total_valid_targets)
    pack_loss_sum = 0.0
    for sub_in, sub_tgt, sub_mask in packed.sub_batches:
        sub_logits = m_pack(sub_in, padding_mask=sub_mask, is_causal=True)
        s_loss = F.cross_entropy(
            sub_logits.float().reshape(-1, sub_logits.size(-1)),
            sub_tgt.reshape(-1),
            ignore_index=PAD_TOKEN_ID,
            reduction="sum",
        )
        (s_loss / denom).backward()
        pack_loss_sum += float(s_loss.item())

    assert abs(pack_loss_sum / denom - float(ref_loss.item())) < 1e-5
    for (n1, p1), (n2, p2) in zip(m_ref.named_parameters(), m_pack.named_parameters()):
        assert n1 == n2
        assert torch.allclose(p1.grad.float(), p2.grad.float(), atol=1e-4, rtol=1e-3), f"Grad mismatch on {n1}"

    # 2. Automatic activation checkpointing equivalence when unpadded_tokens > grad_checkpoint_tokens
    m_ckpt = REMITransformer(_make_tiny_remi_args(grad_checkpoint_tokens=32)).cpu().train()
    m_ckpt.load_state_dict(m_ref.state_dict())
    assert int(full_mask.sum().item()) > 32
    ckpt_logits = m_ckpt(full_in, padding_mask=full_mask, is_causal=True)
    assert torch.allclose(ckpt_logits.float(), ref_logits.float(), atol=1e-5)
    ckpt_loss = F.cross_entropy(
        ckpt_logits.float().reshape(-1, ckpt_logits.size(-1)),
        full_tgt.reshape(-1),
        ignore_index=PAD_TOKEN_ID,
        reduction="mean",
    )
    ckpt_loss.backward()
    for (n1, p1), (_, p2) in zip(m_ref.named_parameters(), m_ckpt.named_parameters()):
        assert torch.allclose(p1.grad.float(), p2.grad.float(), atol=1e-5), f"Ckpt grad mismatch on {n1}"


def test_remi_fsm_grammar_kv_cache_parity_and_continuation():
    torch.manual_seed(77)
    model = REMITransformer(_make_tiny_remi_args()).cpu().eval()
    tok = model.get_symbolic_tokenizer()

    # 1. Batched variable-length prompt KV-cache equivalence vs full causal forward
    slots_a = [SLOT_CODE_DRUMS, 0, SLOT_CODE_EMPTY, SLOT_CODE_EMPTY]
    slots_b = [SLOT_CODE_EMPTY, 24, 40, SLOT_CODE_EMPTY]
    notes_a = np.asarray([[0, 0, 24, 36, 100], [1, 48, 96, 60, 90]], dtype=np.int32)
    notes_b = np.asarray([[1, 0, 48, 55, 85], [2, 24, 72, 62, 95], [1, 96, 144, 64, 105]], dtype=np.int32)
    p_a = tok.encode_notes_window_prompt(notes_a, 0, 8, slots_a, append_eos=False).squeeze(0)
    p_b = tok.encode_notes_window_prompt(notes_b, 0, 16, slots_b, append_eos=False).squeeze(0)
    assert p_a.numel() != p_b.numel()
    padded_prompts = torch.nn.utils.rnn.pad_sequence([p_a, p_b], batch_first=True, padding_value=PAD_TOKEN_ID)
    valid_m = padded_prompts != PAD_TOKEN_ID

    model.reset_kv_cache()
    prefill_logits = model(padded_prompts, padding_mask=valid_m, is_causal=True, is_save_cache=True)
    step_tok = torch.tensor([model.bar_token_id, model.bar_token_id], dtype=torch.long)
    kv_step_logits = model._forward_decode_step(step_tok)

    for b_i, raw_p in enumerate((p_a, p_b)):
        cat_seq = torch.cat([raw_p, step_tok[b_i : b_i + 1]], dim=0).unsqueeze(0)
        full_logits = model(cat_seq, padding_mask=torch.ones_like(cat_seq, dtype=torch.bool), is_causal=True)
        assert torch.allclose(
            prefill_logits[b_i, raw_p.numel() - 1].float(),
            full_logits[0, raw_p.numel() - 1].float(),
            atol=1e-3,
        )
        assert torch.allclose(kv_step_logits[b_i].float(), full_logits[0, -1].float(), atol=1e-3)

    # 2. Unconditional FSM generation + active-slot masking + min_notes / max_notes / max_bars
    gen_seq = model.generate(
        max_notes=10,
        max_bars=4,
        min_notes=4,
        min_bars=2,
        slot_codes=slots_a,
        batch_size=2,
        temperature=1.0,
        top_k=16,
    )
    assert gen_seq.size(0) == 2
    for b_i in range(2):
        bars_seen, n_gen = _assert_valid_remi_grammar(
            gen_seq[b_i], model, max_bars=4, active_slots=[True, True, False, False]
        )
        assert 4 <= n_gen <= 10
        assert 2 <= bars_seen <= 4

    # 3. All-empty slots edge case -> terminates cleanly with 0 notes
    empty_slots = [SLOT_CODE_EMPTY, SLOT_CODE_EMPTY, SLOT_CODE_EMPTY, SLOT_CODE_EMPTY]
    empty_gen = model.generate(max_notes=8, max_bars=4, min_notes=4, slot_codes=empty_slots, batch_size=1)
    _, empty_n = _assert_valid_remi_grammar(empty_gen[0], model, max_bars=4, active_slots=[False] * 4)
    assert empty_n == 0

    # 4. Sub-bar half-beat continuation (3 chunks = pos 72 -> +5 chunks = 1 bar total)
    sub_prompt = tok.encode_notes_window_prompt(
        np.asarray([[1, 0, 24, 60, 100]], dtype=np.int32),
        chunk_start=0,
        chunk_end=3,
        slot_codes=slots_a,
        append_eos=False,
    )
    sub_gen = model.generate(
        max_notes=6,
        max_bars=1,
        min_notes=3,
        min_bar=0,
        min_pos=3 * CHUNK_STEPS,
        max_steps=8 * CHUNK_STEPS,
        prompt=sub_prompt,
        slot_codes=slots_a,
        temperature=1.0,
    )
    _assert_valid_remi_grammar(sub_gen[0], model, max_bars=1, active_slots=[True, True, False, False])
    sub_notes, _ = tok.decode_tokens_to_notes(sub_gen[0], num_chunks=8, slot_codes=slots_a)
    assert len(sub_notes) == 1 + 3 or len(sub_notes) >= 4
    new_sub = sub_notes[sub_notes[:, 1] >= 3 * CHUNK_STEPS]
    assert len(new_sub) >= 3
    assert np.all((new_sub[:, 1] >= 72) & (new_sub[:, 1] < 192))

    # 5. Bar-aligned 4-bar -> 8-bar continuation & Mortm5MidiGenerator 8-bar -> 20-bar sliding window
    bar4_prompt = tok.encode_notes_window_prompt(
        np.asarray([[1, 0, 48, 60, 100], [0, 3 * STEPS_PER_BAR + 24, 3 * STEPS_PER_BAR + 48, 36, 110]], dtype=np.int32),
        chunk_start=0,
        chunk_end=32,
        slot_codes=slots_a,
        append_eos=False,
    )
    bar8_gen = model.generate(
        max_notes=6,
        max_bars=8,
        min_notes=3,
        min_bar=4,
        min_pos=0,
        max_steps=8 * STEPS_PER_BAR,
        prompt=bar4_prompt,
        slot_codes=slots_a,
    )
    # First token emitted right after 4-bar prompt MUST be <BAR> (opening bar index 4)
    assert int(bar8_gen[0, bar4_prompt.size(1)].item()) == model.bar_token_id
    _assert_valid_remi_grammar(bar8_gen[0], model, max_bars=8, active_slots=[True, True, False, False])

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_p = os.path.join(tmp, "remi_tiny.pth")
        torch.save(model.state_dict(), ckpt_p)
        cfg_p = os.path.join(tmp, "remi_tiny.json")
        with open(cfg_p, "w", encoding="utf-8") as cf:
            json.dump(model.args.to_dict(), cf)

        gen_service = Mortm5MidiGenerator.from_pretrained(ablation=cfg_p, ckpt_path=ckpt_p, device="cpu")
        assert gen_service.is_symbolic is True

        # 8-bar prompt (64 chunks) + 12-bar continuation (96 chunks) = 20 bars (exercises >16-bar sliding window)
        prompt_f32 = np.zeros((64, 8, 128, 24), dtype=np.float32)
        prompt_f32[0, 2, 60, 0] = 0.85
        prompt_f32[0, 3, 60, :12] = 1.0
        out_midi = os.path.join(tmp, "remi_20bar.mid")
        res_path = gen_service.generate_to_midi(
            output_path=out_midi,
            slot_codes=slots_a,
            num_chunks=96,
            sampling=Mortm5SamplingConfig(temperature=1.0, top_k=16, max_notes=12, min_notes=4),
            prompt_chunks_f32=prompt_f32,
        )
        assert Path(res_path).is_file() and Path(out_midi).stat().st_size > 0

        gen_chunks = gen_service.generate_chunks(
            slot_codes=slots_a,
            num_chunks=96,
            sampling=Mortm5SamplingConfig(temperature=1.0, top_k=16, max_notes=12, min_notes=4),
            prompt_chunks_f32=prompt_f32,
        )
        assert gen_chunks.shape == (96, 8, 128, 24)


def test_max_steps_and_auto_total_steps_in_worker_and_local_train():
    import colab_gpu_worker as worker

    with tempfile.TemporaryDirectory() as tmp:
        # Build a small synthetic octuple_mmt split directory (10 segments), then init REMI via sibling multitrack_remi path
        oct_split_dir = os.path.join(tmp, "octuple_mmt")
        os.makedirs(oct_split_dir, exist_ok=True)
        oct_tok = OctupleMMTTokenizer()

        seg_rows = []
        bin_chunks = []
        cursor = 0
        for s_i in range(10):
            canon = np.asarray([[0, 0, 1, 60 + (s_i % 5), 24, 96]], dtype=np.int32)
            b_u16 = oct_tok.encode_canonical_segment_notes(canon, num_chunks=16)
            bin_chunks.append(b_u16)
            seg_rows.append([cursor, int(b_u16.size), 1, 1, s_i, 16])
            cursor += int(b_u16.size)

        np.concatenate(bin_chunks).astype(np.uint16).tofile(os.path.join(oct_split_dir, "train_tokens.bin"))
        np.save(os.path.join(oct_split_dir, "train_segments.npy"), np.asarray(seg_rows, dtype=np.int64))
        slots_arr = np.tile(np.asarray([[SLOT_CODE_DRUMS, 0, SLOT_CODE_EMPTY, SLOT_CODE_EMPTY]], dtype=np.int16), (10, 1))
        np.save(os.path.join(oct_split_dir, "train_slots.npy"), slots_arr)
        with open(os.path.join(oct_split_dir, "train_state.json"), "w", encoding="utf-8") as sf:
            json.dump({"tokenizer_key": "octuple_mmt"}, sf)

        class _DummyReq:
            headers = {"content-type": "application/json"}

            def __init__(self, data):
                self._data = data

            async def json(self):
                return self._data

        old_env = os.environ.get("MORTM5_WORKER_DIR")
        os.environ["MORTM5_WORKER_DIR"] = os.path.join(tmp, "worker_state")
        try:
            init_payload = {
                "model_config": _make_tiny_remi_args().to_dict(),
                "train_config": {
                    "optimizer": "muon",
                    "lr_param": 1e-3,
                    "big_batch_size": 10,
                    "batch_size": 2,
                    "accumulation_steps": 1,
                    "max_micro_steps": 512,
                    "max_steps": 3,
                    "num_epochs": 1,
                    "shuffle": False,
                    "scheduler": {
                        "type": "wsd",
                        "warmup_steps": 1,
                        "stable_steps": 1,
                        "min_lr_ratio": 0.05,
                        "total_steps": 3,
                        "auto_total_steps": False,
                    },
                },
                "ablation_id": "remi_160m",
                # Intentionally point to non-existent sibling `multitrack_remi` to test worker fallback!
                "colab_split_dir": os.path.join(tmp, "multitrack_remi"),
                "stage_to_local": False,
                "version": "remi_iso_smoke",
                "clean_old_checkpoints": True,
            }
            init_res = asyncio.run(worker.mortm5_init(_DummyReq(init_payload)))
            assert init_res["status"] == "initialized"
            assert init_res["model_class"] == "REMITransformer"
            assert init_res["total_steps"] == 3
            assert init_res["train_windows"] == 10

            # Give all 10 windows (which would normally run 5 steps at batch_size=2, accum=1), verify hard stop at max_steps=3!
            worker.STATE["mortm5_job_results"] = []
            worker.STATE["mortm5_job_done"] = False
            worker.STATE["mortm5_job_error"] = None
            worker._run_mortm5_outer_batch_worker(
                window_indices=list(range(10)),
                uploaded_comp=None,
                outer_idx=0,
                epoch=0,
            )
            assert worker.STATE["mortm5_job_error"] is None
            assert worker.STATE["mortm5_job_done"] is True
            assert worker.STATE["step"] == 3
            assert len(worker.STATE["mortm5_job_results"]) == 3
        finally:
            if old_env is None:
                os.environ.pop("MORTM5_WORKER_DIR", None)
            else:
                os.environ["MORTM5_WORKER_DIR"] = old_env


def test_remote_hybrid_trainer_and_local_train_mortm5_remi_max_steps():
    import urllib.parse
    import colab_gpu_worker as worker
    from mortm.train.train import train_mortm5
    from run_remote_mortm5 import RemoteMORTM5HybridTrainer

    # 1. NaN / all -inf logits safety on REMITransformer._sample_from_logits
    tiny_model = REMITransformer(_make_tiny_remi_args()).cpu().eval()
    nan_logits = torch.full((2, tiny_model.args.vocab_size), float("nan"))
    inf_logits = torch.full((2, tiny_model.args.vocab_size), float("-inf"))
    for bad in (nan_logits, inf_logits):
        s_tok = tiny_model._sample_from_logits(bad, temperature=1.0, top_k=8, top_p=0.9)
        assert s_tok.shape == (2,)
        assert bool(((s_tok >= 0) & (s_tok < tiny_model.args.vocab_size)).all().item())

    with tempfile.TemporaryDirectory() as tmp:
        oct_dir = os.path.join(tmp, "tokens", "octuple_mmt")
        os.makedirs(oct_dir, exist_ok=True)
        oct_tok = OctupleMMTTokenizer()

        for split_name, n_segs in (("train", 10), ("val", 4)):
            seg_rows = []
            bin_chunks = []
            cursor = 0
            for s_i in range(n_segs):
                canon = np.asarray([[0, (s_i * 12) % 192, 1, 60 + (s_i % 7), 24, 96]], dtype=np.int32)
                b_u16 = oct_tok.encode_canonical_segment_notes(canon, num_chunks=16)
                bin_chunks.append(b_u16)
                seg_rows.append([cursor, int(b_u16.size), 1, 1, s_i, 16])
                cursor += int(b_u16.size)
            np.concatenate(bin_chunks).astype(np.uint16).tofile(os.path.join(oct_dir, f"{split_name}_tokens.bin"))
            np.save(os.path.join(oct_dir, f"{split_name}_segments.npy"), np.asarray(seg_rows, dtype=np.int64))
            slots_arr = np.tile(
                np.asarray([[SLOT_CODE_DRUMS, 0, SLOT_CODE_EMPTY, SLOT_CODE_EMPTY]], dtype=np.int16),
                (n_segs, 1),
            )
            np.save(os.path.join(oct_dir, f"{split_name}_slots.npy"), slots_arr)
            with open(os.path.join(oct_dir, f"{split_name}_state.json"), "w", encoding="utf-8") as sf:
                json.dump({"tokenizer_key": "octuple_mmt"}, sf)

        non_existent_remi_dir = os.path.join(tmp, "tokens", "multitrack_remi")

        # 2. Local train_mortm5 with sibling octuple_mmt On-the-Fly transcoding + max_steps=2 (out of 5 steps)
        local_save = os.path.join(tmp, "local_save")
        train_cfg = {
            "optimizer": "muon",
            "lr_param": 1e-3,
            "big_batch_size": 6,
            "batch_size": 2,
            "accumulation_steps": 1,
            "max_micro_steps": 512,
            "max_steps": 2,
            "num_epochs": 2,
            "is_save_training_progress": True,
            "val_total_tokens": 1000000,
            "shuffle": False,
            "scheduler": {
                "type": "wsd",
                "warmup_steps": 1,
                "stable_steps": 0,
                "min_lr_ratio": 0.05,
                "total_steps": 2,
                "auto_total_steps": False,
            },
        }
        trained_m = train_mortm5(
            model_config=_make_tiny_remi_args().to_dict(),
            train_config=train_cfg,
            token_directory=non_existent_remi_dir,
            save_directory=local_save,
            version="local_iso",
            log_scale=False,
            project_name=None,
            resume=False,
        )
        assert isinstance(trained_m, REMITransformer)
        ckpt_file = os.path.join(local_save, "mortm5_remi_tiny", "MORTM5_REMI_Tiny.local_iso.checkpoint.pt")
        assert os.path.isfile(ckpt_file)
        ckpt_meta = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        assert ckpt_meta["optimizer_step"] == 2

        # 3. RemoteMORTM5HybridTrainer window trimming & exact stop at max_steps=3 (outer0=2 steps, outer1 trimmed to 1 step)
        class _InProcessClient:
            def __init__(self):
                self.window_counts = []

            def health(self):
                return asyncio.run(worker.health())

            def start_stage(self, colab_split_dir: str):
                class _R:
                    async def json(self):
                        return {"colab_split_dir": colab_split_dir}

                return asyncio.run(worker.mortm5_start_stage(_R()))

            def poll_stage(self):
                return asyncio.run(worker.mortm5_poll_stage())

            def init_mortm5(self, payload):
                class _R:
                    headers = {"content-type": "application/json"}

                    async def json(self):
                        return payload

                return asyncio.run(worker.mortm5_init(_R()))

            def start_outer_batch(self, outer_idx: int, epoch: int, window_indices=None):
                if window_indices is not None:
                    self.window_counts.append(len(window_indices))

                class _R:
                    headers = {"content-type": "application/json"}
                    query_params = {}

                    async def json(self):
                        return {"outer_idx": outer_idx, "epoch": epoch, "window_indices": window_indices}

                return asyncio.run(worker.mortm5_start_outer_batch(_R()))

            def poll_outer_batch(self, cursor: int):
                class _R:
                    query_params = {"cursor": str(cursor)}

                return asyncio.run(worker.mortm5_poll_outer_batch(_R()))

            def validate(self, max_batches: int = 0):
                class _R:
                    query_params = {"max_batches": str(max_batches)}

                return asyncio.run(worker.mortm5_validate(_R()))

            def save_checkpoint(self, **kwargs):
                q_map = {
                    "outer": str(kwargs["outer_idx"]),
                    "epoch": str(kwargs["epoch"]),
                    "total_outer": str(kwargs["total_outer"]),
                    "val_loss": f"{kwargs['val_loss']:.6f}",
                    "version": str(kwargs["version"]),
                    "ablation_id": str(kwargs.get("ablation_id", "")),
                    "wandb_id": str(kwargs.get("wandb_id", "")),
                    "prepare_download": "1" if kwargs.get("prepare_download") else "0",
                    "milestone": str(kwargs.get("milestone", "")),
                    "sync_drive": "1" if kwargs.get("sync_drive") else "0",
                }

                class _R:
                    query_params = q_map

                return asyncio.run(worker.mortm5_save_ckpt(_R()))

        old_w_dir = os.environ.get("MORTM5_WORKER_DIR")
        old_d_dir = os.environ.get("MORTM5_DRIVE_CKPT_DIR")
        os.environ["MORTM5_WORKER_DIR"] = os.path.join(tmp, "remote_worker")
        os.environ["MORTM5_DRIVE_CKPT_DIR"] = os.path.join(tmp, "remote_drive")
        try:
            client = _InProcessClient()
            remote_t_args = TrainArgs(
                {
                    "optimizer": "muon",
                    "lr_param": 1e-3,
                    "big_batch_size": 4,
                    "batch_size": 2,
                    "accumulation_steps": 1,
                    "max_micro_steps": 512,
                    "max_steps": 3,
                    "num_epochs": 1,
                    "val_total_tokens": 99999999,
                    "shuffle": False,
                    "checkpoint_percents": [50, 100],
                    "scheduler": {
                        "type": "wsd",
                        "warmup_steps": 1,
                        "stable_steps": 1,
                        "min_lr_ratio": 0.05,
                        "total_steps": 3,
                        "auto_total_steps": False,
                    },
                }
            )
            hybrid = RemoteMORTM5HybridTrainer(
                client=client,  # type: ignore[arg-type]
                m_args=_make_tiny_remi_args(),
                t_args=remote_t_args,
                version="remote_iso",
                save_dir=os.path.join(tmp, "mac_save"),
                ablation_id="remi_160m",
                colab_split_dir=non_existent_remi_dir,
                stage_to_local=False,
            )
            out = hybrid.run(project_name=None)
            assert out["status"] == "completed"
            assert out["final_step"] == 3
            assert out["final_val_loss"] > 0.0
            # Outer batch 0 received 4 windows (2 steps); outer batch 1 trimmed from 4 to 2 windows (1 remaining step)!
            assert client.window_counts == [4, 2]
            drive_sub = os.path.join(tmp, "remote_drive", "remi_160m")
            saved_files = os.listdir(drive_sub)
            assert any("pct100" in f for f in saved_files)
        finally:
            if old_w_dir is None:
                os.environ.pop("MORTM5_WORKER_DIR", None)
            else:
                os.environ["MORTM5_WORKER_DIR"] = old_w_dir
            if old_d_dir is None:
                os.environ.pop("MORTM5_DRIVE_CKPT_DIR", None)
            else:
                os.environ["MORTM5_DRIVE_CKPT_DIR"] = old_d_dir


def test_remi_empty_slot_min_bars_and_allow_eos_false_halting():
    torch.manual_seed(101)
    model = REMITransformer(_make_tiny_remi_args()).cpu().eval()
    empty_slots = [SLOT_CODE_EMPTY, SLOT_CODE_EMPTY, SLOT_CODE_EMPTY, SLOT_CODE_EMPTY]

    # 1. Empty slots + min_bars=4, max_bars=4, allow_eos=True -> exact 4 <BAR> tokens + <EOS>, 0 notes
    gen_eos = model.generate(
        max_notes=8,
        max_bars=4,
        min_notes=4,
        min_bars=4,
        slot_codes=empty_slots,
        batch_size=2,
        allow_eos=True,
        append_eos=True,
    )
    for b_i in range(2):
        bars_seen, n_notes = _assert_valid_remi_grammar(
            gen_eos[b_i], model, max_bars=4, active_slots=[False] * 4
        )
        assert bars_seen == 4
        assert n_notes == 0
        assert int(gen_eos[b_i, -1].item()) == EOS_TOKEN_ID

    # 2. Empty slots + min_bars=4, max_bars=4, allow_eos=False, append_eos=False -> exact 4 <BAR>s, 0 hallucinated notes, 0 <EOS>
    gen_no_eos = model.generate(
        max_notes=8,
        max_bars=4,
        min_notes=4,
        min_bars=4,
        slot_codes=empty_slots,
        batch_size=2,
        allow_eos=False,
        append_eos=False,
    )
    assert gen_no_eos.shape == (2, 1 + NUM_SLOT_TOKENS + 4)
    for b_i in range(2):
        toks = gen_no_eos[b_i].tolist()
        assert EOS_TOKEN_ID not in toks
        assert PAD_TOKEN_ID not in toks
        bars_seen, n_notes = _assert_valid_remi_grammar(
            gen_no_eos[b_i], model, max_bars=4, active_slots=[False] * 4
        )
        assert bars_seen == 4
        assert n_notes == 0

    # 3. Non-empty final-bar exhaustion with allow_eos=False, append_eos=False (prompt already at bar=0, pos=191, single active slot)
    tok = model.get_symbolic_tokenizer()
    active_one = [SLOT_CODE_EMPTY, 0, SLOT_CODE_EMPTY, SLOT_CODE_EMPTY]
    p_end = tok.encode_notes_window_prompt(
        np.asarray([[1, 191, 192, 60, 100]], dtype=np.int32),
        chunk_start=0,
        chunk_end=8,
        slot_codes=active_one,
        append_eos=False,
    )
    exhausted = model.generate(
        max_notes=4,
        max_bars=1,
        min_notes=0,
        min_bars=1,
        prompt=p_end,
        slot_codes=active_one,
        allow_eos=False,
        append_eos=False,
    )
    ex_toks = [int(t) for t in exhausted[0].tolist() if int(t) != PAD_TOKEN_ID]
    assert EOS_TOKEN_ID not in ex_toks
    bars_seen, n_notes = _assert_valid_remi_grammar(
        exhausted[0], model, max_bars=1, active_slots=[False, True, False, False]
    )
    assert bars_seen == 1 and n_notes >= 1


def _write_synthetic_octuple_split(oct_dir: str, split_name: str, n_segs: int, distinct_notes: bool = False) -> None:
    os.makedirs(oct_dir, exist_ok=True)
    oct_tok = OctupleMMTTokenizer()
    seg_rows = []
    bin_chunks = []
    cursor = 0
    for s_i in range(n_segs):
        k_notes = (1 + s_i) if distinct_notes else 1
        canon = np.zeros((k_notes, 6), dtype=np.int32)
        for n_i in range(k_notes):
            canon[n_i] = [0, (n_i * 6 + s_i) % 192, 1, 48 + ((s_i + n_i) % 24), 24, 96]
        # Ensure canonical sorting by (bar, pos, slot, pitch)
        order = np.lexsort((canon[:, 3], canon[:, 2], canon[:, 1], canon[:, 0]))
        canon = canon[order]
        b_u16 = oct_tok.encode_canonical_segment_notes(canon, num_chunks=16)
        bin_chunks.append(b_u16)
        seg_rows.append([cursor, int(b_u16.size), k_notes, k_notes, s_i, 16])
        cursor += int(b_u16.size)
    np.concatenate(bin_chunks).astype(np.uint16).tofile(os.path.join(oct_dir, f"{split_name}_tokens.bin"))
    np.save(os.path.join(oct_dir, f"{split_name}_segments.npy"), np.asarray(seg_rows, dtype=np.int64))
    slots_arr = np.tile(
        np.asarray([[SLOT_CODE_DRUMS, 0, SLOT_CODE_EMPTY, SLOT_CODE_EMPTY]], dtype=np.int16),
        (n_segs, 1),
    )
    np.save(os.path.join(oct_dir, f"{split_name}_slots.npy"), slots_arr)
    with open(os.path.join(oct_dir, f"{split_name}_state.json"), "w", encoding="utf-8") as sf:
        json.dump({"tokenizer_key": "octuple_mmt"}, sf)


def test_remote_milestone_warm_restart_branching_lr_sync_and_trimmed_resume():
    import colab_gpu_worker as worker
    from run_remote_mortm5 import RemoteMORTM5HybridTrainer

    class _InProcessClient:
        def __init__(self):
            self.window_slices: list[list[int]] = []
            self.step_lrs: list[float] = []
            self.last_init_info: dict = {}

        def health(self):
            return asyncio.run(worker.health())

        def start_stage(self, colab_split_dir: str):
            class _R:
                async def json(self):
                    return {"colab_split_dir": colab_split_dir}

            return asyncio.run(worker.mortm5_start_stage(_R()))

        def poll_stage(self):
            return asyncio.run(worker.mortm5_poll_stage())

        def init_mortm5(self, payload):
            class _R:
                headers = {"content-type": "application/json"}

                async def json(self):
                    return payload

            res = asyncio.run(worker.mortm5_init(_R()))
            self.last_init_info = res
            return res

        def start_outer_batch(self, outer_idx: int, epoch: int, window_indices=None):
            if window_indices is not None:
                self.window_slices.append(list(window_indices))

            class _R:
                headers = {"content-type": "application/json"}
                query_params = {}

                async def json(self):
                    return {"outer_idx": outer_idx, "epoch": epoch, "window_indices": window_indices}

            return asyncio.run(worker.mortm5_start_outer_batch(_R()))

        def poll_outer_batch(self, cursor: int):
            class _R:
                query_params = {"cursor": str(cursor)}

            res = asyncio.run(worker.mortm5_poll_outer_batch(_R()))
            for m in res.get("results", []):
                self.step_lrs.append(float(m["lr_muon"]))
            return res

        def validate(self, max_batches: int = 0):
            class _R:
                query_params = {"max_batches": str(max_batches)}

            return asyncio.run(worker.mortm5_validate(_R()))

        def save_checkpoint(self, **kwargs):
            q_map = {
                "outer": str(kwargs["outer_idx"]),
                "epoch": str(kwargs["epoch"]),
                "total_outer": str(kwargs["total_outer"]),
                "val_loss": f"{kwargs['val_loss']:.6f}",
                "version": str(kwargs["version"]),
                "ablation_id": str(kwargs.get("ablation_id", "")),
                "wandb_id": str(kwargs.get("wandb_id", "")),
                "prepare_download": "1" if kwargs.get("prepare_download") else "0",
                "milestone": str(kwargs.get("milestone", "")),
                "sync_drive": "1" if kwargs.get("sync_drive") else "0",
            }

            class _R:
                query_params = q_map

            return asyncio.run(worker.mortm5_save_ckpt(_R()))

    with tempfile.TemporaryDirectory() as tmp:
        oct_dir = os.path.join(tmp, "tokens", "octuple_mmt")
        _write_synthetic_octuple_split(oct_dir, "train", 12)
        _write_synthetic_octuple_split(oct_dir, "val", 4)
        remi_dir = os.path.join(tmp, "tokens", "multitrack_remi")  # non-existent sibling

        old_w = os.environ.get("MORTM5_WORKER_DIR")
        old_d = os.environ.get("MORTM5_DRIVE_CKPT_DIR")
        os.environ["MORTM5_WORKER_DIR"] = os.path.join(tmp, "worker")
        os.environ["MORTM5_DRIVE_CKPT_DIR"] = os.path.join(tmp, "drive")
        try:
            # 1. Full 1-epoch run (12 windows, big_bs=2 -> 6 outer batches, 6 steps; pct50 saved at step 3)
            c1 = _InProcessClient()
            t_full = TrainArgs(
                {
                    "optimizer": "muon",
                    "lr_param": 1e-3,
                    "big_batch_size": 2,
                    "batch_size": 2,
                    "accumulation_steps": 1,
                    "max_micro_steps": 512,
                    "max_steps": 0,
                    "num_epochs": 1,
                    "val_total_tokens": 99999999,
                    "shuffle": False,
                    "checkpoint_percents": [50, 100],
                    "scheduler": {
                        "type": "wsd",
                        "warmup_steps": 1,
                        "stable_steps": 4,
                        "min_lr_ratio": 0.05,
                        "total_steps": 6,
                        "auto_total_steps": False,
                    },
                }
            )
            h_full = RemoteMORTM5HybridTrainer(
                client=c1,  # type: ignore[arg-type]
                m_args=_make_tiny_remi_args(),
                t_args=t_full,
                version="full1ep",
                save_dir=os.path.join(tmp, "mac_save"),
                ablation_id="remi_160m",
                colab_split_dir=remi_dir,
                stage_to_local=False,
            )
            res_full = h_full.run(project_name=None)
            assert res_full["final_step"] == 6

            drive_160m = os.path.join(tmp, "drive", "remi_160m")
            ms_pth = os.path.join(drive_160m, "MORTM5.full1ep_pct50_step3.pth")
            ms_ckpt = os.path.join(drive_160m, "MORTM5.full1ep_pct50_step3.checkpoint.pt")
            assert os.path.isfile(ms_pth), "Weights-only milestone .pth missing"
            assert os.path.isfile(ms_ckpt), "Resumable milestone .checkpoint.pt missing after pct100 completion!"
            ms_meta = torch.load(ms_ckpt, map_location="cpu", weights_only=False)
            assert ms_meta["optimizer_step"] == 3
            assert ms_meta["optimizer_state_dict"] is not None

            # 2. Trap 1D: weights-only .pth WITHOUT a sibling .checkpoint.pt MUST fail fast unless allow_weights_only_resume=True
            lone_pth = os.path.join(tmp, "lone_weights.pth")
            torch.save(ms_meta["model_state_dict"], lone_pth)
            c_bad = _InProcessClient()
            err_info = c_bad.init_mortm5(
                {
                    "model_config": _make_tiny_remi_args().to_dict(),
                    "train_config": t_full.to_dict(),
                    "ablation_id": "remi_160m",
                    "colab_split_dir": remi_dir,
                    "stage_to_local": False,
                    "resume_checkpoint_path": lone_pth,
                    "version": "bad_resume",
                }
            )
            assert "error" in err_info and "optimizer_state_dict" in err_info["error"]

            # 3. Trap 1A + auto-upgrade (.pth -> .checkpoint.pt): Warm-restart branch from step 3 into Iso-FLOPs WSD (total_steps=4)
            # In the 6-step schedule, step 3 was at peak LR (1.0e-3).
            # In the branched schedule (warmup=1, stable=0, total=4, min_lr_ratio=0.05), step 3 is already inside cooldown:
            # progress at step=3 is (3-1)/(4-1) = 2/3 -> cosine factor = 0.05 + 0.95 * 0.5 * (1 + cos(2*pi/3)) = 0.2875 (< 1.0)!
            # Note: worker reports lr AFTER scheduler.step() (i.e. at step=4 -> factor 0.05), AND we check optimizer param_groups BEFORE step 4!
            c_branch = _InProcessClient()
            t_branch = TrainArgs(
                {
                    "optimizer": "muon",
                    "lr_param": 1e-3,
                    "big_batch_size": 2,
                    "batch_size": 2,
                    "accumulation_steps": 1,
                    "max_micro_steps": 512,
                    "max_steps": 4,
                    "num_epochs": 1,
                    "val_total_tokens": 99999999,
                    "shuffle": False,
                    "checkpoint_percents": [100],
                    "scheduler": {
                        "type": "wsd",
                        "warmup_steps": 1,
                        "stable_steps": 0,
                        "min_lr_ratio": 0.05,
                        "total_steps": 4,
                        "auto_total_steps": False,
                    },
                }
            )
            # Verify pre-step LR sync directly via mortm5_init first
            branch_init = c_branch.init_mortm5(
                {
                    "model_config": _make_tiny_remi_args().to_dict(),
                    "train_config": t_branch.to_dict(),
                    "ablation_id": "remi_160m",
                    "colab_split_dir": remi_dir,
                    "stage_to_local": False,
                    "resume_checkpoint_path": ms_pth,  # intentionally pass .pth to test auto-upgrade!
                    "version": "iso_branch",
                }
            )
            assert "error" not in branch_init
            assert branch_init["optimizer_restored"] is True
            assert branch_init["step"] == 3
            expected_lr_step3 = 1e-3 * float(
                get_cosine_schedule_with_warmup(warmup_steps=1, total_steps=4, min_lr_ratio=0.05, stable_steps=0)(3)
            )
            actual_pre_step_lr = float(worker.STATE["optimizer"].param_groups[0]["lr"])
            assert abs(actual_pre_step_lr - expected_lr_step3) < 1e-9
            assert actual_pre_step_lr < 0.5e-3  # Strictly cooled down, NOT stale peak 1e-3!

            h_branch = RemoteMORTM5HybridTrainer(
                client=c_branch,  # type: ignore[arg-type]
                m_args=_make_tiny_remi_args(),
                t_args=t_branch,
                version="iso_branch",
                save_dir=os.path.join(tmp, "mac_branch"),
                ablation_id="remi_160m",
                colab_split_dir=remi_dir,
                stage_to_local=False,
                resume_ckpt_path=ms_pth,
            )
            res_branch = h_branch.run(project_name=None)
            assert res_branch["final_step"] == 4
            assert c_branch.window_slices == [[6, 7]]
            # Source milestone checkpoint must remain untouched at step 3!
            ms_meta_after = torch.load(ms_ckpt, map_location="cpu", weights_only=False)
            assert ms_meta_after["optimizer_step"] == 3

            # 4. Trap 1C: Mid-outer-batch trimmed stop (big_bs=8, max_steps=3 -> outer 0 trimmed to 6 windows) + exact continuation
            c_part1 = _InProcessClient()
            t_part1 = TrainArgs(
                {
                    "optimizer": "muon",
                    "lr_param": 1e-3,
                    "big_batch_size": 8,
                    "batch_size": 2,
                    "accumulation_steps": 1,
                    "max_micro_steps": 512,
                    "max_steps": 3,
                    "num_epochs": 1,
                    "val_total_tokens": 99999999,
                    "shuffle": False,
                    "scheduler": {"type": "wsd", "warmup_steps": 1, "stable_steps": 2, "total_steps": 6, "auto_total_steps": False},
                }
            )
            h_part1 = RemoteMORTM5HybridTrainer(
                client=c_part1,  # type: ignore[arg-type]
                m_args=_make_tiny_remi_args(),
                t_args=t_part1,
                version="trim_cont",
                save_dir=os.path.join(tmp, "mac_trim"),
                ablation_id="remi_160m",
                colab_split_dir=remi_dir,
                stage_to_local=False,
            )
            r_p1 = h_part1.run(project_name=None)
            assert r_p1["final_step"] == 3
            assert c_part1.window_slices == [[0, 1, 2, 3, 4, 5]]

            c_part2 = _InProcessClient()
            t_part2 = TrainArgs({**t_part1.to_dict(), "max_steps": 0})
            trim_ckpt = os.path.join(drive_160m, "MORTM5.trim_cont.checkpoint.pt")
            h_part2 = RemoteMORTM5HybridTrainer(
                client=c_part2,  # type: ignore[arg-type]
                m_args=_make_tiny_remi_args(),
                t_args=t_part2,
                version="trim_cont",
                save_dir=os.path.join(tmp, "mac_trim"),
                ablation_id="remi_160m",
                colab_split_dir=remi_dir,
                stage_to_local=False,
                resume_ckpt_path=trim_ckpt,
            )
            r_p2 = h_part2.run(project_name=None)
            assert r_p2["final_step"] == 6
            # Must resume at window 6 (remaining [6, 7] of outer 0, then [8, 9, 10, 11] of outer 1) -> zero skipped windows!
            assert c_part2.window_slices == [[6, 7], [8, 9, 10, 11]]
        finally:
            if old_w is None:
                os.environ.pop("MORTM5_WORKER_DIR", None)
            else:
                os.environ["MORTM5_WORKER_DIR"] = old_w
            if old_d is None:
                os.environ.pop("MORTM5_DRIVE_CKPT_DIR", None)
            else:
                os.environ["MORTM5_DRIVE_CKPT_DIR"] = old_d


def test_sibling_fallback_empty_dir_split_divergence_and_mac_streaming_val():
    import colab_gpu_worker as worker
    from run_remote_mortm5 import RemoteMORTM5HybridTrainer, _pack_csr_sequences_zlib

    with tempfile.TemporaryDirectory() as tmp:
        oct_dir = os.path.join(tmp, "drive_mount", "octuple_mmt")
        empty_remi_dir = os.path.join(tmp, "drive_mount", "multitrack_remi")
        os.makedirs(empty_remi_dir, exist_ok=True)  # Empty directory exists on disk!
        _write_synthetic_octuple_split(oct_dir, "train", 6)
        _write_synthetic_octuple_split(oct_dir, "val", 4)

        remi_args = _make_tiny_remi_args()
        # 1. Empty directory falls back cleanly to sibling octuple_mmt
        ds_empty_fb = MORTM5MemmapDataset.from_split_dir(empty_remi_dir, "val", remi_args)
        assert isinstance(ds_empty_fb, MORTM5SymbolicMemmapDataset)
        assert ds_empty_fb.transcode_from_octuple is True
        assert len(ds_empty_fb) == 4

        # 2. Mac-streaming RemoteMORTM5HybridTrainer with empty `multitrack_remi` resolves BOTH train and val (`final_val_loss > 0`)
        class _StreamingClient:
            def health(self):
                return asyncio.run(worker.health())

            def init_mortm5(self, payload):
                class _R:
                    headers = {"content-type": "application/json"}

                    async def json(self):
                        return payload

                return asyncio.run(worker.mortm5_init(_R()))

            def upload_variable_sequences(self, seqs):
                worker.STATE["upload_parts"] = [_pack_csr_sequences_zlib(seqs)]

            def start_outer_batch(self, outer_idx: int, epoch: int, window_indices=None):
                class _R:
                    headers = {"content-type": "application/json"}
                    query_params = {}

                    async def json(self):
                        return {"outer_idx": outer_idx, "epoch": epoch, "window_indices": window_indices}

                return asyncio.run(worker.mortm5_start_outer_batch(_R()))

            def poll_outer_batch(self, cursor: int):
                class _R:
                    query_params = {"cursor": str(cursor)}

                return asyncio.run(worker.mortm5_poll_outer_batch(_R()))

            def validate(self, max_batches: int = 0):
                class _R:
                    query_params = {"max_batches": str(max_batches)}

                return asyncio.run(worker.mortm5_validate(_R()))

            def save_checkpoint(self, **kwargs):
                q_map = {
                    "outer": str(kwargs["outer_idx"]),
                    "epoch": str(kwargs["epoch"]),
                    "total_outer": str(kwargs["total_outer"]),
                    "val_loss": f"{kwargs['val_loss']:.6f}",
                    "version": str(kwargs["version"]),
                    "ablation_id": str(kwargs.get("ablation_id", "")),
                    "wandb_id": "",
                    "prepare_download": "0",
                    "milestone": str(kwargs.get("milestone", "")),
                    "sync_drive": "1",
                }

                class _R:
                    query_params = q_map

                return asyncio.run(worker.mortm5_save_ckpt(_R()))

        old_w = os.environ.get("MORTM5_WORKER_DIR")
        old_d = os.environ.get("MORTM5_DRIVE_CKPT_DIR")
        os.environ["MORTM5_WORKER_DIR"] = os.path.join(tmp, "w_stream")
        os.environ.pop("MORTM5_DRIVE_CKPT_DIR", None)
        try:
            t_stream = TrainArgs(
                {
                    "optimizer": "muon",
                    "lr_param": 1e-3,
                    "big_batch_size": 6,
                    "batch_size": 2,
                    "accumulation_steps": 1,
                    "max_micro_steps": 512,
                    "max_steps": 2,
                    "num_epochs": 1,
                    "val_total_tokens": 99999999,
                    "shuffle": False,
                    "scheduler": {"type": "wsd", "warmup_steps": 1, "stable_steps": 0, "total_steps": 2, "auto_total_steps": False},
                }
            )
            h_mac = RemoteMORTM5HybridTrainer(
                client=_StreamingClient(),  # type: ignore[arg-type]
                m_args=remi_args,
                t_args=t_stream,
                version="mac_stream_val",
                save_dir=os.path.join(tmp, "mac_out"),
                ablation_id="remi_160m",
                colab_split_dir=None,
                mac_split_dir=empty_remi_dir,
                stage_to_local=False,
            )
            assert h_mac.mac_train_ds is not None and len(h_mac.mac_train_ds) == 6
            assert h_mac.mac_val_ds is not None and len(h_mac.mac_val_ds) == 4
            mac_res = h_mac.run(project_name=None)
            assert mac_res["final_step"] == 2
            assert mac_res["final_val_loss"] > 0.0

            # 3. Split-level divergence: `val_*` converted into `multitrack_remi`, `train_*` remains only in `octuple_mmt`
            transcode_octuple_split_to_remi(
                octuple_dir=oct_dir,
                out_dir=os.path.join(tmp, "drive_mount"),
                split="val",
                tokenizer_key="multitrack_remi",
            )
            assert os.path.isfile(os.path.join(empty_remi_dir, "val_tokens.bin"))
            assert not os.path.isfile(os.path.join(empty_remi_dir, "train_tokens.bin"))

            class _InitReq:
                headers = {"content-type": "application/json"}

                async def json(self):
                    return {
                        "model_config": remi_args.to_dict(),
                        "train_config": t_stream.to_dict(),
                        "ablation_id": "remi_160m",
                        "colab_split_dir": empty_remi_dir,
                        "stage_to_local": False,
                        "version": "split_div",
                    }

            div_info = asyncio.run(worker.mortm5_init(_InitReq()))
            assert div_info["status"] == "initialized"
            assert div_info["train_windows"] == 6
            assert div_info["val_windows"] == 4
            assert worker.STATE["train_ds"].transcode_from_octuple is True
            assert worker.STATE["val_ds"].transcode_from_octuple is False
        finally:
            if old_w is None:
                os.environ.pop("MORTM5_WORKER_DIR", None)
            else:
                os.environ["MORTM5_WORKER_DIR"] = old_w
            if old_d is not None:
                os.environ["MORTM5_DRIVE_CKPT_DIR"] = old_d


def test_local_train_mortm5_mid_outer_max_steps_resume_and_warm_restart_branch():
    from mortm.train.train import train_mortm5

    with tempfile.TemporaryDirectory() as tmp:
        oct_dir = os.path.join(tmp, "tokens", "octuple_mmt")
        # 12 windows with strictly distinct note counts (1..12 notes) so any duplicate/skipped window alters total tokens!
        _write_synthetic_octuple_split(oct_dir, "train", 12, distinct_notes=True)
        _write_synthetic_octuple_split(oct_dir, "val", 4, distinct_notes=True)
        remi_dir = os.path.join(tmp, "tokens", "multitrack_remi")

        # Reference uninterrupted 1-epoch token sum (6 steps across 12 shuffled windows)
        ref_save = os.path.join(tmp, "ref_save")
        base_cfg = {
            "optimizer": "muon",
            "lr_param": 1e-3,
            "big_batch_size": 8,  # outer 0 has 8 windows (4 steps), outer 1 has 4 windows (2 steps)
            "batch_size": 2,
            "accumulation_steps": 1,
            "max_micro_steps": 512,
            "max_steps": 0,
            "num_epochs": 1,
            "is_save_training_progress": True,
            "val_total_tokens": 99999999,
            "shuffle": True,
            "checkpoint_percents": [50, 100],
            "scheduler": {
                "type": "wsd",
                "warmup_steps": 1,
                "stable_steps": 4,
                "min_lr_ratio": 0.05,
                "total_steps": 6,
                "auto_total_steps": False,
            },
        }
        train_mortm5(
            model_config=_make_tiny_remi_args().to_dict(),
            train_config=base_cfg,
            token_directory=remi_dir,
            save_directory=ref_save,
            version="ref1ep",
            log_scale=False,
            project_name=None,
            resume=False,
        )
        ref_ckpt_path = os.path.join(ref_save, "mortm5_remi_tiny", "MORTM5_REMI_Tiny.ref1ep.checkpoint.pt")
        ref_ckpt = torch.load(ref_ckpt_path, map_location="cpu", weights_only=False)
        assert ref_ckpt["optimizer_step"] == 6
        assert ref_ckpt["epoch"] == 1
        ref_all_tokens = int(ref_ckpt["all_tokens"])

        # Milestone p50 resumable checkpoint must exist alongside .pth
        p50_pth = os.path.join(ref_save, "mortm5_remi_tiny", "MORTM5_REMI_Tiny.ref1ep.ckpt_p50.pth")
        p50_ckpt = os.path.join(ref_save, "mortm5_remi_tiny", "MORTM5_REMI_Tiny.ref1ep.ckpt_p50.checkpoint.pt")
        assert os.path.isfile(p50_pth)
        assert os.path.isfile(p50_ckpt)

        # 1. Stop mid-outer-batch at max_steps=3 (inside outer 0, inner_step_idx=3, epoch=0 preserved!)
        split_save = os.path.join(tmp, "split_save")
        cfg_stop3 = {**base_cfg, "max_steps": 3}
        train_mortm5(
            model_config=_make_tiny_remi_args().to_dict(),
            train_config=cfg_stop3,
            token_directory=remi_dir,
            save_directory=split_save,
            version="split_run",
            log_scale=False,
            project_name=None,
            resume=False,
        )
        split_ckpt_path = os.path.join(split_save, "mortm5_remi_tiny", "MORTM5_REMI_Tiny.split_run.checkpoint.pt")
        c3 = torch.load(split_ckpt_path, map_location="cpu", weights_only=False)
        assert c3["optimizer_step"] == 3
        assert c3["epoch"] == 0, f"Expected mid-epoch stop to save epoch=0, got {c3['epoch']}"
        assert c3["outer_batch_idx"] == 0
        assert c3["inner_step_idx"] == 3

        # 2. Re-running with resume=True and max_steps=3 executes 0 extra steps (pre-epoch / pre-step guard)
        train_mortm5(
            model_config=_make_tiny_remi_args().to_dict(),
            train_config=cfg_stop3,
            token_directory=remi_dir,
            save_directory=split_save,
            version="split_run",
            log_scale=False,
            project_name=None,
            resume=True,
        )
        c3_again = torch.load(split_ckpt_path, map_location="cpu", weights_only=False)
        assert c3_again["optimizer_step"] == 3
        assert int(c3_again["all_tokens"]) == int(c3["all_tokens"])

        # 3. Lifting max_steps (max_steps=0) seamlessly finishes remaining 1 step of outer 0 + 2 steps of outer 1 = exact ref_all_tokens!
        train_mortm5(
            model_config=_make_tiny_remi_args().to_dict(),
            train_config=base_cfg,
            token_directory=remi_dir,
            save_directory=split_save,
            version="split_run",
            log_scale=False,
            project_name=None,
            resume=True,
        )
        c6 = torch.load(split_ckpt_path, map_location="cpu", weights_only=False)
        assert c6["optimizer_step"] == 6
        assert c6["epoch"] == 1
        assert int(c6["all_tokens"]) == ref_all_tokens

        # 4. Warm-restart branching from p50_pth (auto-upgraded to p50_ckpt) into new version without overwriting p50_ckpt
        branch_save = os.path.join(tmp, "branch_save")
        cfg_branch = {
            **base_cfg,
            "max_steps": 4,
            "scheduler": {
                "type": "wsd",
                "warmup_steps": 1,
                "stable_steps": 0,
                "min_lr_ratio": 0.05,
                "total_steps": 4,
                "auto_total_steps": False,
            },
        }
        train_mortm5(
            model_config=_make_tiny_remi_args().to_dict(),
            train_config=cfg_branch,
            token_directory=remi_dir,
            save_directory=branch_save,
            version="iso_local_branch",
            log_scale=False,
            project_name=None,
            resume=True,
            resume_checkpoint_path=p50_pth,
        )
        branch_ckpt_path = os.path.join(branch_save, "mortm5_remi_tiny", "MORTM5_REMI_Tiny.iso_local_branch.checkpoint.pt")
        assert os.path.isfile(branch_ckpt_path)
        b_meta = torch.load(branch_ckpt_path, map_location="cpu", weights_only=False)
        assert b_meta["optimizer_step"] == 4
        # Source milestone p50_ckpt must remain untouched at step 3!
        p50_meta_after = torch.load(p50_ckpt, map_location="cpu", weights_only=False)
        assert p50_meta_after["optimizer_step"] == 3


def test_round3_shuffled_mid_outer_resume_boundary_interrupt_stop_sync_and_failfast():
    import threading
    import time
    import pytest
    import colab_gpu_worker as worker
    from mortm.train.train import train_mortm5
    from run_remote_mortm5 import RemoteMORTM5HybridTrainer, _build_epoch_window_schedule

    # --- Part A: Verify _build_epoch_window_schedule matches legacy 2-stage block permutation ---
    sched_12 = _build_epoch_window_schedule(
        num_train_windows=12, seed=42, epoch=0, shuffle=True, perm_block_bs=8
    )
    legacy_ep = np.random.RandomState(42 + 0).permutation(12).tolist()
    b0 = [legacy_ep[p] for p in worker._make_outer_batch_rng(42, 0, 0).permutation(8).tolist()]
    b1 = [legacy_ep[8 + p] for p in worker._make_outer_batch_rng(42, 0, 1).permutation(4).tolist()]
    assert sched_12 == b0 + b1
    assert sorted(sched_12) == list(range(12))

    with tempfile.TemporaryDirectory() as tmp:
        oct_dir = os.path.join(tmp, "tokens", "octuple_mmt")
        _write_synthetic_octuple_split(oct_dir, "train", 12, distinct_notes=True)
        _write_synthetic_octuple_split(oct_dir, "val", 4, distinct_notes=True)
        remi_dir = os.path.join(tmp, "tokens", "multitrack_remi")

        class _RecordingClient:
            def __init__(self, raise_interrupt_on_outer=None):
                self.dispatched_slices = []
                self.raise_interrupt_on_outer = raise_interrupt_on_outer

            def health(self):
                return asyncio.run(worker.health())

            def start_stage(self, colab_split_dir: str):
                class _R:
                    async def json(self):
                        return {"colab_split_dir": colab_split_dir}

                return asyncio.run(worker.mortm5_start_stage(_R()))

            def poll_stage(self):
                return asyncio.run(worker.mortm5_poll_stage())

            def init_mortm5(self, payload):
                class _R:
                    headers = {"content-type": "application/json"}

                    async def json(self):
                        return payload

                return asyncio.run(worker.mortm5_init(_R()))

            def start_outer_batch(self, outer_idx: int, epoch: int, window_indices=None):
                if self.raise_interrupt_on_outer is not None and (epoch, outer_idx) == self.raise_interrupt_on_outer:
                    self.raise_interrupt_on_outer = None
                    raise KeyboardInterrupt("Simulated Ctrl+C right at outer batch boundary")
                if window_indices is not None:
                    self.dispatched_slices.append(list(window_indices))

                class _R:
                    headers = {"content-type": "application/json"}
                    query_params = {}

                    async def json(self):
                        return {"outer_idx": outer_idx, "epoch": epoch, "window_indices": window_indices}

                return asyncio.run(worker.mortm5_start_outer_batch(_R()))

            def poll_outer_batch(self, cursor: int):
                class _R:
                    query_params = {"cursor": str(cursor)}

                return asyncio.run(worker.mortm5_poll_outer_batch(_R()))

            def validate(self, max_batches: int = 0):
                class _R:
                    query_params = {"max_batches": str(max_batches)}

                return asyncio.run(worker.mortm5_validate(_R()))

            def save_checkpoint(self, **kwargs):
                q_map = {
                    "outer": str(kwargs["outer_idx"]),
                    "epoch": str(kwargs["epoch"]),
                    "total_outer": str(kwargs["total_outer"]),
                    "val_loss": f"{kwargs['val_loss']:.6f}",
                    "version": str(kwargs["version"]),
                    "ablation_id": str(kwargs.get("ablation_id", "")),
                    "wandb_id": str(kwargs.get("wandb_id", "")),
                    "prepare_download": "1" if kwargs.get("prepare_download") else "0",
                    "milestone": str(kwargs.get("milestone", "")),
                    "sync_drive": "1" if kwargs.get("sync_drive") else "0",
                }

                class _R:
                    query_params = q_map

                return asyncio.run(worker.mortm5_save_ckpt(_R()))

        old_w = os.environ.get("MORTM5_WORKER_DIR")
        old_d = os.environ.get("MORTM5_DRIVE_CKPT_DIR")
        os.environ["MORTM5_WORKER_DIR"] = os.path.join(tmp, "worker_r3")
        os.environ["MORTM5_DRIVE_CKPT_DIR"] = os.path.join(tmp, "drive_r3")
        try:
            remi_args = _make_tiny_remi_args()
            base_remote_cfg = {
                "optimizer": "muon",
                "lr_param": 1e-3,
                "big_batch_size": 8,
                "batch_size": 2,
                "accumulation_steps": 1,
                "max_micro_steps": 512,
                "max_steps": 0,
                "num_epochs": 1,
                "val_total_tokens": 99999999,
                "shuffle": True,
                "checkpoint_percents": [50, 100],
                "scheduler": {
                    "type": "wsd",
                    "warmup_steps": 1,
                    "stable_steps": 4,
                    "min_lr_ratio": 0.05,
                    "total_steps": 6,
                    "auto_total_steps": False,
                },
            }

            # 1. Uninterrupted 1-epoch shuffled reference (6 steps across 12 distinct windows)
            ref_client = _RecordingClient()
            ref_trainer = RemoteMORTM5HybridTrainer(
                client=ref_client,  # type: ignore[arg-type]
                m_args=remi_args,
                t_args=TrainArgs(base_remote_cfg),
                version="r3_ref",
                save_dir=os.path.join(tmp, "mac_r3"),
                ablation_id="remi_160m",
                colab_split_dir=remi_dir,
                stage_to_local=False,
            )
            ref_out = ref_trainer.run(project_name=None)
            assert ref_out["final_step"] == 6
            assert ref_client.dispatched_slices == [sched_12[:8], sched_12[8:12]]
            drive_sub = os.path.join(tmp, "drive_r3", "remi_160m")
            ref_ckpt = torch.load(
                os.path.join(drive_sub, "MORTM5.r3_ref.checkpoint.pt"),
                map_location="cpu",
                weights_only=False,
            )
            ref_total_tokens = int(ref_ckpt["all_tokens"])

            # 2. Defect R2-1: max_steps=3 under shuffle=True sees IDENTICAL first 6 windows as ref_trainer!
            iso_client = _RecordingClient()
            iso_trainer = RemoteMORTM5HybridTrainer(
                client=iso_client,  # type: ignore[arg-type]
                m_args=remi_args,
                t_args=TrainArgs({**base_remote_cfg, "max_steps": 3}),
                version="r3_split",
                save_dir=os.path.join(tmp, "mac_r3"),
                ablation_id="remi_160m",
                colab_split_dir=remi_dir,
                stage_to_local=False,
            )
            iso_out = iso_trainer.run(project_name=None)
            assert iso_out["final_step"] == 3
            assert iso_client.dispatched_slices == [sched_12[:6]]
            split_ckpt_path = os.path.join(drive_sub, "MORTM5.r3_split.checkpoint.pt")
            split_c3 = torch.load(split_ckpt_path, map_location="cpu", weights_only=False)
            assert split_c3["epoch_trained_windows"] == 6
            assert split_c3["epoch_perm_block_bs"] == 8

            # Resume from step 3 with same big_batch_size=8 -> dispatches exact remaining 6 windows (0 dup / 0 drop)!
            res_same_client = _RecordingClient()
            res_same_trainer = RemoteMORTM5HybridTrainer(
                client=res_same_client,  # type: ignore[arg-type]
                m_args=remi_args,
                t_args=TrainArgs(base_remote_cfg),
                version="r3_split_res8",
                save_dir=os.path.join(tmp, "mac_r3"),
                ablation_id="remi_160m",
                colab_split_dir=remi_dir,
                stage_to_local=False,
                resume_ckpt_path=split_ckpt_path,
            )
            res_same_out = res_same_trainer.run(project_name=None)
            assert res_same_out["final_step"] == 6
            assert res_same_client.dispatched_slices == [sched_12[6:8], sched_12[8:12]]
            res8_ckpt = torch.load(
                os.path.join(drive_sub, "MORTM5.r3_split_res8.checkpoint.pt"),
                map_location="cpu",
                weights_only=False,
            )
            assert int(res8_ckpt["all_tokens"]) == ref_total_tokens

            # Resume from step 3 while changing big_batch_size 8 -> 4 with resume_prev_big_batch_size=8!
            res_chg_client = _RecordingClient()
            res_chg_trainer = RemoteMORTM5HybridTrainer(
                client=res_chg_client,  # type: ignore[arg-type]
                m_args=remi_args,
                t_args=TrainArgs({**base_remote_cfg, "big_batch_size": 4}),
                version="r3_split_res4",
                save_dir=os.path.join(tmp, "mac_r3"),
                ablation_id="remi_160m",
                colab_split_dir=remi_dir,
                stage_to_local=False,
                resume_ckpt_path=split_ckpt_path,
                resume_prev_big_batch_size=8,
            )
            res_chg_out = res_chg_trainer.run(project_name=None)
            assert res_chg_out["final_step"] == 6
            assert res_chg_client.dispatched_slices == [sched_12[6:8], sched_12[8:12]]
            res4_ckpt = torch.load(
                os.path.join(drive_sub, "MORTM5.r3_split_res4.checkpoint.pt"),
                map_location="cpu",
                weights_only=False,
            )
            assert int(res4_ckpt["all_tokens"]) == ref_total_tokens

            # 3. Defect R2-2: KeyboardInterrupt right as outer_idx=1 begins (saved_outer=1, epoch_trained_windows=8 == lower_win)
            int_client = _RecordingClient(raise_interrupt_on_outer=(0, 1))
            int_trainer = RemoteMORTM5HybridTrainer(
                client=int_client,  # type: ignore[arg-type]
                m_args=remi_args,
                t_args=TrainArgs(base_remote_cfg),
                version="r3_intr",
                save_dir=os.path.join(tmp, "mac_r3"),
                ablation_id="remi_160m",
                colab_split_dir=remi_dir,
                stage_to_local=False,
            )
            int_out = int_trainer.run(project_name=None)
            assert int_out["final_step"] == 4
            intr_ckpt_path = os.path.join(drive_sub, "MORTM5.r3_intr.checkpoint.pt")
            intr_meta = torch.load(intr_ckpt_path, map_location="cpu", weights_only=False)
            assert intr_meta["outer_batch_idx"] == 1
            assert intr_meta["epoch_trained_windows"] == 8  # == lower_win (1 * 8)

            # Resuming must NOT skip outer_idx=1; it must train sched_12[8:12] and reach exact ref_total_tokens!
            intr_res_client = _RecordingClient()
            intr_res_trainer = RemoteMORTM5HybridTrainer(
                client=intr_res_client,  # type: ignore[arg-type]
                m_args=remi_args,
                t_args=TrainArgs(base_remote_cfg),
                version="r3_intr",
                save_dir=os.path.join(tmp, "mac_r3"),
                ablation_id="remi_160m",
                colab_split_dir=remi_dir,
                stage_to_local=False,
                resume_ckpt_path=intr_ckpt_path,
            )
            intr_res_out = intr_res_trainer.run(project_name=None)
            assert intr_res_out["final_step"] == 6
            assert intr_res_client.dispatched_slices == [sched_12[8:12]]
            intr_done = torch.load(intr_ckpt_path, map_location="cpu", weights_only=False)
            assert int(intr_done["all_tokens"]) == ref_total_tokens

            # Cross-epoch boundary interrupt right as (epoch=1, outer_idx=0) begins!
            ep2_cfg = {**base_remote_cfg, "num_epochs": 2}
            ep_int_client = _RecordingClient(raise_interrupt_on_outer=(1, 0))
            ep_int_trainer = RemoteMORTM5HybridTrainer(
                client=ep_int_client,  # type: ignore[arg-type]
                m_args=remi_args,
                t_args=TrainArgs(ep2_cfg),
                version="r3_ep_intr",
                save_dir=os.path.join(tmp, "mac_r3"),
                ablation_id="remi_160m",
                colab_split_dir=remi_dir,
                stage_to_local=False,
            )
            ep_int_out = ep_int_trainer.run(project_name=None)
            assert ep_int_out["final_step"] == 6
            ep_intr_ckpt = os.path.join(drive_sub, "MORTM5.r3_ep_intr.checkpoint.pt")
            ep_meta = torch.load(ep_intr_ckpt, map_location="cpu", weights_only=False)
            assert ep_meta["epoch"] == 1
            assert ep_meta["outer_batch_idx"] == 0
            assert ep_meta["epoch_trained_windows"] == 0

            # Defect R3-1: Even if stale outer_idx=1 is sent to save_checkpoint while worker STATE["epoch"]==0,
            # mortm5_save_ckpt clamps outer_batch_idx=0 & epoch_trained_windows=0 so resuming trains full Epoch 1!
            stale_ep_save = ep_int_client.save_checkpoint(
                outer_idx=1,
                epoch=1,
                total_outer=2,
                val_loss=1.0,
                version="r3_ep_stale_outer",
                ablation_id="remi_160m",
                milestone="stop",
                sync_drive=True,
            )
            assert stale_ep_save["outer"] == 0
            stale_ep_ckpt = os.path.join(drive_sub, "MORTM5.r3_ep_stale_outer.checkpoint.pt")
            stale_meta = torch.load(stale_ep_ckpt, map_location="cpu", weights_only=False)
            assert stale_meta["epoch"] == 1
            assert stale_meta["outer_batch_idx"] == 0
            assert stale_meta["epoch_trained_windows"] == 0
            sched_12_ep1 = _build_epoch_window_schedule(
                num_train_windows=12, seed=42, epoch=1, shuffle=True, perm_block_bs=8
            )
            ep_res_client = _RecordingClient()
            ep_res_trainer = RemoteMORTM5HybridTrainer(
                client=ep_res_client,  # type: ignore[arg-type]
                m_args=remi_args,
                t_args=TrainArgs({**ep2_cfg, "scheduler": {**base_remote_cfg["scheduler"], "total_steps": 12}}),
                version="r3_ep_stale_outer",
                save_dir=os.path.join(tmp, "mac_r3"),
                ablation_id="remi_160m",
                colab_split_dir=remi_dir,
                stage_to_local=False,
                resume_ckpt_path=stale_ep_ckpt,
            )
            ep_res_out = ep_res_trainer.run(project_name=None)
            assert ep_res_out["final_step"] == 12
            assert ep_res_client.dispatched_slices == [sched_12_ep1[:8], sched_12_ep1[8:12]]
            ep2_done = torch.load(stale_ep_ckpt, map_location="cpu", weights_only=False)
            assert int(ep2_done["all_tokens"]) == 2 * ref_total_tokens

            # 4. Defect R2-3: _stop_active_outer_batch stops in-flight GPU worker & rolls back uncommitted sub-batch!
            # Configure max_micro_steps tiny enough so each 2-window step splits into 2 sub-batches, and pause after sub-batch 1!
            init_r3 = ref_client.init_mortm5(
                {
                    "model_config": remi_args.to_dict(),
                    "train_config": {**base_remote_cfg, "max_micro_steps": 1, "shuffle": False},
                    "ablation_id": "remi_160m",
                    "colab_split_dir": remi_dir,
                    "stage_to_local": False,
                    "version": "r3_live_stop",
                    "clean_old_checkpoints": True,
                }
            )
            assert init_r3["status"] == "initialized"
            entered_sub1 = threading.Event()
            release_sub1 = threading.Event()
            orig_epoch_fc = worker.STATE["trainer"].epoch_fc
            call_counter = {"n": 0}

            def _hooked_epoch_fc(m, pack, prog):
                out = orig_epoch_fc(m, pack, prog)
                call_counter["n"] += 1
                if call_counter["n"] == 3:
                    # Step 1 (2 sub-batches) committed; now inside sub-batch 1 of Step 2 (uncommitted)!
                    entered_sub1.set()
                    release_sub1.wait(timeout=5.0)
                return out

            worker.STATE["trainer"].epoch_fc = _hooked_epoch_fc
            try:
                ref_client.start_outer_batch(outer_idx=0, epoch=0, window_indices=[0, 1, 2, 3])
                assert entered_sub1.wait(timeout=5.0), "Worker did not reach mid-step sub-batch hook"
                assert worker._ACTIVE_OUTER_THREAD is not None and worker._ACTIVE_OUTER_THREAD.is_alive()

                # Trigger mortm5_save_ckpt from a concurrent caller while releasing the hook right after stop signal is set
                def _delayed_release():
                    while not worker._OUTER_STOP_EVENT.is_set():
                        time.sleep(0.005)
                    release_sub1.set()

                rel_t = threading.Thread(target=_delayed_release, daemon=True)
                rel_t.start()
                live_saved = ref_client.save_checkpoint(
                    outer_idx=0,
                    epoch=0,
                    total_outer=2,
                    val_loss=1.23,
                    version="r3_live_stop",
                    ablation_id="remi_160m",
                    milestone="stop",
                    sync_drive=True,
                )
                rel_t.join(timeout=2.0)
                assert worker._ACTIVE_OUTER_THREAD is None
                # Step 1 committed (windows [0, 1]), while Step 2's partial sub-batch (window [2]) was rolled back!
                assert live_saved["step"] == 1
                expected_step1_tokens = int((worker.STATE["train_ds"][0][1:] != PAD_TOKEN_ID).sum().item()) + int(
                    (worker.STATE["train_ds"][1][1:] != PAD_TOKEN_ID).sum().item()
                )
                assert live_saved["tokens"] == expected_step1_tokens
                for p in worker.STATE["model"].parameters():
                    assert p.grad is None
            finally:
                worker.STATE["trainer"].epoch_fc = orig_epoch_fc

            # 5. Defect R2-4 (Remote): bare .pth basename in Drive whose .pth was deleted still upgrades to .checkpoint.pt!
            ms_pth = os.path.join(drive_sub, "MORTM5.r3_ref_pct50_step4.pth")
            ms_ckpt = os.path.join(drive_sub, "MORTM5.r3_ref_pct50_step4.checkpoint.pt")
            assert os.path.isfile(ms_ckpt)
            if os.path.isfile(ms_pth):
                os.remove(ms_pth)
            assert not os.path.isfile(ms_pth)

            del_pth_init = ref_client.init_mortm5(
                {
                    "model_config": remi_args.to_dict(),
                    "train_config": base_remote_cfg,
                    "ablation_id": "remi_160m",
                    "colab_split_dir": remi_dir,
                    "stage_to_local": False,
                    "version": "r3_from_deleted_pth",
                    "resume_checkpoint_path": "MORTM5.r3_ref_pct50_step4.pth",
                }
            )
            assert del_pth_init["status"] == "initialized"
            assert del_pth_init["optimizer_restored"] is True
            assert del_pth_init["step"] == 4
        finally:
            if old_w is None:
                os.environ.pop("MORTM5_WORKER_DIR", None)
            else:
                os.environ["MORTM5_WORKER_DIR"] = old_w
            if old_d is None:
                os.environ.pop("MORTM5_DRIVE_CKPT_DIR", None)
            else:
                os.environ["MORTM5_DRIVE_CKPT_DIR"] = old_d

        # 6. Defect R2-4 (Local train_mortm5): fail-fast on missing path & weights-only .pth (even with resume=False),
        #    plus auto-upgrade when .pth was deleted while .checkpoint.pt exists!
        local_cfg = {
            "optimizer": "muon",
            "lr_param": 1e-3,
            "big_batch_size": 8,
            "batch_size": 2,
            "accumulation_steps": 1,
            "max_micro_steps": 512,
            "max_steps": 4,
            "num_epochs": 1,
            "is_save_training_progress": True,
            "val_total_tokens": 99999999,
            "shuffle": True,
            "checkpoint_percents": [50, 100],
            "scheduler": {
                "type": "wsd",
                "warmup_steps": 1,
                "stable_steps": 1,
                "min_lr_ratio": 0.05,
                "total_steps": 4,
                "auto_total_steps": False,
            },
        }
        for res_flag in (True, False):
            with pytest.raises(FileNotFoundError):
                train_mortm5(
                    model_config=_make_tiny_remi_args().to_dict(),
                    train_config=local_cfg,
                    token_directory=remi_dir,
                    save_directory=os.path.join(tmp, f"missing_res_{res_flag}"),
                    version="missing_test",
                    log_scale=False,
                    project_name=None,
                    resume=res_flag,
                    resume_checkpoint_path=os.path.join(tmp, "does_not_exist.checkpoint.pt"),
                )

        # Standalone weights-only .pth with NO .checkpoint.pt sibling -> RuntimeError (even with resume=False)
        orphan_pth = os.path.join(tmp, "orphan_weights_only.pth")
        torch.save(REMITransformer(_make_tiny_remi_args()).state_dict(), orphan_pth)
        for res_flag in (True, False):
            with pytest.raises(RuntimeError, match="optimizer_state_dict"):
                train_mortm5(
                    model_config=_make_tiny_remi_args().to_dict(),
                    train_config=local_cfg,
                    token_directory=remi_dir,
                    save_directory=os.path.join(tmp, f"orphan_res_{res_flag}"),
                    version="orphan_test",
                    log_scale=False,
                    project_name=None,
                    resume=res_flag,
                    resume_checkpoint_path=orphan_pth,
                )

        # Produce a local milestone checkpoint, delete its .pth, and pass the deleted .pth with resume=False!
        seed_save = os.path.join(tmp, "local_seed_save")
        train_mortm5(
            model_config=_make_tiny_remi_args().to_dict(),
            train_config={**local_cfg, "max_steps": 2, "checkpoint_percents": [100]},
            token_directory=remi_dir,
            save_directory=seed_save,
            version="seed2",
            log_scale=False,
            project_name=None,
            resume=False,
        )
        seed_ckpt = os.path.join(seed_save, "mortm5_remi_tiny", "MORTM5_REMI_Tiny.seed2.checkpoint.pt")
        deleted_seed_pth = os.path.join(seed_save, "mortm5_remi_tiny", "MORTM5_REMI_Tiny.seed2.pth")
        assert os.path.isfile(seed_ckpt) and not os.path.isfile(deleted_seed_pth)

        upgraded_save = os.path.join(tmp, "local_upgraded_save")
        train_mortm5(
            model_config=_make_tiny_remi_args().to_dict(),
            train_config=local_cfg,
            token_directory=remi_dir,
            save_directory=upgraded_save,
            version="from_del_pth",
            log_scale=False,
            project_name=None,
            resume=False,  # Even with resume=False, explicit resume_checkpoint_path must be honored!
            resume_checkpoint_path=deleted_seed_pth,
        )
        up_ckpt_path = os.path.join(upgraded_save, "mortm5_remi_tiny", "MORTM5_REMI_Tiny.from_del_pth.checkpoint.pt")
        up_meta = torch.load(up_ckpt_path, map_location="cpu", weights_only=False)
        assert up_meta["optimizer_step"] == 4



