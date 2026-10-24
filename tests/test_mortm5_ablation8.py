import json
import os
import tempfile
import mido
import numpy as np
import pytest
import torch

from mortm.models.modules.config import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    INST_DRUMS_TOKEN_ID,
    INST_EMPTY_TOKEN_ID,
    INST_PROGRAM_OFFSET,
    INST_RVQ_TOKEN_OFFSET,
    MORTM5Args,
    NUM_SLOT_TOKENS,
    PAD_TOKEN_ID,
    RVQ_TOKEN_OFFSET,
    SLOT_CODE_DRUMS,
    SLOT_CODE_EMPTY,
    slot_code_to_token_id,
    token_id_to_slot_code,
)
from mortm.models.mortm5 import (
    ABLATION_MODEL_CONFIGS,
    AblationTransformer1,
    AblationTransformer2,
    build_mortm5_ablation_model,
)
from mortm.train.datasets import MORTM5MemmapDataset, mortm5_collate_fn
from mortm.utils.convert_pianoroll import (
    _choose_slots,
    chunks_to_midi,
    extract_notes,
    extract_slot_codes_from_midi,
    slot_codes_to_roll_meta,
    slot_codes_to_slots,
    slots_to_slot_codes,
)


def test_instrument_token_bijection_and_bounds():
    # Empty slot (-1) <-> 132
    assert slot_code_to_token_id(SLOT_CODE_EMPTY) == INST_EMPTY_TOKEN_ID
    assert token_id_to_slot_code(INST_EMPTY_TOKEN_ID) == SLOT_CODE_EMPTY

    # Drums slot (128) <-> 131
    assert slot_code_to_token_id(SLOT_CODE_DRUMS) == INST_DRUMS_TOKEN_ID
    assert token_id_to_slot_code(INST_DRUMS_TOKEN_ID) == SLOT_CODE_DRUMS

    # All 128 MIDI programs (0..127) <-> (3..130)
    for prog in range(128):
        tid = slot_code_to_token_id(prog)
        assert INST_PROGRAM_OFFSET <= tid < INST_DRUMS_TOKEN_ID
        assert token_id_to_slot_code(tid) == prog

    with pytest.raises(ValueError):
        slot_code_to_token_id(-2)
    with pytest.raises(ValueError):
        slot_code_to_token_id(129)
    with pytest.raises(ValueError):
        token_id_to_slot_code(BOS_TOKEN_ID)
    with pytest.raises(ValueError):
        token_id_to_slot_code(INST_RVQ_TOKEN_OFFSET)


def test_all_8_ablation_configs_and_recompute_vocab():
    expected_specs = {
        "m3_flat_shared": ("m3_g8l2", "flat", False, 8, 2, 2048, 16517),
        "m3_flat_separated": ("m3_g8l2", "flat", True, 8, 2, 2048, 32901),
        "m3_grouped_shared": ("m3_g8l2", "grouped", False, 8, 2, 2048, 16517),
        "m3_grouped_separated": ("m3_g8l2", "grouped", True, 8, 2, 2048, 32901),
        "m4_flat_shared": ("m4_g4l4", "flat", False, 4, 4, 2048, 8325),
        "m4_flat_separated": ("m4_g4l4", "flat", True, 4, 4, 2048, 32901),
        "m4_grouped_shared": ("m4_g4l4", "grouped", False, 4, 4, 2048, 8325),
        "m4_grouped_separated": ("m4_g4l4", "grouped", True, 4, 4, 2048, 32901),
    }
    assert set(ABLATION_MODEL_CONFIGS.keys()) == set(expected_specs.keys())

    for key, (tok_key, mode, sep, g, l, k, vocab) in expected_specs.items():
        cfg_path = ABLATION_MODEL_CONFIGS[key]
        assert os.path.exists(cfg_path), f"Missing config file: {cfg_path}"
        args = MORTM5Args(cfg_path)
        assert getattr(args, "tokenizer_key", None) == tok_key
        assert args.model_mode == mode
        assert args.separate_rvq_layers is sep
        assert args.use_instrument_tokens is True
        assert args.vq_groups == g
        assert args.vq_layers == l
        assert args.codebook_size == k
        assert args.token_offset == INST_RVQ_TOKEN_OFFSET
        assert args.vocab_size == vocab
        assert args.recompute_vocab() == vocab

        # Verify modern Dense Transformer backbone defaults
        assert args.normalize_type == "rmsnorm"
        assert args.use_rope is True
        assert args.use_silu is True
        assert args.use_bias is False
        assert args.use_moe_decoder is False
        assert args.d_model == 768
        assert args.d_layer == 12
        assert args.num_heads == 12
        assert args.dim_feedforward == 2048


def test_midi_slot_extraction_and_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        mid_path = os.path.join(tmpdir, "test_song.mid")
        roll = np.zeros((2, 8, 24, 128), dtype=np.float32)
        # Slot 0 (drums): kick at pitch 36
        roll[0, 0, 0, 36] = 0.9
        roll[0, 1, 0:4, 36] = 1.0
        # Slot 1 (program 24): note at pitch 60
        roll[0, 2, 0, 60] = 0.8
        roll[0, 3, 0:8, 60] = 1.0
        # Slot 2 (program 40): note at pitch 67
        roll[0, 4, 0, 67] = 0.85
        roll[0, 5, 0:8, 67] = 1.0

        target_codes = [SLOT_CODE_DRUMS, 24, 40, SLOT_CODE_EMPTY]
        meta = slot_codes_to_roll_meta(target_codes)
        chunks_to_midi(roll, meta=meta, output_path=mid_path)

        extracted_codes = extract_slot_codes_from_midi(mid_path)
        mid_obj = mido.MidiFile(mid_path)
        _, chosen_slots, _ = _choose_slots(extract_notes(mid_obj))
        ref_codes = slots_to_slot_codes(chosen_slots)

        assert np.array_equal(extracted_codes, ref_codes)
        assert list(extracted_codes) == target_codes


def _create_synthetic_tok_dataset(tmpdir: str, g: int, l: int, k: int = 64, n_songs: int = 3):
    tpc = g * l
    all_tokens = []
    offsets = []
    slots = [
        [SLOT_CODE_DRUMS, 0, 32, 48],
        [SLOT_CODE_EMPTY, 24, SLOT_CODE_EMPTY, SLOT_CODE_EMPTY],
        [SLOT_CODE_DRUMS, 56, 73, SLOT_CODE_EMPTY],
    ][:n_songs]

    rng = np.random.default_rng(42)
    cursor = 0
    for s_idx in range(n_songs):
        n_chunks = 6 + s_idx * 4  # 6, 10, 14 chunks
        # Raw tokens in .bin always have shape [n_chunks, l, g] with values RVQ_TOKEN_OFFSET + g * k + local_k
        local_k = rng.integers(0, k, size=(n_chunks, l, g), dtype=np.int64)
        group_off = np.arange(g, dtype=np.int64)[None, None, :] * k + RVQ_TOKEN_OFFSET
        song_toks = (local_k + group_off).reshape(-1).astype(np.uint16)
        all_tokens.append(song_toks)
        offsets.append((cursor, int(song_toks.size)))
        cursor += int(song_toks.size)

    bin_path = os.path.join(tmpdir, "synth_tokens.bin")
    off_path = os.path.join(tmpdir, "synth_offsets.npy")
    slots_path = os.path.join(tmpdir, "synth_slots.npy")
    state_path = os.path.join(tmpdir, "synth_state.json")

    np.concatenate(all_tokens, axis=0).tofile(bin_path)
    np.save(off_path, np.asarray(offsets, dtype=np.int64))
    np.save(slots_path, np.asarray(slots, dtype=np.int16))
    with open(state_path, "w", encoding="utf-8") as sf:
        json.dump(
            {
                "vq_groups": g,
                "vq_layers": l,
                "codebook_size": k,
                "token_offset": RVQ_TOKEN_OFFSET,
            },
            sf,
        )
    return bin_path, off_path, slots_path, slots


@pytest.mark.parametrize("tok_spec", [(8, 2), (4, 4)])
@pytest.mark.parametrize("mode", ["flat", "grouped"])
@pytest.mark.parametrize("separate_rvq_layers", [False, True])
@pytest.mark.parametrize("use_instrument_tokens", [False, True])
def test_8way_ablation_end_to_end_matrix(
    tok_spec, mode, separate_rvq_layers, use_instrument_tokens
):
    g, l = tok_spec
    k = 64
    device = torch.device("cpu")

    with tempfile.TemporaryDirectory() as tmpdir:
        bin_path, off_path, slots_path, slots_list = _create_synthetic_tok_dataset(
            tmpdir, g=g, l=l, k=k
        )
        cfg_dict = {
            "model_mode": mode,
            "separate_rvq_layers": separate_rvq_layers,
            "use_instrument_tokens": use_instrument_tokens,
            "vq_groups": g,
            "vq_layers": l,
            "encoder_layer": l,
            "encoder_wout": g * 32,
            "codebook_size": k,
            "d_model": 64,
            "d_layer": 2,
            "num_heads": 4,
            "dim_feedforward": 128,
            "dropout": 0.0,
            "position_length": 512,
        }
        cfg_file = os.path.join(tmpdir, "cfg.json")
        with open(cfg_file, "w", encoding="utf-8") as f:
            json.dump(cfg_dict, f)

        args = MORTM5Args(cfg_file)
        expected_offset = INST_RVQ_TOKEN_OFFSET if use_instrument_tokens else RVQ_TOKEN_OFFSET
        expected_eff_l = l if separate_rvq_layers else 1
        assert args.token_offset == expected_offset
        assert args.vocab_size == expected_offset + expected_eff_l * g * k

        ds = MORTM5MemmapDataset.from_args(
            args=args,
            tokens_bin_path=bin_path,
            offsets_npy_path=off_path,
            slots_npy_path=slots_path,
            max_chunks=6,
            min_chunks=4,
            stride_chunks=4,
        )
        assert len(ds) >= 3

        # Check multi-window song has instrument prefix on every window
        for w_idx in range(len(ds)):
            seq = ds[w_idx]
            song_idx = int(ds.windows[w_idx, 4])
            expected_slot_toks = [slot_code_to_token_id(c) for c in slots_list[song_idx]]
            unit = 1 if mode == "flat" else g
            assert torch.all(seq[:unit] == BOS_TOKEN_ID)
            assert torch.all(seq[-unit:] == EOS_TOKEN_ID)
            if use_instrument_tokens:
                inst_slice = seq[unit : unit * (1 + NUM_SLOT_TOKENS)]
                if mode == "flat":
                    assert inst_slice.tolist() == expected_slot_toks
                else:
                    inst_3d = inst_slice.reshape(NUM_SLOT_TOKENS, g)
                    for s_i in range(NUM_SLOT_TOKENS):
                        assert torch.all(inst_3d[s_i] == expected_slot_toks[s_i])
            body = seq[unit * (1 + (NUM_SLOT_TOKENS if use_instrument_tokens else 0)) : -unit]
            body_3d = body.reshape(-1, l, g)
            for layer_i in range(l):
                eff_l = layer_i if separate_rvq_layers else 0
                for grp_i in range(g):
                    low = expected_offset + (eff_l * g + grp_i) * k
                    high = low + k
                    vals = body_3d[:, layer_i, grp_i]
                    assert int(vals.min()) >= low and int(vals.max()) < high

        # Collate & Forward & Backward
        batch = [ds[0], ds[1]]
        in_ids, tgts, pad_mask = mortm5_collate_fn(batch, mode=mode, vq_groups=g)
        model = build_mortm5_ablation_model(args).to(device)
        model.eval()

        logits = model(in_ids, padding_mask=pad_mask, is_causal=True)
        if mode == "flat":
            assert isinstance(model, AblationTransformer1)
            assert logits.shape == (2, in_ids.size(1), args.vocab_size)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, args.vocab_size),
                tgts.reshape(-1),
                ignore_index=PAD_TOKEN_ID,
            )
        else:
            assert isinstance(model, AblationTransformer2)
            s_soft = in_ids.size(1) // g
            assert logits.shape == (2, s_soft, g, model.group_vocab_size)
            local_tgts = model.to_local_targets(tgts, ignore_index=PAD_TOKEN_ID)
            # Verify exact round-trip on valid non-pad tokens
            valid_seq_3d = ds[0].reshape(1, -1, g)
            rt_seq_3d = model.local_to_global_ids(model.to_local_targets(valid_seq_3d))
            assert torch.equal(valid_seq_3d, rt_seq_3d)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, model.group_vocab_size),
                local_tgts.reshape(-1),
                ignore_index=PAD_TOKEN_ID,
            )

        assert torch.isfinite(loss)
        loss.backward()
        assert model.embedding.weight.grad is not None

        # Verify strict causality: mutating the last step must not alter earlier steps' logits
        in_clone = in_ids.clone()
        if mode == "flat":
            in_clone[:, -5:] = expected_offset
            logits_mut = model(in_clone, padding_mask=pad_mask, is_causal=True)
            assert torch.allclose(logits[:, :-5, :], logits_mut[:, :-5, :], atol=1e-5)
        else:
            in_clone[:, -2 * g :] = expected_offset
            logits_mut = model(in_clone, padding_mask=pad_mask, is_causal=True)
            assert torch.allclose(logits[:, :-2, :, :], logits_mut[:, :-2, :, :], atol=1e-5)

        # Verify grammar-constrained KV-cache generate() + RVQ & slot extraction
        cond_slots = [SLOT_CODE_DRUMS, 12, 40, SLOT_CODE_EMPTY]
        gen = model.generate(
            num_chunks=2,
            slot_codes=cond_slots,
            batch_size=2,
            temperature=1.0,
            top_k=8,
            top_p=0.9,
            append_eos=True,
        )
        for b_i in range(2):
            rvq_extracted = model.extract_rvq_tokens(gen[b_i])
            assert rvq_extracted.numel() == 2 * l * g
            # Extracted RVQ tokens are normalized to [RVQ_TOKEN_OFFSET, RVQ_TOKEN_OFFSET + g * k)
            rvq_3d = rvq_extracted.reshape(2, l, g)
            for grp_i in range(g):
                low = RVQ_TOKEN_OFFSET + grp_i * k
                high = low + k
                assert int(rvq_3d[:, :, grp_i].min()) >= low
                assert int(rvq_3d[:, :, grp_i].max()) < high
            if use_instrument_tokens:
                ext_slots = model.extract_slot_codes(gen[b_i])
                assert ext_slots is not None and list(ext_slots) == cond_slots


def test_ablation2_soft_token_embedding_unit_variance():
    torch.manual_seed(123)
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_file = os.path.join(tmpdir, "cfg.json")
        with open(cfg_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_mode": "grouped",
                    "separate_rvq_layers": True,
                    "use_instrument_tokens": True,
                    "vq_groups": 8,
                    "vq_layers": 2,
                    "codebook_size": 256,
                    "d_model": 512,
                    "d_layer": 1,
                    "num_heads": 4,
                    "dim_feedforward": 512,
                },
                f,
            )
        args = MORTM5Args(cfg_file)
        model = AblationTransformer2(args).to("cpu")

        # 1. Special token repeated G=8 times -> must equal exact single embedding (variance ~ 1.0)
        bos_soft = torch.full((1, 1, 8), BOS_TOKEN_ID, dtype=torch.long)
        agg_bos = model._aggregate_group_embeddings(bos_soft)
        single_bos = model.embedding(torch.tensor([[BOS_TOKEN_ID]], dtype=torch.long))
        assert torch.allclose(agg_bos, single_bos, atol=1e-6)
        assert 0.7 < float(agg_bos.var().item()) < 1.3

        # 2. RVQ tokens across G=8 independent groups -> scaled by 1/sqrt(G), variance ~ 1.0
        rng = torch.Generator().manual_seed(999)
        local_k = torch.randint(0, 256, (256, 8), generator=rng)
        group_off = torch.arange(8) * 256 + args.token_offset
        rvq_soft = local_k + group_off
        agg_rvq = model._aggregate_group_embeddings(rvq_soft)
        var_rvq = float(agg_rvq.var().item())
        assert 0.85 < var_rvq < 1.15, f"Expected unit variance (~1.0), got {var_rvq}"


class TestMORTM5MuonAndTrainer:
    """Verify native torch.optim.Muon + AdamW, Cosine schedule, MORTM5TrainSet, and train_mortm5 E2E."""

    def test_muon_optimizer_partition_and_scheduler_sync(self, tmp_path):
        from mortm.train.muon import build_muon_optimizer, HybridMuonOptimizer
        from mortm.train.noam import get_cosine_schedule_with_warmup
        from torch.optim.lr_scheduler import LambdaLR

        args = MORTM5Args("configs/models/mortm5/ablation/m4_grouped_shared.json")
        args.d_model = 64
        args.d_layer = 2
        args.num_heads = 4
        args.dim_feedforward = 128
        args.codebook_size = 256
        args.recompute_vocab()
        model = build_mortm5_ablation_model(args).to("cpu")
        opt = build_muon_optimizer(model, lr=1e-3, weight_decay=0.1, verbose=False)
        assert isinstance(opt, HybridMuonOptimizer)
        assert isinstance(opt.muon, torch.optim.Muon)
        assert isinstance(opt.adamw, torch.optim.AdamW)

        # Ensure all Muon params are strictly 2D and do not include embedding or Wout
        muon_param_ids = {id(p) for g in opt.muon.param_groups for p in g["params"]}
        assert id(model.embedding.weight) not in muon_param_ids
        assert id(model.Wout.weight) not in muon_param_ids
        for g in opt.muon.param_groups:
            for p in g["params"]:
                assert p.ndim == 2

        sched = LambdaLR(
            opt,
            lr_lambda=get_cosine_schedule_with_warmup(total_steps=10, warmup_steps=2, min_lr_ratio=0.1),
        )
        x = torch.randint(args.token_offset, args.token_offset + 50, (2, 8, args.vq_groups))
        logits = model(x)
        tgt = model.to_local_targets(x).reshape(-1)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt, ignore_index=PAD_TOKEN_ID)
        loss.backward()
        opt.step()
        opt.zero_grad()
        sched.step()
        # Both Muon and AdamW param_groups must receive the exact same scheduled LR
        assert opt.muon.param_groups[0]["lr"] == opt.adamw.param_groups[0]["lr"]

    @pytest.mark.parametrize(
        "cfg_path",
        [
            "configs/models/mortm5/ablation/m4_grouped_shared.json",
            "configs/models/mortm5/ablation/m4_flat_shared.json",
        ],
    )
    def test_train_mortm5_e2e_and_checkpoint_resume(self, tmp_path, cfg_path):
        from mortm.train.train import train_mortm5

        tok_dir = tmp_path / "tokens"
        save_dir = tmp_path / "save"
        tok_dir.mkdir(parents=True, exist_ok=True)
        save_dir.mkdir(parents=True, exist_ok=True)

        for split, n_songs in [("train", 4), ("val", 2)]:
            toks_per_song = 8 * 16
            raw = np.random.randint(3, 200, size=(n_songs * toks_per_song,), dtype=np.uint16)
            raw.tofile(str(tok_dir / f"{split}_tokens.bin"))
            offsets = np.array([[i * toks_per_song, toks_per_song] for i in range(n_songs)], dtype=np.int64)
            np.save(str(tok_dir / f"{split}_offsets.npy"), offsets)
            slots = np.tile(np.array([128, 0, 32, -1], dtype=np.int16), (n_songs, 1))
            np.save(str(tok_dir / f"{split}_slots.npy"), slots)

        with open(cfg_path, "r") as f:
            m_cfg = json.load(f)
        m_cfg["d_model"] = 32
        m_cfg["d_layer"] = 1
        m_cfg["num_heads"] = 4
        m_cfg["dim_feedforward"] = 64
        m_cfg["position_length"] = 64 if m_cfg["model_mode"] == "grouped" else 256
        m_cfg_file = tmp_path / "model.json"
        m_cfg_file.write_text(json.dumps(m_cfg))

        with open("configs/train/mortm5/ablation_muon_cos.json", "r") as f:
            t_cfg = json.load(f)
        t_cfg["big_batch_size"] = 2
        t_cfg["batch_size"] = 2
        t_cfg["accumulation_steps"] = 1
        t_cfg["num_epochs"] = 1
        t_cfg["val_total_tokens"] = 50
        t_cfg_file = tmp_path / "train.json"
        t_cfg_file.write_text(json.dumps(t_cfg))

        model = train_mortm5(
            model_config=str(m_cfg_file),
            train_config=str(t_cfg_file),
            token_directory=str(tok_dir),
            save_directory=str(save_dir),
            version="pytest_v1",
            project_name=None,
            resume=True,
        )
        from mortm.models.modules.config import resolve_ablation_subdir
        ablation_subdir = resolve_ablation_subdir(m_cfg["name"])
        ckpt_file = save_dir / ablation_subdir / f"{m_cfg['name']}.pytest_v1.checkpoint.pt"
        assert ckpt_file.exists()
        ckpt = torch.load(str(ckpt_file), map_location="cpu", weights_only=False)
        assert ckpt["optimizer_step"] == 2
        assert "muon" in ckpt["optimizer_state_dict"]
        assert "adamw" in ckpt["optimizer_state_dict"]

        # Verify resume into epoch 2 works cleanly
        t_cfg["num_epochs"] = 2
        t_cfg_file.write_text(json.dumps(t_cfg))
        model2 = train_mortm5(
            model_config=str(m_cfg_file),
            train_config=str(t_cfg_file),
            token_directory=str(tok_dir),
            save_directory=str(save_dir),
            version="pytest_v1",
            project_name=None,
            resume=True,
        )
        ckpt2 = torch.load(str(ckpt_file), map_location="cpu", weights_only=False)
        assert ckpt2["optimizer_step"] == 4

    def test_package_presets_and_dict_init_without_json_files(self):
        import mortm
        assert mortm.__version__.startswith("5.0.0")

        for idx, key in enumerate(mortm.MORTM5_ABLATION_PRESETS.keys(), start=1):
            a_by_key = mortm.MORTM5Args(key)
            a_by_num = mortm.MORTM5Args(f"#{idx}")
            a_by_dict = mortm.MORTM5Args(a_by_key.to_dict())
            assert a_by_key.name == a_by_num.name == a_by_dict.name
            assert a_by_key.vocab_size == a_by_num.vocab_size == a_by_dict.vocab_size

        t_args = mortm.TrainArgs("muon_cos")
        assert t_args.optimizer_type == "muon"
        t_roundtrip = mortm.TrainArgs(t_args.to_dict())
        assert t_roundtrip.lr_param == t_args.lr_param

    def test_decode_codebook_head_routing_and_gradient_isolation(self):
        from mortm.models.modules.layers import DecodeCodebookHead

        # 1. Shared (#7) uses nn.Linear; Separated (#8) uses DecodeCodebookHead
        args_shared = MORTM5Args("#7")
        args_shared.d_model = 32
        args_shared.d_layer = 1
        args_shared.num_heads = 4
        args_shared.dim_feedforward = 64
        args_shared.codebook_size = 64
        args_shared.recompute_vocab()
        model_shared = AblationTransformer2(args_shared).to("cpu")
        assert isinstance(model_shared.Wout, torch.nn.Linear)
        assert not isinstance(model_shared.Wout, DecodeCodebookHead)

        args_sep = MORTM5Args("#8")
        args_sep.d_model = 32
        args_sep.d_layer = 1
        args_sep.num_heads = 4
        args_sep.dim_feedforward = 64
        args_sep.codebook_size = 64
        args_sep.recompute_vocab()
        model_sep = AblationTransformer2(args_sep).to("cpu")
        assert isinstance(model_sep.Wout, DecodeCodebookHead)
        assert model_sep.group_vocab_size == args_sep.token_offset + args_sep.codebook_size

        # 2. Gradient isolation: loss at soft_pos predicting L2 (input pos = 4 + 2 = 6)
        # must ONLY backprop into decode[2][0..3] and special_decode, leaving decode[0,1,3] untouched
        prefix = model_sep.build_prefix_tokens(batch_size=2, device=torch.device("cpu"))  # [2, 5, 4]
        rvq_steps = torch.full((2, 4, 4), args_sep.token_offset + 5, dtype=torch.long)
        seq = torch.cat([prefix, rvq_steps], dim=1)  # [2, 9, 4], input positions 0..8
        model_sep.zero_grad()
        logits = model_sep(seq, is_causal=True)
        assert logits.shape == (2, 9, 4, args_sep.token_offset + args_sep.codebook_size)

        # Input pos = 6 (which is Chunk 0, L1) predicts Chunk 0, L2 (layer_idx = (6 - 4) % 4 = 2)
        target_l2 = torch.full((2, 4), args_sep.token_offset + 10, dtype=torch.long)
        loss_l2 = torch.nn.functional.cross_entropy(
            logits[:, 6, :, :].reshape(-1, model_sep.group_vocab_size),
            target_l2.reshape(-1),
        )
        loss_l2.backward()

        assert model_sep.Wout.special_decode.weight.grad is not None
        for l_idx in range(args_sep.vq_layers):
            for g_idx in range(args_sep.vq_groups):
                grad = model_sep.Wout.decode[l_idx][g_idx].weight.grad
                if l_idx == 2:
                    assert grad is not None and grad.abs().sum().item() > 0.0
                else:
                    assert grad is None or grad.abs().sum().item() == 0.0

        # 3. Step-by-step KV-cache decoding logits must match full-sequence forward logits across all L0..L3
        model_sep.eval()
        with torch.no_grad():
            full_logits = model_sep(seq, is_causal=True, is_save_cache=False)
            model_sep.reset_kv_cache()
            prefill_logits = model_sep(seq[:, :5, :], is_causal=True, is_save_cache=True)
            assert torch.allclose(full_logits[:, :5, :, :], prefill_logits, atol=1e-4)
            for pos in range(5, seq.size(1)):
                step_logits = model_sep(seq[:, pos : pos + 1, :], is_causal=True, is_save_cache=True)
                assert torch.allclose(full_logits[:, pos : pos + 1, :, :], step_logits, atol=1e-4)
            model_sep.reset_kv_cache()

    def test_colab_hybrid_worker_endpoints_end_to_end(self, tmp_path):
        import asyncio
        import sys
        from pathlib import Path
        root_dir = str(Path(__file__).resolve().parents[1])
        if root_dir not in sys.path:
            sys.path.insert(0, root_dir)
        import colab_gpu_worker as worker
        from run_remote_mortm5 import _pack_token_tensor_zlib

        tok_dir = tmp_path / "tokens"
        tok_dir.mkdir(parents=True, exist_ok=True)
        for split, n_songs in (("train", 4), ("val", 2)):
            toks_per_song = 16 * 12
            raw = np.random.randint(3, 500, size=(n_songs * toks_per_song,), dtype=np.uint16)
            raw.tofile(str(tok_dir / f"{split}_tokens.bin"))
            offsets = np.array([[i * toks_per_song, toks_per_song] for i in range(n_songs)], dtype=np.int64)
            np.save(str(tok_dir / f"{split}_offsets.npy"), offsets)
            slots = np.tile(np.array([128, 0, 32, -1], dtype=np.int16), (n_songs, 1))
            np.save(str(tok_dir / f"{split}_slots.npy"), slots)

        import mortm
        m_dict = mortm.MORTM5Args("#7").to_dict()
        m_dict.update({"d_model": 32, "d_layer": 1, "num_heads": 4, "dim_feedforward": 64, "position_length": 64})
        t_dict = mortm.TrainArgs("muon_cos").to_dict()
        t_dict.update({"big_batch_size": 2, "batch_size": 2, "accumulation_steps": 1, "num_epochs": 1})

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

        init_res = asyncio.run(
            worker.mortm5_init(
                _DummyReq(
                    json_data={
                        "model_config": m_dict,
                        "train_config": t_dict,
                        "colab_split_dir": str(tok_dir),
                        "stage_to_local": False,
                    }
                )
            )
        )
        assert init_res["status"] == "initialized"
        assert init_res["model_class"] == "AblationTransformer2"
        assert init_res["train_windows"] == 4
        assert init_res["total_steps"] == 2

        # Mode A: Colab local dataset outer batch execution
        start_res = asyncio.run(
            worker.mortm5_start_outer_batch(_DummyReq(json_data={"outer_idx": 0, "epoch": 0, "window_indices": [0, 1]}))
        )
        assert start_res["status"] == "started"
        for _ in range(50):
            poll_res = asyncio.run(worker.mortm5_poll_outer_batch(_DummyReq(query_params={"cursor": "0"})))
            if poll_res["done"]:
                break
            import time
            time.sleep(0.05)
        assert poll_res["done"] is True
        assert poll_res["error"] is None
        assert poll_res["step"] == 1
        assert len(poll_res["results"]) == 1

        # Mode B: Uploaded token batch execution (Mac streaming mode)
        asyncio.run(worker.clear_upload())
        sample_tensor = torch.stack([worker.STATE["train_ds"][2], worker.STATE["train_ds"][3]], dim=0)
        packed = _pack_token_tensor_zlib(sample_tensor)
        asyncio.run(worker.append_upload(_DummyReq(body_bytes=packed, headers={"content-type": "application/octet-stream"})))
        worker.STATE["train_ds"] = None  # Force uploaded tensor path
        asyncio.run(worker.mortm5_start_outer_batch(_DummyReq(json_data={"outer_idx": 1, "epoch": 0})))
        for _ in range(50):
            poll_res2 = asyncio.run(worker.mortm5_poll_outer_batch(_DummyReq(query_params={"cursor": "0"})))
            if poll_res2["done"]:
                break
            import time
            time.sleep(0.05)
        assert poll_res2["done"] is True
        assert poll_res2["error"] is None
        assert poll_res2["step"] == 2

        val_res = asyncio.run(worker.mortm5_validate(_DummyReq(query_params={"max_batches": "1"})))
        assert val_res["val_loss"] > 0.0
        assert val_res["val_tokens"] > 0

        # Checkpoint save + download preparation + resume
        os.environ["MORTM5_WORKER_DIR"] = str(tmp_path / "worker_ckpt")
        save_res = asyncio.run(
            worker.mortm5_save_ckpt(
                _DummyReq(
                    query_params={
                        "outer": "1",
                        "epoch": "0",
                        "total_outer": "2",
                        "val_loss": str(val_res["val_loss"]),
                        "version": "pytest_hybrid",
                        "prepare_download": "1",
                        "milestone": "pct100",
                    }
                )
            )
        )
        assert save_res["status"] == "saved"
        assert os.path.isfile(save_res["colab_path"])
        assert save_res["download_bytes"] > 0

        # Full RemoteMORTM5HybridTrainer.run() in both Colab mode and Mac streaming mode via in-process client adapter
        from run_remote_mortm5 import RemoteColabMORTM5Client, RemoteMORTM5HybridTrainer
        import urllib.parse

        class _InProcessClient(RemoteColabMORTM5Client):
            def __init__(self):
                super().__init__("http://inprocess.local")

            def health(self):
                return asyncio.run(worker.health())

            def init_mortm5(self, init_payload):
                return asyncio.run(worker.mortm5_init(_DummyReq(json_data=init_payload)))

            def upload_token_batch(self, batch_tensor, chunk_bytes=2 * 1024 * 1024):
                packed_bytes = _pack_token_tensor_zlib(batch_tensor)
                asyncio.run(worker.clear_upload())
                asyncio.run(
                    worker.append_upload(
                        _DummyReq(body_bytes=packed_bytes, headers={"content-type": "application/octet-stream"})
                    )
                )
                return len(packed_bytes)

            def start_outer_batch(self, outer_idx, epoch, window_indices=None):
                payload = {"outer_idx": int(outer_idx), "epoch": int(epoch)}
                if window_indices is not None:
                    payload["window_indices"] = window_indices
                return asyncio.run(worker.mortm5_start_outer_batch(_DummyReq(json_data=payload)))

            def poll_outer_batch(self, cursor):
                return asyncio.run(worker.mortm5_poll_outer_batch(_DummyReq(query_params={"cursor": str(cursor)})))

            def validate(self, max_batches=0):
                return asyncio.run(worker.mortm5_validate(_DummyReq(query_params={"max_batches": str(max_batches)})))

            def save_checkpoint(
                self, outer_idx, epoch, total_outer, val_loss, version, ablation_id="", wandb_id="", prepare_download=False, milestone="", sync_drive=False
            ):
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
                                "prepare_download": "1" if prepare_download else "0",
                                "milestone": milestone,
                                "sync_drive": "1" if sync_drive else "0",
                            }
                        )
                    )
                )

            def download_prepared_checkpoint(self, total_bytes, chunk_bytes=2 * 1024 * 1024):
                import io, zlib
                resp = asyncio.run(
                    worker.get_state_raw_chunk(_DummyReq(query_params={"start": "0", "end": str(total_bytes)}))
                )
                raw_bytes = getattr(resp, "body", getattr(resp, "content", b""))
                return torch.load(io.BytesIO(zlib.decompress(raw_bytes)), map_location="cpu", weights_only=False)

        mac_save_dir = tmp_path / "mac_out"
        trainer_colab_mode = RemoteMORTM5HybridTrainer(
            client=_InProcessClient(),
            m_args=mortm.MORTM5Args(m_dict),
            t_args=mortm.TrainArgs(t_dict),
            version="e2e_colab_mode",
            save_dir=str(mac_save_dir),
            colab_split_dir=str(tok_dir),
            stage_to_local=False,
            download_ckpt_to_mac=True,
            val_max_batches=1,
        )
        res_a = trainer_colab_mode.run(project_name=None)
        assert res_a["status"] == "completed"
        assert res_a["final_step"] == 2
        assert res_a["ablation_subdir"] == "7"
        assert (mac_save_dir / "7" / "MORTM5.e2e_colab_mode_step2.pth").is_file()

        trainer_mac_mode = RemoteMORTM5HybridTrainer(
            client=_InProcessClient(),
            m_args=mortm.MORTM5Args(m_dict),
            t_args=mortm.TrainArgs(t_dict),
            version="e2e_mac_stream",
            save_dir=str(mac_save_dir),
            colab_split_dir=None,
            mac_split_dir=str(tok_dir),
            stage_to_local=False,
            download_ckpt_to_mac=True,
            val_max_batches=1,
        )
        res_b = trainer_mac_mode.run(project_name=None)
        assert res_b["status"] == "completed"
        assert res_b["final_step"] == 2
        assert (mac_save_dir / "7" / "MORTM5.e2e_mac_stream_step2.pth").is_file()

        # Resume Case 1: Resume from completed epoch (outer_idx=1, step=2 of 2) -> 0 extra steps, no UnboundLocalError
        ckpt_step2 = os.path.join(worker._get_worker_dir(), "7", "MORTM5.e2e_colab_mode.checkpoint.pt")
        assert os.path.isfile(ckpt_step2)
        trainer_resume_done = RemoteMORTM5HybridTrainer(
            client=_InProcessClient(),
            m_args=mortm.MORTM5Args(m_dict),
            t_args=mortm.TrainArgs(t_dict),
            version="e2e_colab_mode",
            save_dir=str(mac_save_dir),
            colab_split_dir=str(tok_dir),
            stage_to_local=False,
            resume_ckpt_path=ckpt_step2,
            val_max_batches=1,
        )
        res_done = trainer_resume_done.run(project_name=None)
        assert res_done["status"] == "completed"
        assert res_done["final_step"] == 2
        assert res_done["final_val_loss"] > 0.0

        # Resume Case 2: Resume from outer_idx=0 (step=1 of 2) -> executes only outer_idx=1 and finishes at step=2
        ckpt_data_0 = torch.load(ckpt_step2, map_location="cpu", weights_only=False)
        ckpt_data_0["outer_batch_idx"] = 0
        ckpt_data_0["optimizer_step"] = 1
        ckpt_step1 = os.path.join(worker._get_worker_dir(), "MORTM5.e2e_outer0.checkpoint.pt")
        torch.save(ckpt_data_0, ckpt_step1)
        trainer_resume_mid = RemoteMORTM5HybridTrainer(
            client=_InProcessClient(),
            m_args=mortm.MORTM5Args(m_dict),
            t_args=mortm.TrainArgs(t_dict),
            version="e2e_outer0",
            save_dir=str(mac_save_dir),
            colab_split_dir=str(tok_dir),
            stage_to_local=False,
            resume_ckpt_path=ckpt_step1,
            val_max_batches=1,
        )
        res_mid = trainer_resume_mid.run(project_name=None)
        assert res_mid["status"] == "completed"
        assert res_mid["final_step"] == 2




