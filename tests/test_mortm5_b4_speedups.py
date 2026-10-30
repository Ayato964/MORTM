import asyncio
import json
import os
import time
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch

import mortm
from mortm.models.modules.attention import (
    BatchedSdpaVarlenQkvStrategy,
    FlashSelfAttentionM,
    PackedVarlenLayout,
    RotaryEmbedding,
    pad_input,
    unpad_input,
)
from mortm.models.modules.config import MORTM5Args, MORTMArgs, resolve_ablation_subdir
from mortm.train.config import TrainArgs
from mortm.train.datasets import MORTM5Collate, MORTM5MemmapDataset
from mortm.train.noam import get_cosine_schedule_with_warmup


def _reference_per_sequence_varlen_sdpa(
    qkv: torch.Tensor,
    cu_seqlens: torch.Tensor,
    causal: bool,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Exact per-sequence reference SDPA without cross-sequence padding."""
    cu = cu_seqlens.detach().cpu().tolist()
    outs: List[torch.Tensor] = []
    for i in range(len(cu) - 1):
        s, e = int(cu[i]), int(cu[i + 1])
        if e <= s:
            continue
        q_i, k_i, v_i = qkv[s:e].unbind(dim=1)
        o_i = torch.nn.functional.scaled_dot_product_attention(
            q_i.permute(1, 0, 2).unsqueeze(0),
            k_i.permute(1, 0, 2).unsqueeze(0),
            v_i.permute(1, 0, 2).unsqueeze(0),
            dropout_p=0.0,
            is_causal=causal,
            scale=softmax_scale,
        )
        outs.append(o_i.squeeze(0).permute(1, 0, 2))
    if not outs:
        return torch.empty((0, qkv.size(2), qkv.size(3)), device=qkv.device, dtype=qkv.dtype)
    return torch.cat(outs, dim=0)


def test_packed_varlen_layout_and_unpad_pad_fastpaths():
    device = torch.device("cpu")

    # Case 1: Uniform batch -> zero-copy fast path
    x_uni = torch.randn(4, 16, 32, device=device)
    mask_uni = torch.ones(4, 16, dtype=torch.bool, device=device)
    unpadded, indices, cu_seqlens, max_seqlen, seqlens = unpad_input(x_uni, mask_uni)
    layout_uni = PackedVarlenLayout(cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, total_tokens=unpadded.size(0))
    assert layout_uni.is_uniform is True
    assert layout_uni.batch_size == 4
    assert torch.equal(layout_uni.position_ids(device), torch.arange(16).repeat(4))
    re_padded = pad_input(unpadded, indices, 4, 16)
    assert torch.equal(re_padded, x_uni)

    # Case 2: Variable-length right-padded batch
    lengths = [17, 5, 29, 1]
    s_max = max(lengths)
    x_var = torch.randn(len(lengths), s_max, 32, device=device)
    mask_var = torch.zeros(len(lengths), s_max, dtype=torch.bool, device=device)
    for b, length in enumerate(lengths):
        mask_var[b, :length] = True
    unpadded_v, indices_v, cu_v, max_v, _ = unpad_input(x_var, mask_var)
    layout_var = PackedVarlenLayout(cu_seqlens=cu_v, max_seqlen=max_v, total_tokens=unpadded_v.size(0))
    assert layout_var.is_uniform is False
    expected_pos = torch.cat([torch.arange(length, dtype=torch.long) for length in lengths])
    assert torch.equal(layout_var.position_ids(device), expected_pos)
    assert torch.equal(layout_var.flat_padded_indices(device), indices_v)

    re_padded_v = pad_input(unpadded_v, indices_v, len(lengths), s_max)
    for b, length in enumerate(lengths):
        assert torch.equal(re_padded_v[b, :length], x_var[b, :length])
        assert torch.count_nonzero(re_padded_v[b, length:]) == 0


@pytest.mark.parametrize(
    "lengths,causal,dtype,atol",
    [
        ([16, 16, 16, 16], True, torch.float32, 1e-5),
        ([16, 16, 16, 16], False, torch.float32, 1e-5),
        ([17, 5, 29, 1], True, torch.float32, 1e-5),
        ([12, 0, 9, 4], True, torch.float32, 1e-5),
        ([17, 5, 29, 1], True, torch.bfloat16, 2e-2),
        ([11, 7, 19, 3], False, torch.float32, 1e-5),
    ],
)
def test_batched_sdpa_varlen_qkv_strategy_forward_and_backward_equivalence(
    lengths: List[int],
    causal: bool,
    dtype: torch.dtype,
    atol: float,
):
    torch.manual_seed(42)
    strategy = BatchedSdpaVarlenQkvStrategy()
    cu = torch.tensor([0] + list(np.cumsum(lengths)), dtype=torch.int32)
    total_tokens = int(cu[-1].item())
    max_seqlen = max(lengths)
    n_heads, head_dim = 4, 32

    base_qkv = torch.randn(total_tokens, 3, n_heads, head_dim, dtype=dtype)
    grad_out = torch.randn(total_tokens, n_heads, head_dim, dtype=dtype)

    qkv_fast = base_qkv.clone().detach().requires_grad_(True)
    out_fast = strategy.execute(
        qkv=qkv_fast,
        cu_seqlens=cu,
        max_seqlen=max_seqlen,
        dropout_p=0.0,
        causal=causal,
    )
    (out_fast * grad_out).sum().backward()

    qkv_ref = base_qkv.clone().detach().requires_grad_(True)
    out_ref = _reference_per_sequence_varlen_sdpa(
        qkv=qkv_ref,
        cu_seqlens=cu,
        causal=causal,
    )
    (out_ref * grad_out).sum().backward()

    assert torch.allclose(out_fast, out_ref, atol=atol, rtol=atol)
    assert qkv_fast.grad is not None and qkv_ref.grad is not None
    assert torch.allclose(qkv_fast.grad, qkv_ref.grad, atol=atol, rtol=atol)
    assert not torch.isnan(qkv_fast.grad).any()


def test_dynamic_rope_and_kv_cache_expansion_past_initial_capacity():
    torch.manual_seed(7)
    m_dict = MORTM5Args("#8").to_dict()
    m_dict.update(
        {
            "d_model": 64,
            "d_layer": 2,
            "num_heads": 4,
            "dim_feedforward": 128,
            "position_length": 8,  # Initial KV cache allocated for 8 + 512 = 520 soft tokens
            "dropout": 0.0,
        }
    )
    model = mortm.AblationTransformer2(MORTM5Args(m_dict)).to(device="cpu").eval()
    seq = torch.randint(133, 300, (1, 528, 4), dtype=torch.long)

    with torch.no_grad():
        full_logits = model(seq, is_causal=True, is_save_cache=False)
        model.reset_kv_cache()
        # Prefill 518 soft tokens (just below 520 initial capacity)
        prefill_logits = model(seq[:, :518, :], is_causal=True, is_save_cache=True)
        assert prefill_logits.shape[:2] == (1, 518)

        # Generate 10 more single-step soft tokens -> crosses 520 boundary dynamically without IndexError
        step_outs = []
        for pos in range(518, 528):
            step_logits = model(seq[:, pos : pos + 1, :], is_causal=True, is_save_cache=True)
            assert step_logits.shape[:2] == (1, 1)
            assert not torch.isnan(step_logits).any()
            step_outs.append(step_logits)

        cat_steps = torch.cat(step_outs, dim=1)
        assert torch.allclose(full_logits[:, 518:528, :, :].float(), cat_steps.float(), atol=5e-2)
        attn0 = model.decoder.layers[0].self_attention
        assert attn0.cache_seqlens is not None and int(attn0.cache_seqlens[0].item()) == 528
        assert attn0.kv_cache is not None and attn0.kv_cache[0].size(1) >= 528
        assert attn0.rotary_emb.max_position_embeddings >= 528
        model.reset_kv_cache()


def test_resolve_ablation_subdir_preserves_presets_and_custom_ids():
    assert (
        resolve_ablation_subdir("MORTM5_M4_Grouped_Shared", "configs/models/mortm5/ablation/ablation7.json")
        == "7"
    )
    assert resolve_ablation_subdir("ablation_#8", "#8") == "8"
    assert resolve_ablation_subdir("MORTM5_SyMIc_G4L4_160M", "symic_g4l4_160m_4ep") == "symic_g4l4_160m_4ep"
    assert (
        resolve_ablation_subdir("", "configs/models/mortm5/symic_main/symic_g4l4_160m.json")
        == "default"
    )


def test_staging_service_prefetch_loader_and_multi_epoch_sync_drive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import sys

    root_dir = str(Path(__file__).resolve().parents[1])
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

    import colab_gpu_worker as worker
    from run_remote_mortm5 import RemoteColabMORTM5Client, RemoteMORTM5HybridTrainer

    tok_dir = tmp_path / "drive_split"
    tok_dir.mkdir(parents=True, exist_ok=True)
    for split, n_songs in (("train", 4), ("val", 2)):
        toks_per_song = 16 * 12
        raw = np.random.randint(3, 500, size=(n_songs * toks_per_song,), dtype=np.uint16)
        raw.tofile(str(tok_dir / f"{split}_tokens.bin"))
        offsets = np.array([[i * toks_per_song, toks_per_song] for i in range(n_songs)], dtype=np.int64)
        np.save(str(tok_dir / f"{split}_offsets.npy"), offsets)
        slots = np.tile(np.array([128, 0, 32, -1], dtype=np.int16), (n_songs, 1))
        np.save(str(tok_dir / f"{split}_slots.npy"), slots)

    cache_root = tmp_path / "nvme_cache"
    monkeypatch.setenv("MORTM5_LOCAL_CACHE_DIR", str(cache_root))
    monkeypatch.setenv("MORTM5_FORCE_STAGE", "1")
    monkeypatch.setenv("MORTM5_WORKER_DIR", str(tmp_path / "worker_ckpts"))

    # 1. Verify LocalTokenStagingService copies files and reuses cache on second call
    staging_svc = worker.LocalTokenStagingService()
    staging_svc.start_async(str(tok_dir))
    st1 = staging_svc.get_status()
    for _ in range(50):
        st1 = staging_svc.get_status()
        if st1["done"]:
            break
        time.sleep(0.02)
    assert st1["done"] is True and st1["error"] is None
    assert Path(st1["local_dir"], "train_tokens.bin").is_file()
    assert staging_svc.resolve_cached_dir(str(tok_dir)) == st1["local_dir"]

    # 2. Verify PrefetchedMicroBatchLoader covers all indices cleanly
    m_dict = MORTM5Args("#8").to_dict()
    m_dict.update(
        {
            "name": "MORTM5_SyMIc_G4L4_160M",
            "d_model": 32,
            "d_layer": 1,
            "num_heads": 4,
            "dim_feedforward": 64,
            "position_length": 64,
        }
    )
    m_args = MORTM5Args(m_dict)
    ds = MORTM5MemmapDataset.from_split_dir(token_dir=st1["local_dir"], split="train", args=m_args)
    collate = MORTM5Collate(mode=m_args.model_mode, vq_groups=m_args.vq_groups)
    loader = worker.PrefetchedMicroBatchLoader(
        dataset=ds,
        indices=[0, 1, 2, 3],
        batch_size=2,
        collate_fn=collate,
        shuffle=True,
        pin_memory=False,
        prefetch_batches=2,
    )
    batches = list(loader)
    assert len(batches) == 2
    assert sum(int(b[0].size(0)) for b in batches) == 4

    # 3. Verify _detach_state_to_cpu produces independent CPU tensors
    nested = {"step": 5, "buf": torch.ones(4, dtype=torch.float32)}
    detached = worker._detach_state_to_cpu(nested)
    nested["buf"].fill_(9.0)
    assert detached["buf"].device.type == "cpu"
    assert torch.all(detached["buf"] == 1.0)

    # 4. Verify multi-epoch RemoteMORTM5HybridTrainer stages tokens & sets sync_drive ONLY on final epoch's final outer batch
    t_dict = TrainArgs("muon_cos").to_dict()
    t_dict.update(
        {
            "batch_size": 2,
            "big_batch_size": 2,
            "accumulation_steps": 1,
            "num_epochs": 2,
            "scheduler": {"type": "wsd", "warmup_steps": 1, "stable_ratio": 0.5, "min_lr_ratio": 0.1},
            "checkpoint_percents": [50, 100],
        }
    )

    class _DummyReq:
        def __init__(self, json_data=None, query_params=None, headers=None):
            self._json = json_data or {}
            self.query_params = query_params or {}
            self.headers = headers or {"content-type": "application/json"}

        async def json(self):
            return self._json

    sync_drive_calls: List[tuple[int, int, bool]] = []

    class _StagingInProcessClient(RemoteColabMORTM5Client):
        def __init__(self):
            super().__init__("http://inprocess.local")

        def health(self):
            return asyncio.run(worker.health())

        def start_stage_tokens(self, colab_split_dir: str):
            return asyncio.run(worker.mortm5_start_stage(_DummyReq(json_data={"colab_split_dir": colab_split_dir})))

        def poll_stage_tokens(self):
            return asyncio.run(worker.mortm5_poll_stage(_DummyReq()))

        def init_mortm5(self, init_payload):
            return asyncio.run(worker.mortm5_init(_DummyReq(json_data=init_payload)))

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
            sync_drive_calls.append((int(epoch), int(outer_idx), bool(sync_drive)))
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

    trainer = RemoteMORTM5HybridTrainer(
        client=_StagingInProcessClient(),
        m_args=m_args,
        t_args=TrainArgs(t_dict),
        version="symic_4ep_smoke",
        save_dir=str(tmp_path / "mac_out"),
        colab_split_dir=str(tok_dir),
        stage_to_local=True,
        use_torch_compile=False,
        ablation_id="symic_g4l4_160m_4ep",
        download_ckpt_to_mac=False,
        val_max_batches=1,
    )
    res = trainer.run(project_name=None)
    assert res["status"] == "completed"
    assert res["final_step"] == 4  # 2 outer batches/epoch * 2 epochs = 4 steps
    assert res["ablation_subdir"] == "symic_g4l4_160m_4ep"
    # Only the very last save call (epoch=1, outer_idx=1) should force synchronous sync_drive=True
    assert sync_drive_calls[-1] == (1, 1, True)
    assert all(sd is False for _, _, sd in sync_drive_calls[:-1])


def test_symic_4ep_production_config_integrity():
    repo_root = Path(__file__).resolve().parents[1]
    t_cfg_path = repo_root / "configs" / "train" / "mortm5" / "symic_main_160m_4ep.json"
    m_cfg_path = repo_root / "configs" / "models" / "mortm5" / "symic_main" / "symic_g4l4_160m.json"
    assert t_cfg_path.is_file()
    assert m_cfg_path.is_file()

    t_args = TrainArgs(str(t_cfg_path))
    m_args = MORTM5Args(str(m_cfg_path))
    assert t_args.num_epochs == 4
    assert t_args.big_batch_size == 16384
    assert t_args.batch_size == 128
    assert t_args.accumulation_steps == 2
    assert t_args.scheduler["type"] == "wsd"
    assert t_args.scheduler["total_steps"] == 58440
    assert t_args.scheduler["warmup_steps"] == 730
    assert t_args.scheduler["stable_ratio"] == pytest.approx(0.90)
    assert m_args.d_model == 1024
    assert m_args.d_layer == 12
    assert m_args.num_heads == 16
    assert m_args.dim_feedforward == 3072

    stable_steps = int((58440 - 730) * 0.90)
    lr_lambda = get_cosine_schedule_with_warmup(
        total_steps=58440,
        warmup_steps=730,
        min_lr_ratio=0.02,
        stable_steps=stable_steps,
    )
    # Step 730 -> peak 1.0; Step 730 + stable_steps - 1 -> 1.0; Step 58440 -> 0.02
    assert lr_lambda(730) == pytest.approx(1.0)
    assert lr_lambda(730 + stable_steps - 1) == pytest.approx(1.0)
    assert lr_lambda(58440) == pytest.approx(0.02)
    assert mortm.__version__ == "5.0.0b4"
