"""Unit and integration tests for Unified CARL (`MORTM5 SyMIc` & `MORTM4.5` engine)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from mortm.models.modules.config import MORTM5Args
from mortm.models.mortm5 import MidiVision, build_mortm5_ablation_model
from mortm.train.rl import (
    CARLActorAdapter,
    CARLCriticAdapter,
    CARLExample,
    GuardrailContextBuilder,
    GuardrailRule,
    GuardrailSuite,
    LoRALinearAdapter,
    RolloutBatch,
    StepLogProbsKL,
    SyMIcCriticAdapter,
    SyMIcGroupedActorAdapter,
    SyMIcMemmapRepository,
    UnifiedCARLDriver,
    UnifiedEvenConfig,
    UnifiedOddConfig,
    UnifiedSamplingConfig,
    build_symic_guardrail_suite,
)
from mortm.train.rl.interfaces import InMemorySamplePool


def _tiny_symic_config() -> Dict[str, Any]:
    return {
        "name": "Tiny_SyMIc_CARL_Test",
        "tokenizer_key": "m4_g4l4",
        "model_mode": "grouped",
        "separate_rvq_layers": True,
        "use_instrument_tokens": True,
        "token_offset": 133,
        "vocab_size": 1157,  # 133 + 4 * 4 * 64
        "vq_groups": 4,
        "vq_layers": 4,
        "encoder_layer": 2,
        "encoder_wout": 32,
        "codebook_size": 64,
        "d_model": 64,
        "d_layer": 2,
        "num_heads": 4,
        "dim_feedforward": 128,
        "dropout": 0.1,
        "position_length": 128,
        "norm_type": "rmsnorm",
        "use_rope": True,
        "use_silu": True,
        "use_bias": False,
        "use_moe_decoder": False,
    }


def _tiny_vq_config() -> Dict[str, Any]:
    return {
        "name": "Tiny_MidiVision_Test",
        "d_model": 64,
        "dim_feedforward": 64,
        "num_heads": 4,
        "dropout": 0.0,
        "roll_w": 24,
        "roll_h": 128,
        "track_size": 4,
        "first_channel": 16,
        "max_channel": 32,
        "bottle_channel": 32,
        "encoder_wout": 32,
        "vq_groups": 4,
        "encoder_layer": 4,
        "codebook_size": 64,
        "position_length": 512,
        "token_offset": 133,
    }


def _select_test_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _make_synthetic_symic_example(
    model_args: MORTM5Args,
    song_id: int,
    prompt_bars: int = 1,
    total_bars: int = 2,
    seed: int = 0,
) -> CARLExample:
    rng = np.random.default_rng(seed + song_id)
    g = int(model_args.vq_groups)
    l_cnt = int(model_args.vq_layers)
    k_sz = int(model_args.codebook_size)
    offset = int(model_args.token_offset)

    prompt_chunks = prompt_bars * 8
    total_chunks = total_bars * 8
    prefix = np.array([[1] * g, [131] * g, [3] * g, [35] * g, [132] * g], dtype=np.int64)

    rvq_rows: List[np.ndarray] = []
    for _ in range(total_chunks):
        for l_i in range(l_cnt):
            layer_base = offset + l_i * (g * k_sz)
            codes = np.array(
                [layer_base + g_i * k_sz + int(rng.integers(0, k_sz)) for g_i in range(g)],
                dtype=np.int64,
            )
            rvq_rows.append(codes)

    full = np.concatenate([prefix, np.stack(rvq_rows, axis=0)], axis=0)
    prompt_soft_len = 5 + prompt_chunks * l_cnt
    return CARLExample(
        example_id=f"test_ex_{song_id:03d}",
        song_id=song_id,
        prompt_tokens=full[:prompt_soft_len].copy(),
        human_full_tokens=full.copy(),
        slot_codes=(128, 0, 32, -1),
    )


def test_lora_linear_adapter_identity_and_no_merge_mutation() -> None:
    torch.manual_seed(0)
    base = nn.Linear(32, 48, bias=False)
    orig_weight = base.weight.detach().clone()

    wrapped = LoRALinearAdapter(base, rank=4, alpha=8.0)
    x = torch.randn(3, 5, 32)

    # Zero-initialized lora_B guarantees exact identity at initialization.
    assert torch.allclose(wrapped(x), base(x), atol=1e-6)

    # Mutate lora_B and toggle train/eval repeatedly: base weight must NEVER change.
    nn.init.normal_(wrapped.lora_B, std=0.1)
    wrapped.eval()
    out_eval_1 = wrapped(x)
    wrapped.train()
    wrapped.eval()
    out_eval_2 = wrapped(x)

    assert torch.equal(base.weight, orig_weight)
    assert torch.allclose(out_eval_1, out_eval_2, atol=1e-6)


def test_symic_actor_unfreeze_exact_kl_and_depth_gradients() -> None:
    dev = _select_test_device()
    m_args = MORTM5Args(_tiny_symic_config())
    model_a = build_mortm5_ablation_model(m_args).to(dev)
    model_a.Wout._legacy_untrained_depth.fill_(False)

    model_b = build_mortm5_ablation_model(m_args).to(dev)
    model_b.load_state_dict(model_a.state_dict())

    actor = SyMIcGroupedActorAdapter(model_a, dev, prompt_bars=1, total_bars=2)
    base_actor = SyMIcGroupedActorAdapter(model_b, dev, prompt_bars=1, total_bars=2)
    base_actor.freeze_all()

    n_train, n_total = actor.unfreeze_for_grpo()
    assert 0 < n_train < n_total
    assert not model_a.embedding.weight.requires_grad
    assert not model_a.decoder.layers[0].ffn.w1.weight.requires_grad
    assert model_a.decoder.layers[0].self_attention.qkv_block.qkv_weight.weight.requires_grad
    assert model_a.Wout.depth_decoder.layers[0].q_up.weight.requires_grad
    assert model_a.Wout.depth_decoder.layers[0].self_attention.qkv_block.qkv_weight.weight.requires_grad

    actor.prepare_for_grpo_forward()
    ex = _make_synthetic_symic_example(m_args, song_id=1, prompt_bars=1, total_bars=2)
    stats = actor.forward_logprobs_and_exact_kl(
        base_actor=base_actor,
        examples=[ex],
        generated_sequences=[ex.human_full_tokens],
    )
    # Identical Actor & Base in `.eval()` mode must yield 0 exact KL!
    assert float(stats.exact_kl.abs().max().item()) < 1e-5
    assert stats.action_logprobs.shape == (1, 8 * 4)

    loss = -stats.action_logprobs.mean()
    loss.backward()
    assert model_a.decoder.layers[0].self_attention.qkv_block.qkv_weight.weight.grad is not None
    assert model_a.Wout.depth_decoder.layers[0].q_up.weight.grad is not None


def test_symic_memmap_repository_strict_song_disjoint_split(tmp_path: Path) -> None:
    m_args = MORTM5Args(_tiny_symic_config())
    g = int(m_args.vq_groups)
    l_cnt = int(m_args.vq_layers)
    k_sz = int(m_args.codebook_size)
    tokens_per_chunk = g * l_cnt
    chunks_per_song = 16  # 2 bars

    rng = np.random.default_rng(123)
    num_songs = 6
    all_tokens: List[np.ndarray] = []
    offsets: List[Tuple[int, int]] = []
    slots: List[List[int]] = []
    cursor = 0
    for _ in range(num_songs):
        song_len = chunks_per_song * tokens_per_chunk
        # Raw .bin uses RVQ_TOKEN_OFFSET=3 + group*K + k
        song_toks = rng.integers(3, 3 + g * k_sz, size=(song_len,), dtype=np.uint16)
        all_tokens.append(song_toks)
        offsets.append((cursor, song_len))
        slots.append([128, 0, 32, -1])
        cursor += song_len

    np.concatenate(all_tokens).tofile(tmp_path / "train_tokens.bin")
    np.save(tmp_path / "train_offsets.npy", np.asarray(offsets, dtype=np.int64))
    np.save(tmp_path / "train_slots.npy", np.asarray(slots, dtype=np.int16))

    repo = SyMIcMemmapRepository(
        token_dir=tmp_path,
        model_args=m_args,
        split="train",
        prompt_bars=1,
        total_bars=2,
        min_active_tracks=2,
    )
    ana_pool, gem_pool = repo.load_disjoint_pools(seed=7)
    assert len(ana_pool) == 3
    assert len(gem_pool) == 3

    ana_songs = {ana_pool[i].song_id for i in range(len(ana_pool))}
    gem_songs = {gem_pool[i].song_id for i in range(len(gem_pool))}
    assert ana_songs.isdisjoint(gem_songs)
    assert ana_pool[0].prompt_tokens.shape == (5 + 8 * 4, 4)
    assert ana_pool[0].human_full_tokens.shape == (5 + 16 * 4, 4)


def test_symic_guardrails_zero_on_human_and_end_to_end_carl_driver(tmp_path: Path) -> None:
    dev = _select_test_device()
    m_args = MORTM5Args(_tiny_symic_config())
    vq_args = MORTM5Args(_tiny_vq_config())

    vq = MidiVision(vq_args, sync_codebook=False).to(dev).eval()
    # Bias MidiVision decoder slightly positive so synthetic random tokens produce active notes.
    with torch.no_grad():
        for mod in vq.decoder.modules():
            if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)) and mod.bias is not None:
                mod.bias.add_(0.25)

    m_actor = build_mortm5_ablation_model(m_args).to(dev)
    m_actor.Wout._legacy_untrained_depth.fill_(False)
    m_base = build_mortm5_ablation_model(m_args).to(dev)
    m_base.load_state_dict(m_actor.state_dict())

    actor = SyMIcGroupedActorAdapter(m_actor, dev, prompt_bars=1, total_bars=2)
    base_actor = SyMIcGroupedActorAdapter(m_base, dev, prompt_bars=1, total_bars=2)
    base_actor.freeze_all()
    critic = SyMIcCriticAdapter.from_pretrained(
        m_args, ckpt_path=None, device=dev, prompt_bars=1, lora_rank=4, lora_alpha=8.0
    )

    ex_ana = [_make_synthetic_symic_example(m_args, song_id=i, prompt_bars=1, total_bars=2) for i in range(4)]
    ex_gem = [_make_synthetic_symic_example(m_args, song_id=10 + i, prompt_bars=1, total_bars=2) for i in range(4)]

    # 1. Verify non-copy guardrails evaluate to strictly 0.0 on Human Ground Truth.
    suite_no_copy = build_symic_guardrail_suite(vq, actor, enable_gt_copy_penalty=False)
    gt_batch = RolloutBatch(
        examples=(ex_gem[0], ex_gem[1]),
        generated_sequences=(ex_gem[0].human_full_tokens, ex_gem[1].human_full_tokens),
        critic_sequences=(ex_gem[0].human_full_tokens, ex_gem[1].human_full_tokens),
        group_size=1,
    )
    gt_eval = suite_no_copy.evaluate(gt_batch)
    if not bool(gt_eval.is_empty.all()):
        non_empty = ~gt_eval.is_empty
        assert torch.allclose(
            gt_eval.penalties[non_empty], torch.zeros_like(gt_eval.penalties[non_empty]), atol=1e-6
        )

    # 2. Verify full Odd -> Even UnifiedCARLDriver cycle + checkpoint persistence + resume.
    full_suite = build_symic_guardrail_suite(vq, actor, enable_gt_copy_penalty=True)
    odd_cfg = UnifiedOddConfig(
        batch_size=4,
        pool_size=4,
        lr=1e-3,
        max_steps=3,
        eval_window=2,
        log_every=2,
        gen_chunk=4,
        critic_chunk=4,
        sampling=UnifiedSamplingConfig(temperature=1.0, fine_temperature=0.7, top_p=0.95),
    )
    even_cfg = UnifiedEvenConfig(
        group_size=2,
        prompts_per_step=2,
        lr=1e-4,
        max_steps=2,
        gen_chunk=4,
        critic_chunk=4,
        train_chunk=2,
        stop_window=2,
        snapshot_every=1,
        log_every=1,
        sampling=UnifiedSamplingConfig(temperature=1.0, fine_temperature=0.7, top_p=0.95, min_p=0.05),
    )

    out_dir = tmp_path / "carl_run"
    driver = UnifiedCARLDriver(
        actor=actor,
        base_actor=base_actor,
        critic=critic,
        ana_pool=InMemorySamplePool(ex_ana),
        gem_pool=InMemorySamplePool(ex_gem),
        guardrail_suite=full_suite,
        out_dir=out_dir,
        odd_cfg=odd_cfg,
        even_cfg=even_cfg,
        q_max=16,
        seed=42,
        ckpt_prefix="SyMIcTest",
        use_wandb=False,
    )
    history = driver.run(max_total_steps=10, target_hits=2, start_round=1, max_rounds=2)
    assert len(history) == 2
    assert history[0]["kind"] == "odd"
    assert history[1]["kind"] == "even"
    assert len(driver.qtable) > 0
    assert (out_dir / "SyMIcTest-ana.carl.r1.pth").is_file()
    assert (out_dir / "SyMIcTest-gem.carl.r2.pth").is_file()
    assert (out_dir / "qtable.r1.pt").is_file()

    # Verify resume roundtrip onto fresh adapters.
    driver.resume(
        actor_ckpt=out_dir / "SyMIcTest-gem.carl.r2.pth",
        critic_ckpt=out_dir / "SyMIcTest-ana.carl.r1.pth",
        qtable_ckpt=out_dir / "qtable.r1.pt",
    )


def test_generic_engine_supports_custom_symbolic_adapter(tmp_path: Path) -> None:
    """Prove `UnifiedCARLDriver` is genuinely model-agnostic (MORTM4.5 / Symbolic compatible)."""

    class _TinySymbolicActor(CARLActorAdapter):
        def __init__(self) -> None:
            self._net = nn.Sequential(nn.Embedding(32, 16), nn.Linear(16, 32))

        @property
        def actor_module(self) -> nn.Module:
            return self._net

        @property
        def device(self) -> torch.device:
            return torch.device("cpu")

        def unfreeze_for_grpo(self) -> Tuple[int, int]:
            for p in self._net.parameters():
                p.requires_grad_(True)
            tot = sum(p.numel() for p in self._net.parameters())
            return tot, tot

        def rollout_group(
            self,
            examples: Sequence[CARLExample],
            group_size: int,
            sampling: UnifiedSamplingConfig,
            chunk_size: int = 16,
            seed: Optional[int] = None,
        ) -> RolloutBatch:
            exs, gens = [], []
            for ex in examples:
                for g_i in range(group_size):
                    exs.append(ex)
                    gen = np.array([5, 6, 7 + (g_i % 5), 8], dtype=np.int64)
                    gens.append(gen)
            return RolloutBatch(tuple(exs), tuple(gens), tuple(gens), group_size)

        def forward_logprobs_and_exact_kl(
            self,
            base_actor: CARLActorAdapter,
            examples: Sequence[CARLExample],
            generated_sequences: Sequence[np.ndarray],
        ) -> StepLogProbsKL:
            x = torch.as_tensor(np.stack(generated_sequences), dtype=torch.long)
            logp = torch.log_softmax(self._net(x), dim=-1)
            with torch.no_grad():
                logp_b = torch.log_softmax(base_actor.actor_module(x), dim=-1)
            act_lp = logp.gather(-1, x.unsqueeze(-1)).squeeze(-1)
            kl = (logp.exp() * (logp - logp_b)).sum(dim=-1)
            return StepLogProbsKL(act_lp, kl, torch.ones_like(act_lp, dtype=torch.bool))

        def export_checkpoint_state(self) -> Dict[str, Any]:
            return {k: v.clone() for k, v in self._net.state_dict().items()}

        def load_checkpoint_state(self, state: Dict[str, Any]) -> None:
            self._net.load_state_dict(state)

    class _TinySymbolicCritic(CARLCriticAdapter):
        def __init__(self) -> None:
            self._emb = nn.Embedding(32, 8)
            self._head = nn.Linear(8, 1)
            self._mod = nn.ModuleDict({"emb": self._emb, "head": self._head})

        @property
        def critic_module(self) -> nn.Module:
            return self._mod

        @property
        def device(self) -> torch.device:
            return torch.device("cpu")

        def mark_trainable(self) -> Dict[str, int]:
            for p in self._emb.parameters():
                p.requires_grad_(False)
            for p in self._head.parameters():
                p.requires_grad_(True)
            return {"head": sum(p.numel() for p in self._head.parameters())}

        def score_disc_logits(
            self,
            critic_sequences: Sequence[np.ndarray],
            chunk_size: Optional[int] = None,
            grad_enabled: bool = False,
        ) -> Tensor:
            x = torch.as_tensor(np.stack(critic_sequences), dtype=torch.long)
            ctx = torch.enable_grad() if grad_enabled else torch.no_grad()
            with ctx:
                return self._head(self._emb(x).mean(dim=1)).squeeze(-1)

        def export_compact_state(self) -> Dict[str, Any]:
            return {k: v.clone() for k, v in self._head.state_dict().items()}

        def load_compact_state(self, state: Dict[str, Any]) -> None:
            self._head.load_state_dict(state)

    class _NoOpBuilder(GuardrailContextBuilder[int]):
        def build_context(self, batch: RolloutBatch) -> int:
            return len(batch)

    class _ZeroRule(GuardrailRule[int]):
        @property
        def name(self) -> str:
            return "zero"

        def evaluate(self, context: int) -> Tuple[Tensor, Dict[str, float]]:
            return torch.zeros((context,), dtype=torch.float32), {}

    actor = _TinySymbolicActor()
    base = _TinySymbolicActor()
    base.load_checkpoint_state(actor.export_checkpoint_state())
    base.freeze_all()
    critic = _TinySymbolicCritic()
    pool = InMemorySamplePool(
        [
            CARLExample(
                example_id=f"s_{i}",
                song_id=i,
                prompt_tokens=np.array([1, 2], dtype=np.int64),
                human_full_tokens=np.array([3, 4, 5, 6], dtype=np.int64),
            )
            for i in range(4)
        ]
    )
    suite = GuardrailSuite(_NoOpBuilder(), lambda n: torch.zeros((n,), dtype=torch.bool), (_ZeroRule(),))
    drv = UnifiedCARLDriver(
        actor=actor,
        base_actor=base,
        critic=critic,
        ana_pool=pool,
        gem_pool=pool,
        guardrail_suite=suite,
        out_dir=tmp_path / "sym_carl",
        odd_cfg=UnifiedOddConfig(batch_size=4, pool_size=4, max_steps=2),
        even_cfg=UnifiedEvenConfig(group_size=2, prompts_per_step=2, max_steps=2, train_chunk=2),
        q_max=8,
    )
    hist = drv.run(max_rounds=2)
    assert len(hist) == 2



def test_master_weight_adamw_updates_bf16_attention_at_low_lr() -> None:
    """Regression test for Defect #1: `bfloat16` parameters must accumulate `lr=2e-6` updates."""
    from mortm.train.rl.optim import MasterWeightAdamW

    # `|w| = 0.03125` in bf16 has half-ULP `2^-13 ≈ 1.22e-4`, which is 61x larger than `lr = 2e-6`.
    w_bf16 = nn.Parameter(torch.full((8, 8), 0.03125, dtype=torch.bfloat16))
    initial = w_bf16.detach().clone()

    # Verify naive AdamW stagnates completely (`bf16(w - 2e-6) == w`), whereas MasterWeightAdamW crosses ULP!
    opt = MasterWeightAdamW([w_bf16], lr=2e-6, weight_decay=0.0)
    assert opt.num_low_precision_tensors == 1
    for _ in range(80):
        w_bf16.grad = torch.ones_like(w_bf16)
        opt.step_and_zero_grad(grad_clip=1.0)

    assert not torch.equal(w_bf16, initial)
    assert float((initial.float() - w_bf16.float()).mean().item()) > 1e-4

    # Also verify LoRALinearAdapter keeps `lora_A` / `lora_B` in float32 even when base Linear is bf16!
    bf16_linear = nn.Linear(16, 16, bias=False).to(dtype=torch.bfloat16)
    lora_wrap = LoRALinearAdapter(bf16_linear, rank=4, alpha=8.0)
    assert lora_wrap.lora_A.dtype == torch.float32
    assert lora_wrap.lora_B.dtype == torch.float32


def test_reward_hacking_rollback_immediately_commits_to_qtable_and_persists(tmp_path: Path) -> None:
    """Regression test for Defect #2 & #7: hacked rollouts must enter `qtable.table` immediately and snapshot on CPU."""

    class _DriftingActor(CARLActorAdapter):
        def __init__(self) -> None:
            self._net = nn.Sequential(nn.Embedding(16, 8), nn.Linear(8, 16))

        @property
        def actor_module(self) -> nn.Module:
            return self._net

        @property
        def device(self) -> torch.device:
            return torch.device("cpu")

        def unfreeze_for_grpo(self) -> Tuple[int, int]:
            for p in self._net.parameters():
                p.requires_grad_(True)
            tot = sum(p.numel() for p in self._net.parameters())
            return tot, tot

        def rollout_group(
            self,
            examples: Sequence[CARLExample],
            group_size: int,
            sampling: UnifiedSamplingConfig,
            chunk_size: int = 16,
            seed: Optional[int] = None,
        ) -> RolloutBatch:
            exs, gens = [], []
            for ex in examples:
                for g_i in range(group_size):
                    exs.append(ex)
                    gens.append(np.array([9, 9, 9, 10 + (g_i % 3)], dtype=np.int64))
            return RolloutBatch(tuple(exs), tuple(gens), tuple(gens), group_size)

        def forward_logprobs_and_exact_kl(
            self,
            base_actor: CARLActorAdapter,
            examples: Sequence[CARLExample],
            generated_sequences: Sequence[np.ndarray],
        ) -> StepLogProbsKL:
            x = torch.as_tensor(np.stack(generated_sequences), dtype=torch.long)
            logp = torch.log_softmax(self._net(x), dim=-1)
            act_lp = logp.gather(-1, x.unsqueeze(-1)).squeeze(-1)
            # Simulate severe KL divergence (> kl_stop) to trigger anti-hacking rollback.
            high_kl = torch.full_like(act_lp, 0.95)
            return StepLogProbsKL(act_lp, high_kl, torch.ones_like(act_lp, dtype=torch.bool))

        def export_checkpoint_state(self) -> Dict[str, Any]:
            return {k: v.clone() for k, v in self._net.state_dict().items()}

        def load_checkpoint_state(self, state: Dict[str, Any]) -> None:
            self._net.load_state_dict(state)

    class _FooledCritic(CARLCriticAdapter):
        def __init__(self) -> None:
            self._p = nn.Linear(4, 1)

        @property
        def critic_module(self) -> nn.Module:
            return self._p

        @property
        def device(self) -> torch.device:
            return torch.device("cpu")

        def mark_trainable(self) -> Dict[str, int]:
            return {"p": 5}

        def score_disc_logits(
            self,
            critic_sequences: Sequence[np.ndarray],
            chunk_size: Optional[int] = None,
            grad_enabled: bool = False,
        ) -> Tensor:
            # Negative logit (-4.0) -> `R_disc = clamp(0.5 - (-4)/8) = 1.0 > disc_stop (0.8)`
            return torch.full((len(critic_sequences),), -4.0, dtype=torch.float32)

        def export_compact_state(self) -> Dict[str, Any]:
            return {}

        def load_compact_state(self, state: Dict[str, Any]) -> None:
            pass

    class _NoOpBuilder(GuardrailContextBuilder[int]):
        def build_context(self, batch: RolloutBatch) -> int:
            return len(batch)

    class _ZeroRule(GuardrailRule[int]):
        @property
        def name(self) -> str:
            return "zero"

        def evaluate(self, context: int) -> Tuple[Tensor, Dict[str, float]]:
            return torch.zeros((context,), dtype=torch.float32), {}

    actor = _DriftingActor()
    snap = actor.snapshot_state()
    assert all(t.device.type == "cpu" for t in snap.values())

    base = _DriftingActor()
    base.load_checkpoint_state(actor.export_checkpoint_state())
    base.freeze_all()
    critic = _FooledCritic()
    pool = InMemorySamplePool(
        [
            CARLExample(
                example_id=f"p_{i}",
                song_id=i,
                prompt_tokens=np.array([1, 2], dtype=np.int64),
                human_full_tokens=np.array([3, 4, 5, 6], dtype=np.int64),
            )
            for i in range(4)
        ]
    )
    suite = GuardrailSuite(_NoOpBuilder(), lambda n: torch.zeros((n,), dtype=torch.bool), (_ZeroRule(),))
    out_dir = tmp_path / "hack_test"
    drv = UnifiedCARLDriver(
        actor=actor,
        base_actor=base,
        critic=critic,
        ana_pool=pool,
        gem_pool=pool,
        guardrail_suite=suite,
        out_dir=out_dir,
        even_cfg=UnifiedEvenConfig(
            group_size=2,
            prompts_per_step=2,
            lr=1e-2,
            max_steps=5,
            stop_window=2,
            kl_stop=0.30,
            disc_stop=0.80,
            train_chunk=2,
        ),
        q_max=16,
    )
    # Pre-seed 1 item so `prefill_qtable` doesn't overwrite our clean check.
    drv.qtable.inject_committed([{"seq": np.array([1, 1, 1, 1], dtype=np.int64)}])
    even_out = drv.run_even(round_no=2)
    assert even_out["rolled_back"] is True
    assert even_out["injected_hacked"] == 4
    # Hacked items `[9, 9, 9, ...]` must already be inside `qtable.table` for immediate sampling!
    sampled = drv.qtable.sample(len(drv.qtable))
    assert any(int(it["seq"][0]) == 9 for it in sampled)
    assert (out_dir / "qtable.r2.pt").is_file()


def test_cli_backend_registry_defaults_and_guardrail_invariants() -> None:
    """Regression tests for Defects #3, #4, #5, #6, and #9."""
    from mortm.train.rl import carl_paths as legacy_paths
    from mortm.train.rl.adapters.mortm45 import Mortm45ActorAdapter
    from mortm.train.rl.adapters.symic import (
        SyMIcDecodedContext,
        SyMIcRelativeDensityRule,
        SyMIcScaleOutlierRule,
    )
    from mortm.train.rl.cli import (
        DEFAULT_SYMIC_CKPT,
        DEFAULT_SYMIC_CONFIG,
        build_arg_parser,
        resolve_default_paths_for_backend,
    )

    parser = build_arg_parser()
    args_m45 = parser.parse_args(["--backend", "mortm45", "--out-dir", "out/tmp_m45"])
    assert args_m45.config is None and args_m45.ckpt is None
    assert resolve_default_paths_for_backend("mortm45", args_m45.config, args_m45.ckpt) == (
        legacy_paths.GEM_CONFIG,
        legacy_paths.GEM_CKPT,
    )
    assert resolve_default_paths_for_backend("symic", None, None) == (
        DEFAULT_SYMIC_CONFIG,
        DEFAULT_SYMIC_CKPT,
    )

    # Defect #6: ScaleOutlierRule default `tau_floor` must match plan spec `0.15`.
    scale_rule = SyMIcScaleOutlierRule()
    assert abs(scale_rule._tau_floor - 0.15) < 1e-9

    # Defect #5: RelativeDensityRule must be 0.0 on Human GT AND 0.0 when `gt == 0` and Actor generates 20 notes.
    dens_rule = SyMIcRelativeDensityRule(min_ref_onsets=16.0)
    z = torch.zeros((2,), dtype=torch.float32)
    ctx = SyMIcDecodedContext(
        batch_size=2,
        is_empty=torch.zeros((2,), dtype=torch.bool),
        gen_onset_counts=torch.tensor([20.0, 2.0]),
        gt_onset_counts=torch.tensor([0.0, 2.0]),
        gen_blank_bar_ratio=z,
        gt_blank_bar_ratio=z,
        gen_out_of_scale=z,
        gt_out_of_scale=z,
        has_melodic_gt=torch.ones((2,), dtype=torch.bool),
        gt_copy_iou=z,
        gen_adj_2bar_iou=z,
        gt_adj_2bar_iou=z,
    )
    pens, _ = dens_rule.evaluate(ctx)
    assert torch.allclose(pens, torch.zeros_like(pens), atol=1e-6)

    # Defect #4: Mortm45ActorAdapter.load_checkpoint_state must reject un-marked checkpoints with non-zero lora_B.
    dummy_m45 = Mortm45ActorAdapter.__new__(Mortm45ActorAdapter)
    dummy_m45._model = nn.Linear(4, 4)
    with pytest.raises(RuntimeError, match="Legacy merged LoRA checkpoint"):
        dummy_m45.load_checkpoint_state({"layer.lora_B": torch.ones(4, 2)})



def test_symic_midi_repository_small_limit_does_not_monopolize_single_song(tmp_path: Path) -> None:
    """Regression test: small `limit=4` on long multi-window MIDIs must still split across >=2 songs."""
    import json
    from mortm.train.rl.adapters.symic import SyMIcMidiRepository
    from mortm.utils.convert_pianoroll import _notes_to_midi, slot_codes_to_roll_meta

    slot_codes = (128, 0, 32, -1)
    meta = slot_codes_to_roll_meta(slot_codes)
    midi_paths: List[str] = []
    # Create 2 long songs (12 bars each -> six 2-bar windows per song!) so song #0 alone has 6 windows > limit=4.
    for s_idx in range(2):
        tuples = []
        for bar in range(12):
            base_step = bar * 192
            for beat in range(4):
                st = base_step + beat * 48
                tuples.append((0, st, st + 24, 36, 100))
                tuples.append((1, st, st + 36, 60 + ((bar + s_idx) % 7), 95))
        mp = tmp_path / f"long_song_{s_idx}.mid"
        _notes_to_midi(tuples, meta, mp)
        midi_paths.append(str(mp))

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(midi_paths), encoding="utf-8")

    dev = _select_test_device()
    m_args = MORTM5Args(_tiny_symic_config())
    vq_args = MORTM5Args(_tiny_vq_config())
    model = build_mortm5_ablation_model(m_args).to(dev).eval()
    vq = MidiVision(vq_args, sync_codebook=False).to(dev).eval()

    repo = SyMIcMidiRepository(
        midi_manifest_path=manifest,
        model=model,
        vq_model=vq,
        device=dev,
        prompt_bars=1,
        total_bars=2,
        min_active_tracks=2,
        min_prompt_notes=2,
        min_cont_notes=2,
    )
    for lim in (2, 4):
        ana_p, gem_p = repo.load_disjoint_pools(limit=lim, seed=42)
        assert len(ana_p) >= 1 and len(gem_p) >= 1
        assert {ana_p[i].song_id for i in range(len(ana_p))}.isdisjoint(
            {gem_p[i].song_id for i in range(len(gem_p))}
        )



def test_mortm45_loralib_eval_mode_load_never_corrupts_weights() -> None:
    """Regression test for Critic Defect #1: loading unmerged state onto `.eval()` (`merged=True`) `loralib.Linear`."""
    import loralib as lora
    from mortm.train.rl.adapters.mortm45 import Mortm45ActorAdapter, Mortm45CriticAdapter

    torch.manual_seed(7)
    x = torch.randn(4, 16)

    # 1. Actor (`Mortm45ActorAdapter.export_checkpoint_state` / `load_checkpoint_state`)
    src_actor_net = nn.Sequential(lora.Linear(16, 16, r=4, lora_alpha=8))
    nn.init.normal_(src_actor_net[0].lora_A, std=0.2)
    nn.init.normal_(src_actor_net[0].lora_B, std=0.2)
    base_weight_expected = src_actor_net[0].weight.detach().clone()  # unmerged base weight in train()

    src_actor_net.eval()  # merges B @ A * scaling into weight (`merged=True`)
    expected_eval_out = src_actor_net(x).detach().clone()

    src_adapter = Mortm45ActorAdapter.__new__(Mortm45ActorAdapter)
    src_adapter._model = src_actor_net
    exported = src_adapter.export_checkpoint_state()

    # Target model is ALREADY in `.eval()` (`merged=True`) when `load_checkpoint_state` is called!
    dst_actor_net = nn.Sequential(lora.Linear(16, 16, r=4, lora_alpha=8))
    nn.init.normal_(dst_actor_net[0].lora_A, std=0.5)
    nn.init.normal_(dst_actor_net[0].lora_B, std=0.5)
    dst_actor_net.eval()
    assert dst_actor_net[0].merged is True

    dst_adapter = Mortm45ActorAdapter.__new__(Mortm45ActorAdapter)
    dst_adapter._model = dst_actor_net
    dst_adapter.load_checkpoint_state(exported)

    assert dst_actor_net.training is False
    assert dst_actor_net[0].merged is True
    assert torch.allclose(dst_actor_net(x), expected_eval_out, atol=1e-5)

    # Switching back to `.train()` must cleanly recover exact `base_weight_expected` (no reverse subtraction!).
    dst_actor_net.train()
    assert torch.allclose(dst_actor_net[0].weight, base_weight_expected, atol=1e-5)

    # 2. Critic (`Mortm45CriticAdapter.export_compact_state` / `load_compact_state`)
    class _DummyM45Critic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = lora.Linear(16, 16, r=4, lora_alpha=8)
            self.mil = nn.Linear(16, 1)

    ana_mod = _DummyM45Critic()
    frozen_bb_weight = ana_mod.attn.weight.detach().clone()
    nn.init.normal_(ana_mod.attn.lora_A, std=0.3)
    nn.init.normal_(ana_mod.attn.lora_B, std=0.3)
    ana_mod.eval()  # `merged=True` with old LoRA baked into `attn.weight`!

    c_adapter = Mortm45CriticAdapter.__new__(Mortm45CriticAdapter)
    c_adapter._ana = ana_mod  # type: ignore[assignment]

    new_compact = {
        "attn.lora_A": torch.full_like(ana_mod.attn.lora_A, 0.1),
        "attn.lora_B": torch.full_like(ana_mod.attn.lora_B, -0.2),
        "mil.weight": torch.ones_like(ana_mod.mil.weight),
        "mil.bias": torch.zeros_like(ana_mod.mil.bias),
    }
    c_adapter.load_compact_state(new_compact)
    assert ana_mod.training is False
    ana_mod.train()
    # Frozen backbone weight must remain 100% unpolluted by old LoRA!
    assert torch.allclose(ana_mod.attn.weight, frozen_bb_weight, atol=1e-5)


def test_symic_memmap_rejects_all_silent_and_skips_silent_before_limit(tmp_path: Path) -> None:
    """Regression test for Critic Defect #2: all-silent raises RuntimeError; mixed pool skips silent before `limit`."""
    m_args = MORTM5Args(_tiny_symic_config())
    g = int(m_args.vq_groups)
    l_cnt = int(m_args.vq_layers)
    k_sz = int(m_args.codebook_size)
    tpc = g * l_cnt
    chunks_per_song = 16  # 2 bars
    song_len = chunks_per_song * tpc

    # Case A: 100% constant/silent tokens -> must raise RuntimeError (never fall back to silent windows!).
    silent_dir = tmp_path / "all_silent"
    silent_dir.mkdir()
    np.full((4 * song_len,), 10, dtype=np.uint16).tofile(silent_dir / "train_tokens.bin")
    offsets_a = np.array([(i * song_len, song_len) for i in range(4)], dtype=np.int64)
    slots_a = np.tile(np.array([[128, 0, 32, -1]], dtype=np.int16), (4, 1))
    np.save(silent_dir / "train_offsets.npy", offsets_a)
    np.save(silent_dir / "train_slots.npy", slots_a)

    repo_silent = SyMIcMemmapRepository(
        token_dir=silent_dir, model_args=m_args, split="train", prompt_bars=1, total_bars=2
    )
    with pytest.raises(RuntimeError, match="After filtering silent"):
        repo_silent.load_disjoint_pools(limit=2, seed=0)

    # Case B: Each song has 3 silent windows + 1 active window; `limit=2` must still find active windows!
    mix_dir = tmp_path / "mixed"
    mix_dir.mkdir()
    rng = np.random.default_rng(99)
    parts: List[np.ndarray] = []
    offsets_b: List[Tuple[int, int]] = []
    cur = 0
    for _ in range(4):
        s_win = np.full((3 * song_len,), 10, dtype=np.uint16)
        a_win = rng.integers(3, 3 + g * k_sz, size=(song_len,), dtype=np.uint16)
        full_song = np.concatenate([s_win, a_win])
        parts.append(full_song)
        offsets_b.append((cur, full_song.size))
        cur += full_song.size

    np.concatenate(parts).tofile(mix_dir / "train_tokens.bin")
    np.save(mix_dir / "train_offsets.npy", np.asarray(offsets_b, dtype=np.int64))
    np.save(mix_dir / "train_slots.npy", slots_a)

    repo_mix = SyMIcMemmapRepository(
        token_dir=mix_dir, model_args=m_args, split="train", prompt_bars=1, total_bars=2
    )
    ana_p, gem_p = repo_mix.load_disjoint_pools(limit=2, seed=1)
    assert len(ana_p) == 1 and len(gem_p) == 1


def test_non_finite_grad_and_loss_protection_and_strict_resume_errors(tmp_path: Path) -> None:
    """Regression test for Critic Defects #3, #4, and #5."""
    from mortm.train.rl.optim import MasterWeightAdamW

    # Defect #5: Config boundary validation
    with pytest.raises(ValueError, match="batch_size must be >= 2"):
        UnifiedOddConfig(batch_size=1)
    with pytest.raises(ValueError, match="auc_slope_window must be >= 2"):
        UnifiedOddConfig(auc_slope_window=1)
    with pytest.raises(ValueError, match="disc_span"):
        UnifiedEvenConfig(disc_span=0.0)

    # Defect #4a: MasterWeightAdamW skips Inf/NaN gradients without corrupting bf16 or FP32 master weights
    w = nn.Parameter(torch.full((4, 4), 0.25, dtype=torch.bfloat16))
    orig = w.detach().clone()
    opt = MasterWeightAdamW([w], lr=1e-3)
    w.grad = torch.full_like(w, float("inf"))
    assert opt.step_and_zero_grad(grad_clip=1.0) is False
    assert torch.equal(w, orig)
    w.grad = torch.full_like(w, float("nan"))
    assert opt.step_and_zero_grad(grad_clip=1.0) is False
    assert torch.equal(w, orig)

    # Defect #3 & #4b: Missing resume path raises FileNotFoundError; NaN KL triggers rollback to healthy_snap
    class _NaNActor(CARLActorAdapter):
        def __init__(self) -> None:
            self._net = nn.Linear(4, 4)

        @property
        def actor_module(self) -> nn.Module:
            return self._net

        @property
        def device(self) -> torch.device:
            return torch.device("cpu")

        def unfreeze_for_grpo(self) -> Tuple[int, int]:
            for p in self._net.parameters():
                p.requires_grad_(True)
            return 20, 20

        def rollout_group(
            self,
            examples: Sequence[CARLExample],
            group_size: int,
            sampling: UnifiedSamplingConfig,
            chunk_size: int = 16,
            seed: Optional[int] = None,
        ) -> RolloutBatch:
            exs = tuple(ex for ex in examples for _ in range(group_size))
            gens = tuple(np.array([1, 2, 3, 4], dtype=np.int64) for _ in exs)
            return RolloutBatch(exs, gens, gens, group_size)

        def forward_logprobs_and_exact_kl(
            self,
            base_actor: CARLActorAdapter,
            examples: Sequence[CARLExample],
            generated_sequences: Sequence[np.ndarray],
        ) -> StepLogProbsKL:
            bsz = len(generated_sequences)
            lp = self._net.weight.sum() * torch.zeros((bsz, 4), dtype=torch.float32)
            nan_kl = torch.full((bsz, 4), float("nan"), dtype=torch.float32)
            return StepLogProbsKL(lp, nan_kl, torch.ones((bsz, 4), dtype=torch.bool))

        def export_checkpoint_state(self) -> Dict[str, Any]:
            return {k: v.clone() for k, v in self._net.state_dict().items()}

        def load_checkpoint_state(self, state: Dict[str, Any]) -> None:
            self._net.load_state_dict(state)

    class _SimpleCritic(CARLCriticAdapter):
        def __init__(self) -> None:
            self._p = nn.Linear(4, 1)

        @property
        def critic_module(self) -> nn.Module:
            return self._p

        @property
        def device(self) -> torch.device:
            return torch.device("cpu")

        def mark_trainable(self) -> Dict[str, int]:
            return {"p": 5}

        def score_disc_logits(
            self,
            critic_sequences: Sequence[np.ndarray],
            chunk_size: Optional[int] = None,
            grad_enabled: bool = False,
        ) -> Tensor:
            return torch.zeros((len(critic_sequences),), dtype=torch.float32)

        def export_compact_state(self) -> Dict[str, Any]:
            return {}

        def load_compact_state(self, state: Dict[str, Any]) -> None:
            pass

    class _NoOpBuilder(GuardrailContextBuilder[int]):
        def build_context(self, batch: RolloutBatch) -> int:
            return len(batch)

    class _ZeroRule(GuardrailRule[int]):
        @property
        def name(self) -> str:
            return "zero"

        def evaluate(self, context: int) -> Tuple[Tensor, Dict[str, float]]:
            return torch.zeros((context,), dtype=torch.float32), {}

    actor = _NaNActor()
    base = _NaNActor()
    critic = _SimpleCritic()
    pool = InMemorySamplePool(
        [
            CARLExample("e0", 0, np.array([1], dtype=np.int64), np.array([1, 2], dtype=np.int64)),
            CARLExample("e1", 1, np.array([2], dtype=np.int64), np.array([3, 4], dtype=np.int64)),
        ]
    )
    suite = GuardrailSuite(_NoOpBuilder(), lambda n: torch.zeros((n,), dtype=torch.bool), (_ZeroRule(),))
    drv = UnifiedCARLDriver(
        actor=actor,
        base_actor=base,
        critic=critic,
        ana_pool=pool,
        gem_pool=pool,
        guardrail_suite=suite,
        out_dir=tmp_path / "nan_guard",
        even_cfg=UnifiedEvenConfig(group_size=2, prompts_per_step=1, max_steps=4, train_chunk=2),
        q_max=8,
    )
    with pytest.raises(FileNotFoundError, match="Actor resume checkpoint not found"):
        drv.resume(actor_ckpt=tmp_path / "does_not_exist.pth")
    with pytest.raises(FileNotFoundError, match="Critic resume checkpoint not found"):
        drv.resume(critic_ckpt=tmp_path / "does_not_exist_ana.pth")
    with pytest.raises(FileNotFoundError, match="QTable resume checkpoint not found"):
        drv.resume(qtable_ckpt=tmp_path / "does_not_exist_q.pt")

    drv.qtable.inject_committed([{"seq": np.array([1, 2], dtype=np.int64)}])
    out = drv.run_even(round_no=2)
    assert out["steps"] == 1
    assert str(out["reason"]).startswith("non_finite_abort")
    assert out["rolled_back"] is True
    assert all(torch.isfinite(p).all() for p in actor.actor_module.parameters())



def test_batched_kvcache_sdpa_and_single_group_depth_head_exact_equivalence(capsys: Any) -> None:
    """Verify batched `flash_attn_with_kvcache`, `DecodeCodebookHead(target_group=g)`, and early Odd AUC logs."""
    from mortm.models.modules.attention import flash_attn_with_kvcache
    from mortm.train.rl.engine import OddRoundEngine
    from mortm.train.rl.q_table import QTable

    torch.manual_seed(2026)

    # 1. Uniform & Ragged batched `flash_attn_with_kvcache` equivalence against per-item reference
    B, Sq, H, D = 6, 1, 4, 16
    q = torch.randn(B, Sq, H, D)
    k_new = torch.randn(B, Sq, H, D)
    v_new = torch.randn(B, Sq, H, D)
    k_cache_1 = torch.randn(B, 16, H, D)
    v_cache_1 = torch.randn(B, 16, H, D)
    k_cache_2 = k_cache_1.clone()
    v_cache_2 = v_cache_1.clone()

    # Ragged lengths `[2, 7, 4, 9, 1, 6]`
    ragged_lens = torch.tensor([2, 7, 4, 9, 1, 6], dtype=torch.int32)
    out_batched = flash_attn_with_kvcache(q, k_cache_1, v_cache_1, k=k_new, v=v_new, cache_seqlens=ragged_lens, causal=True)

    ref_outs = []
    for i in range(B):
        p = int(ragged_lens[i].item())
        k_cache_2[i, p : p + 1] = k_new[i]
        v_cache_2[i, p : p + 1] = v_new[i]
        o_i = torch.nn.functional.scaled_dot_product_attention(
            q[i].permute(1, 0, 2).unsqueeze(0),
            k_cache_2[i, : p + 1].permute(1, 0, 2).unsqueeze(0),
            v_cache_2[i, : p + 1].permute(1, 0, 2).unsqueeze(0),
        )
        ref_outs.append(o_i.squeeze(0).permute(1, 0, 2))
    assert torch.allclose(out_batched, torch.stack(ref_outs, dim=0), atol=1e-5)

    # Uniform length `pos = 5` (`uniform_seqlen=5`, zero CPU sync path)
    uni_lens = torch.full((B,), 5, dtype=torch.int32)
    out_uni = flash_attn_with_kvcache(
        q, k_cache_1, v_cache_1, k=k_new, v=v_new, cache_seqlens=uni_lens, causal=True, uniform_seqlen=5
    )
    assert out_uni.shape == (B, Sq, H, D)
    assert torch.isfinite(out_uni).all()

    # 2. `DecodeCodebookHead(target_group=g_i)` equivalence with non-zero trained `q_up` Depth weights
    dev = _select_test_device()
    m_args = MORTM5Args(_tiny_symic_config())
    model = build_mortm5_ablation_model(m_args).to(dev).eval()
    with torch.no_grad():
        for name, p in model.Wout.depth_decoder.named_parameters():
            if name.endswith("q_up.weight"):
                nn.init.normal_(p, std=0.1)
    model.Wout._legacy_untrained_depth.fill_(False)

    h_t = torch.randn(4, int(m_args.d_model), device=dev)
    g_embs = torch.randn(4, 3, int(m_args.d_model), device=dev)
    for s_pos in (1, 132):
        full_logits = model.Wout(h_t, start_pos=s_pos, group_embs=g_embs).float()
        for g_i in range(4):
            single_g = model.Wout(h_t, start_pos=s_pos, group_embs=g_embs, target_group=g_i).float()
            assert torch.allclose(single_g, full_logits[:, g_i, :], atol=2e-2)

    # 3. Verify `OddRoundEngine` prints finite AUC & logits even when `step (2) < eval_window (100)`
    actor = SyMIcGroupedActorAdapter(model, dev, prompt_bars=1, total_bars=2)
    critic = SyMIcCriticAdapter.from_pretrained(m_args, None, dev, prompt_bars=1, lora_rank=4)
    ana_pool = InMemorySamplePool(
        [_make_synthetic_symic_example(m_args, song_id=i, prompt_bars=1, total_bars=2) for i in range(4)]
    )
    odd = OddRoundEngine(
        actor=actor,
        critic=critic,
        qtable=QTable(q_max=8),
        cfg=UnifiedOddConfig(batch_size=4, pool_size=4, max_steps=2, eval_window=100, log_every=2),
    )
    odd.run(ana_pool, round_no=1)
    captured = capsys.readouterr().out
    assert "AUC(gem)=nan" not in captured
    assert "best=-1.0000" not in captured
    assert "+nan/1k" not in captured
