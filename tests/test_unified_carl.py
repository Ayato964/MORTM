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
    np.asarray(offsets, dtype=np.int64). save if False else np.save(
        tmp_path / "train_offsets.npy", np.asarray(offsets, dtype=np.int64)
    )
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
    m_critic_bb = build_mortm5_ablation_model(m_args).to(dev)
    m_critic_bb.load_state_dict(m_actor.state_dict())

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
