from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from mortm.models.modules.config import (
    EOS_TOKEN_ID,
    MORTM5Args,
    SLOT_CODE_EMPTY,
)
from mortm.models.mortm5 import (
    AblationTransformer2,
    MidiVision,
    OctupleTransformer,
    REMITransformer,
    _BaseMORTM5Transformer,
    build_mortm5_ablation_model,
)
from mortm.utils.convert_pianoroll import (
    CHUNKS_PER_BAR,
    CHUNK_STEPS,
    STEPS_PER_BAR,
    _notes_to_midi,
    chunks_to_midi,
    rasterize_notes_to_chunks_float,
    slot_codes_to_roll_meta,
)
from mortm.utils.mortm5_infer import Mortm5PromptEncoder, TOKENIZER_CONFIGS
from mortm.utils.symbolic_tokenizers import extract_segment_canonical_notes

from .domain import ModelCandidateSpec, RolloutTelemetry, SamplingSpec, TestSegmentItem


def synchronize_device(device: torch.device) -> None:
    """Synchronize asynchronous GPU/MPS command queues for accurate latency profiling."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def compute_decoder_kv_bytes_per_step(model: torch.nn.Module) -> int:
    """
    Compute exact active KV-cache bytes per sequence step across all main Transformer
    decoder layers (`2 * num_heads * head_dim * dtype_bytes`).
    """
    layers = getattr(getattr(model, "decoder", None), "layers", ())
    total_bytes_per_step = 0
    for layer in layers:
        attn = layer.self_attention
        n_heads = int(attn.args.num_heads)
        head_dim = int(attn.head_dim)
        elem_bytes = int(attn.qkv_block.qkv_weight.weight.element_size())
        total_bytes_per_step += 2 * n_heads * head_dim * elem_bytes
    return total_bytes_per_step


class ParadigmEvaluationStrategy(ABC):
    """
    Abstract Strategy (OCP / DIP) encapsulating paradigm-specific encoding, controlled
    fixed-horizon hardware latency profiling (§5.1), and 4->16 bar continuation (§5.2).
    """

    def __init__(
        self,
        spec: ModelCandidateSpec,
        model: _BaseMORTM5Transformer,
        device: torch.device,
    ) -> None:
        self._spec = spec
        self._model = model
        self._device = device
        self._kv_bytes_per_step = compute_decoder_kv_bytes_per_step(model)

    @property
    def spec(self) -> ModelCandidateSpec:
        return self._spec

    @property
    def model(self) -> _BaseMORTM5Transformer:
        return self._model

    @property
    def device(self) -> torch.device:
        return self._device

    def active_kv_mib(self, seq_steps: int) -> float:
        return (float(max(0, int(seq_steps))) * float(self._kv_bytes_per_step)) / float(1024 * 1024)

    @abstractmethod
    def count_16bar_ar_steps_and_bar_checkpoints(
        self,
        segment: TestSegmentItem,
    ) -> Tuple[int, Tuple[Tuple[int, float], ...]]:
        """
        Return `(total_ar_steps_S, ((bar_idx, kv_mib), ...))` for a ground-truth 16-bar segment
        at bars `(4, 8, 12, 16)`.
        """

    @abstractmethod
    def profile_ground_truth_horizon_rollout(
        self,
        segment: TestSegmentItem,
        sampling: SamplingSpec,
        prompt_chunks: int = 32,
    ) -> RolloutTelemetry:
        """
        Measure hardware inference latency (`ms/bar`, `bars/sec`) and active KV-cache MiB
        when decoding from the 4-bar prompt for the exact step budget matching `segment`'s
        16-bar density (§5.1).
        """

    @abstractmethod
    def generate_continuation_to_midi(
        self,
        segment: TestSegmentItem,
        output_16bar_midi: Path,
        sampling: SamplingSpec,
        prompt_chunks: int = 32,
        seed: Optional[int] = None,
    ) -> RolloutTelemetry:
        """
        Execute 4->16 bar prompted continuation under `sampling`, export the full 16-bar
        quantized multi-track MIDI to `output_16bar_midi`, and return execution telemetry (§5.2).
        """


class SyMIcGroupedStrategy(ParadigmEvaluationStrategy):
    """Strategy for `SyMIc-G4L4` (`AblationTransformer2` + `MidiVision` RVQ + Depth Head)."""

    def __init__(
        self,
        spec: ModelCandidateSpec,
        model: AblationTransformer2,
        vq_model: MidiVision,
        device: torch.device,
    ) -> None:
        super().__init__(spec=spec, model=model, device=device)
        self._typed_model = model
        self._vq_model = vq_model

    def count_16bar_ar_steps_and_bar_checkpoints(
        self,
        segment: TestSegmentItem,
    ) -> Tuple[int, Tuple[Tuple[int, float], ...]]:
        prefix_len = 1 + (4 if self._typed_model.use_instrument_tokens else 0)
        layers = int(self._typed_model.vq_layers)
        traj: List[Tuple[int, float]] = []
        for bar_mark in (4, 8, 12, 16):
            chunks_at_bar = min(segment.num_chunks, bar_mark * CHUNKS_PER_BAR)
            steps_at_bar = prefix_len + chunks_at_bar * layers + (1 if bar_mark == 16 else 0)
            traj.append((bar_mark, self.active_kv_mib(steps_at_bar - 1)))
        total_s = prefix_len + segment.num_chunks * layers + 1
        return total_s, tuple(traj)

    @torch.inference_mode()
    def _run_rvq_rollout(
        self,
        prompt_notes_5col: np.ndarray,
        slot_codes: Sequence[int],
        prompt_chunks: int,
        total_chunks: int,
        sampling: SamplingSpec,
        include_codec_in_timing: bool = True,
    ) -> Tuple[np.ndarray, RolloutTelemetry]:
        model = self._typed_model
        vq = self._vq_model
        dev = self._device
        L = int(model.vq_layers)
        G = int(model.vq_groups)
        new_chunks = total_chunks - prompt_chunks
        gen_bars = float(new_chunks) / float(CHUNKS_PER_BAR)

        prompt_f32 = rasterize_notes_to_chunks_float(prompt_chunks, prompt_notes_5col)
        model.eval()
        model.reset_kv_cache()

        synchronize_device(dev)
        t0 = time.perf_counter()

        prompt_seq = Mortm5PromptEncoder.encode_chunks_to_prompt(
            prompt_f32,
            vq_model=vq,
            model=model,
            slot_codes=slot_codes,
            device=dev,
        )
        prompt_steps = int(prompt_seq.size(1))
        target_soft_len = prompt_steps + new_chunks * L

        h_t = model._forward_backbone_last_hidden(prompt_seq, is_prefill=True)
        synchronize_device(dev)
        t_prefill = time.perf_counter()

        cur_soft_pos = prompt_steps
        generated: List[Tensor] = [prompt_seq]
        use_depth_ar = hasattr(model.Wout, "_legacy_untrained_depth") and not bool(
            model.Wout._legacy_untrained_depth
        )
        layer_stride = G * int(model.codebook_size)
        prefix_soft_len = 1 + (4 if model.use_instrument_tokens else 0)

        while cur_soft_pos < target_soft_len:
            step_pos = cur_soft_pos - 1
            cur_layer_idx = max(0, cur_soft_pos - prefix_soft_len) % L
            step_temp = sampling.temperature if cur_layer_idx == 0 else sampling.fine_temperature
            step_top_k = sampling.top_k if cur_layer_idx == 0 else sampling.fine_top_k
            step_top_p = sampling.top_p if cur_layer_idx == 0 else sampling.fine_top_p

            if use_depth_ar:
                cur_group_embs = torch.zeros(
                    (1, G - 1, model.args.d_model),
                    device=dev,
                    dtype=model.Wout.weight.dtype,
                )
                sampled_locals: List[Tensor] = []
                eff_layer = cur_layer_idx % model.effective_layers
                for g_i in range(G):
                    with torch.autocast(device_type=dev.type, enabled=False):
                        depth_logits = model.Wout(
                            h_t.to(model.Wout.weight.dtype),
                            start_pos=step_pos,
                            group_embs=cur_group_embs,
                        ).float()
                    constrained = model._apply_grouped_grammar_mask(
                        depth_logits, soft_pos=cur_soft_pos
                    )
                    tok_g = model._sample_from_logits(
                        constrained[:, g_i, :],
                        temperature=step_temp,
                        top_k=step_top_k,
                        top_p=step_top_p,
                        min_p=sampling.min_p,
                    )
                    sampled_locals.append(tok_g)
                    if g_i < G - 1:
                        local_k_g = (tok_g - model.token_offset).clamp(min=0) % model.codebook_size
                        global_id_g = (
                            model.token_offset
                            + eff_layer * layer_stride
                            + g_i * model.codebook_size
                            + local_k_g
                        )
                        cur_group_embs[:, g_i, :] = model.embedding(global_id_g).to(
                            dtype=cur_group_embs.dtype
                        )
                local_sampled = torch.stack(sampled_locals, dim=-1)
            else:
                with torch.autocast(device_type=dev.type, enabled=False):
                    next_logits = model.Wout(
                        h_t.to(model.Wout.weight.dtype), start_pos=step_pos
                    ).float()
                constrained = model._apply_grouped_grammar_mask(next_logits, soft_pos=cur_soft_pos)
                local_sampled = model._sample_from_logits(
                    constrained,
                    temperature=step_temp,
                    top_k=step_top_k,
                    top_p=step_top_p,
                    min_p=sampling.min_p,
                )

            next_global = model.local_to_global_ids(local_sampled, layer_idx=cur_layer_idx).unsqueeze(1)
            generated.append(next_global)
            cur_soft_pos += 1
            if cur_soft_pos < target_soft_len:
                h_t = model._forward_backbone_last_hidden(next_global, is_prefill=False)

        full_seq = torch.cat(generated, dim=1)
        rvq_all = model.extract_rvq_tokens(full_seq)
        tokens_per_chunk = L * G
        new_rvq = rvq_all[prompt_chunks * tokens_per_chunk :].view(new_chunks, L, G)

        if include_codec_in_timing:
            logits_dec = vq.decode_tokens(new_rvq.to(dev))
            new_probs = torch.sigmoid(logits_dec).detach().cpu().numpy()
            synchronize_device(dev)
            t_end = time.perf_counter()
        else:
            synchronize_device(dev)
            t_end = time.perf_counter()
            logits_dec = vq.decode_tokens(new_rvq.to(dev))
            new_probs = torch.sigmoid(logits_dec).detach().cpu().numpy()

        model.reset_kv_cache()

        for slot_idx, code in enumerate(slot_codes):
            if int(code) == SLOT_CODE_EMPTY:
                new_probs[:, slot_idx * 2 : slot_idx * 2 + 2] = 0.0

        full_probs = np.concatenate([prompt_f32, new_probs], axis=0)
        gen_steps = target_soft_len - prompt_steps
        total_steps = target_soft_len + 1  # +1 for EOS
        traj = tuple(
            (b, self.active_kv_mib(prefix_soft_len + b * CHUNKS_PER_BAR * L))
            for b in (4, 8, 12, 16)
        )
        telemetry = RolloutTelemetry(
            prompt_steps=prompt_steps,
            generated_steps=gen_steps,
            total_steps=total_steps,
            prefill_ms=(t_prefill - t0) * 1000.0,
            decode_ms=(t_end - t_prefill) * 1000.0,
            total_ms=(t_end - t0) * 1000.0,
            generated_bars=gen_bars,
            peak_active_kv_mib=self.active_kv_mib(target_soft_len),
            kv_bar_trajectory_mib=traj,
        )
        return full_probs, telemetry

    def profile_ground_truth_horizon_rollout(
        self,
        segment: TestSegmentItem,
        sampling: SamplingSpec,
        prompt_chunks: int = 32,
    ) -> RolloutTelemetry:
        prompt_notes = segment.prompt_5col_notes(prompt_chunks=prompt_chunks)
        _, telemetry = self._run_rvq_rollout(
            prompt_notes_5col=prompt_notes,
            slot_codes=segment.slot_codes,
            prompt_chunks=prompt_chunks,
            total_chunks=segment.num_chunks,
            sampling=sampling,
            include_codec_in_timing=True,
        )
        return telemetry

    def generate_continuation_to_midi(
        self,
        segment: TestSegmentItem,
        output_16bar_midi: Path,
        sampling: SamplingSpec,
        prompt_chunks: int = 32,
        seed: Optional[int] = None,
    ) -> RolloutTelemetry:
        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed) % (2**32 - 1))
        prompt_notes = segment.prompt_5col_notes(prompt_chunks=prompt_chunks)
        full_probs, telemetry = self._run_rvq_rollout(
            prompt_notes_5col=prompt_notes,
            slot_codes=segment.slot_codes,
            prompt_chunks=prompt_chunks,
            total_chunks=segment.num_chunks,
            sampling=sampling,
            include_codec_in_timing=True,
        )
        output_16bar_midi.parent.mkdir(parents=True, exist_ok=True)
        meta = slot_codes_to_roll_meta(segment.slot_codes)
        chunks_to_midi(full_probs, meta, output_16bar_midi)
        return telemetry


class OctupleCanonicalStrategy(ParadigmEvaluationStrategy):
    """
    Strategy for `Octuple-MMT` (`OctupleTransformer`) using exact Bayesian joint `(bar, pos)` +
    `(slot, pitch)` sampling with strict canonical `(bar, pos, slot, pitch)` monotonicity and
    active-slot masking (§5.2.B).
    """

    def __init__(
        self,
        spec: ModelCandidateSpec,
        model: OctupleTransformer,
        device: torch.device,
    ) -> None:
        super().__init__(spec=spec, model=model, device=device)
        self._typed_model = model
        self._tok = model.get_symbolic_tokenizer()
        self._dur_lut = self._tok.grid_spec.build_dur_lut()
        vel_bins = int(self._tok.grid_spec.velocity_bins)
        self._vel_lut = self._tok.grid_spec.dequantize_vel_np(np.arange(vel_bins, dtype=np.int32))

    def count_16bar_ar_steps_and_bar_checkpoints(
        self,
        segment: TestSegmentItem,
    ) -> Tuple[int, Tuple[Tuple[int, float], ...]]:
        prefix_len = 1 + (4 if self._typed_model.use_instrument_tokens else 0)
        canon = segment.canonical_notes_view()
        bars = canon[:, 0]
        traj: List[Tuple[int, float]] = []
        for bar_mark in (4, 8, 12, 16):
            notes_up_to = int(np.sum(bars < bar_mark))
            steps_at_bar = prefix_len + notes_up_to + (1 if bar_mark == 16 else 0)
            traj.append((bar_mark, self.active_kv_mib(steps_at_bar - 1)))
        total_s = prefix_len + int(len(canon)) + 1
        return total_s, tuple(traj)

    @torch.inference_mode()
    def profile_ground_truth_horizon_rollout(
        self,
        segment: TestSegmentItem,
        sampling: SamplingSpec,
        prompt_chunks: int = 32,
    ) -> RolloutTelemetry:
        """
        Measure exact hardware latency and KV trajectory for `Octuple-MMT` when decoding
        the exact number of note-tuple steps in `segment`'s 12-bar continuation.
        """
        model = self._typed_model
        dev = self._device
        canon = segment.canonical_notes_view()
        prompt_bar_limit = prompt_chunks // CHUNKS_PER_BAR
        prompt_canon = extract_segment_canonical_notes(
            segment.prompt_5col_notes(prompt_chunks=prompt_chunks),
            chunk_start=0,
            chunk_end=prompt_chunks,
            max_bars=segment.num_bars,
        )
        cont_note_budget = max(1, int(np.sum(canon[:, 0] >= prompt_bar_limit)))

        prefix_toks = model.build_prefix_tokens(
            slot_codes=segment.slot_codes, batch_size=1, device=dev
        )
        body_u16 = self._tok.encode_canonical_segment_notes(prompt_canon, num_chunks=prompt_chunks)
        if len(body_u16) > 0:
            body_t = (
                torch.from_numpy(body_u16.astype(np.int64))
                .to(device=dev)
                .reshape(1, -1, model.tuple_size)
            )
            prompt_seq = torch.cat([prefix_toks, body_t], dim=1)
        else:
            prompt_seq = prefix_toks

        prompt_steps = int(prompt_seq.size(1))
        cont_canon = canon[canon[:, 0] >= prompt_bar_limit]
        cont_u16 = self._tok.encode_canonical_segment_notes(cont_canon, num_chunks=segment.num_chunks)
        cont_tuples = (
            torch.from_numpy(cont_u16.astype(np.int64))
            .to(device=dev)
            .reshape(1, -1, model.tuple_size)
            if len(cont_u16) > 0
            else prefix_toks[:, :1, :]
        )

        model.eval()
        model.reset_kv_cache()
        synchronize_device(dev)
        t0 = time.perf_counter()

        logits = model.forward(prompt_seq, is_causal=True, is_save_cache=True)
        _ = logits[:, -1, :, :]
        synchronize_device(dev)
        t_prefill = time.perf_counter()

        for idx in range(cont_note_budget):
            step_logits = model.forward(
                cont_tuples[:, idx : idx + 1, :], is_causal=True, is_save_cache=True
            )
            _ = _BaseMORTM5Transformer._sample_from_logits(
                step_logits[:, -1, :, :].float(),
                temperature=sampling.temperature,
                top_k=sampling.top_k,
                top_p=sampling.top_p,
                min_p=sampling.min_p,
            )

        synchronize_device(dev)
        t_end = time.perf_counter()
        model.reset_kv_cache()

        _, traj = self.count_16bar_ar_steps_and_bar_checkpoints(segment)
        total_steps = prompt_steps + cont_note_budget + 1
        gen_bars = float(segment.num_chunks - prompt_chunks) / float(CHUNKS_PER_BAR)
        return RolloutTelemetry(
            prompt_steps=prompt_steps,
            generated_steps=cont_note_budget,
            total_steps=total_steps,
            prefill_ms=(t_prefill - t0) * 1000.0,
            decode_ms=(t_end - t_prefill) * 1000.0,
            total_ms=(t_end - t0) * 1000.0,
            generated_bars=gen_bars,
            peak_active_kv_mib=self.active_kv_mib(prompt_steps + cont_note_budget),
            kv_bar_trajectory_mib=traj,
        )

    @torch.inference_mode()
    def generate_continuation_to_midi(
        self,
        segment: TestSegmentItem,
        output_16bar_midi: Path,
        sampling: SamplingSpec,
        prompt_chunks: int = 32,
        seed: Optional[int] = None,
    ) -> RolloutTelemetry:
        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed) % (2**32 - 1))

        model = self._typed_model
        dev = self._device
        b_size, p_size, s_size, k_size, d_size, v_size = (int(x) for x in model.attr_sizes)
        tok_off = int(model.token_offset)
        attr_global_offsets = tuple(int(off) for off in self._tok.vocab_spec.attr_offsets)
        full_horizon_steps = segment.num_chunks * CHUNK_STEPS
        last_bar_start_step = max(0, (segment.num_bars - 1) * p_size)
        min_local_step = prompt_chunks * CHUNK_STEPS

        prompt_notes = segment.prompt_5col_notes(prompt_chunks=prompt_chunks)
        global_notes: List[List[int]] = [
            [int(r[0]), int(r[1]), int(r[2]), int(r[3]), int(r[4])] for r in prompt_notes
        ]
        last_note_by_sp: Dict[Tuple[int, int], List[int]] = {
            (n[0], n[3]): n for n in global_notes
        }

        seg_canonical = extract_segment_canonical_notes(
            prompt_notes,
            chunk_start=0,
            chunk_end=prompt_chunks,
            max_bars=segment.num_bars,
        )
        body_flat_u16 = self._tok.encode_canonical_segment_notes(
            seg_canonical, num_chunks=prompt_chunks
        )

        raw_active = [
            idx < len(segment.slot_codes) and int(segment.slot_codes[idx]) != SLOT_CODE_EMPTY
            for idx in range(s_size)
        ]
        active_flags = raw_active if any(raw_active) else [True] * s_size
        active_slot_mask_cpu = torch.tensor(active_flags, dtype=torch.bool, device="cpu")
        sp_arange_cpu = torch.arange(s_size * k_size, dtype=torch.long, device="cpu")

        model.eval()
        model.reset_kv_cache()
        synchronize_device(dev)
        t0 = time.perf_counter()

        prefix_toks = model.build_prefix_tokens(
            slot_codes=segment.slot_codes, batch_size=1, device=dev
        )
        if len(body_flat_u16) > 0:
            body_tuples = (
                torch.from_numpy(body_flat_u16.astype(np.int64))
                .to(device=dev)
                .reshape(1, -1, model.tuple_size)
            )
            prompt_seq = torch.cat([prefix_toks, body_tuples], dim=1)
        else:
            prompt_seq = prefix_toks

        prompt_steps = int(prompt_seq.size(1))
        logits = model.forward(prompt_seq, is_causal=True, is_save_cache=True)
        next_logits_cpu = logits[:, -1, :, :].detach().to(device="cpu", dtype=torch.float32)
        synchronize_device(dev)
        t_prefill = time.perf_counter()

        prev_step = min_local_step
        prev_sp = -1
        max_gen_notes = 3072
        gen_steps = 0
        bar_step_counts = {4: prompt_steps, 8: prompt_steps, 12: prompt_steps, 16: prompt_steps}

        for note_idx in range(max_gen_notes):
            row_logits = _BaseMORTM5Transformer._sanitize_logits_f32(
                next_logits_cpu[0], posinf_value=1.0e30
            )
            raw_bar = row_logits[0, tok_off : tok_off + b_size]
            raw_pos = row_logits[1, tok_off : tok_off + p_size]
            raw_slot = row_logits[2, tok_off : tok_off + s_size]
            raw_pitch = row_logits[3, tok_off : tok_off + k_size]
            raw_dur = row_logits[4, tok_off : tok_off + d_size]
            raw_vel = row_logits[5, tok_off : tok_off + v_size]
            raw_eos = row_logits[0, EOS_TOKEN_ID]

            slot_logits = raw_slot.masked_fill(~active_slot_mask_cpu, float("-inf"))
            slot_lp = torch.log_softmax(slot_logits, dim=-1)
            pitch_lp = torch.log_softmax(raw_pitch, dim=-1)
            joint_sp_lp = (slot_lp[:, None] + pitch_lp[None, :]).reshape(-1)

            same_step_sp_lp = joint_sp_lp.masked_fill(sp_arange_cpu <= prev_sp, float("-inf"))
            if bool(torch.isfinite(same_step_sp_lp).any().item()):
                same_step_log_mass = torch.logsumexp(same_step_sp_lp, dim=-1)
            else:
                same_step_log_mass = joint_sp_lp.new_tensor(float("-inf"))

            head0_lp = torch.log_softmax(torch.cat([raw_bar, raw_eos.view(1)], dim=0), dim=-1)
            bar_lp = head0_lp[:b_size]
            eos_lp = head0_lp[b_size]
            pos_lp = torch.log_softmax(raw_pos, dim=-1)

            joint_time_lp = (bar_lp[:, None] + pos_lp[None, :]).reshape(-1).clone()
            if prev_step > 0:
                joint_time_lp[:prev_step] = float("-inf")
            if 0 <= prev_step < full_horizon_steps:
                joint_time_lp[prev_step] = joint_time_lp[prev_step] + same_step_log_mass
            else:
                joint_time_lp.fill_(float("-inf"))

            allow_eos_now = (prev_step >= last_bar_start_step) and (note_idx >= 1)
            eff_eos_lp = (
                eos_lp.view(1)
                if allow_eos_now
                else joint_time_lp.new_full((1,), float("-inf"))
            )
            choice_logits = torch.cat([joint_time_lp, eff_eos_lp], dim=0)
            if not bool(torch.isfinite(choice_logits).any().item()):
                break

            t_idx = int(
                _BaseMORTM5Transformer._sample_from_logits(
                    choice_logits.unsqueeze(0),
                    temperature=sampling.temperature,
                    top_k=sampling.top_k,
                    top_p=sampling.top_p,
                    min_p=sampling.min_p,
                )[0].item()
            )
            gen_steps += 1
            if t_idx >= full_horizon_steps:
                break

            cond_sp_lp = same_step_sp_lp if (t_idx == prev_step) else joint_sp_lp
            if not bool(torch.isfinite(cond_sp_lp).any().item()):
                break

            sp_idx = int(
                _BaseMORTM5Transformer._sample_from_logits(
                    cond_sp_lp.unsqueeze(0),
                    temperature=sampling.temperature,
                    top_k=sampling.top_k,
                    top_p=sampling.top_p,
                    min_p=sampling.min_p,
                )[0].item()
            )
            dur_bin = int(
                _BaseMORTM5Transformer._sample_from_logits(
                    raw_dur.unsqueeze(0),
                    temperature=sampling.temperature,
                    top_k=sampling.top_k,
                    top_p=sampling.top_p,
                    min_p=sampling.min_p,
                )[0].item()
            )
            vel_bin = int(
                _BaseMORTM5Transformer._sample_from_logits(
                    raw_vel.unsqueeze(0),
                    temperature=sampling.temperature,
                    top_k=sampling.top_k,
                    top_p=sampling.top_p,
                    min_p=sampling.min_p,
                )[0].item()
            )

            sampled_bar = t_idx // p_size
            sampled_pos = t_idx % p_size
            sampled_slot = sp_idx // k_size
            sampled_pitch = sp_idx % k_size
            dur_steps = max(1, int(self._dur_lut[min(max(0, dur_bin), len(self._dur_lut) - 1)]))
            velocity = int(self._vel_lut[min(max(0, vel_bin), len(self._vel_lut) - 1)])

            global_start = t_idx
            global_end = min(full_horizon_steps, global_start + dur_steps)
            sp_key = (sampled_slot, sampled_pitch)
            prev_same = last_note_by_sp.get(sp_key)
            if prev_same is not None and prev_same[2] > global_start:
                prev_same[2] = max(prev_same[1] + 1, global_start)

            if global_end > global_start:
                new_note = [sampled_slot, global_start, global_end, sampled_pitch, velocity]
                global_notes.append(new_note)
                last_note_by_sp[sp_key] = new_note

            prev_step = t_idx
            prev_sp = sp_idx
            cur_total_steps = prompt_steps + gen_steps
            for bar_mark in (8, 12, 16):
                if sampled_bar < bar_mark:
                    bar_step_counts[bar_mark] = cur_total_steps

            if note_idx + 1 < max_gen_notes:
                global_tuple = [
                    attr_global_offsets[0] + sampled_bar,
                    attr_global_offsets[1] + sampled_pos,
                    attr_global_offsets[2] + sampled_slot,
                    attr_global_offsets[3] + sampled_pitch,
                    attr_global_offsets[4] + dur_bin,
                    attr_global_offsets[5] + vel_bin,
                ]
                next_global_dev = torch.tensor([[global_tuple]], dtype=torch.long, device=dev)
                step_logits = model.forward(next_global_dev, is_causal=True, is_save_cache=True)
                next_logits_cpu = (
                    step_logits[:, -1, :, :].detach().to(device="cpu", dtype=torch.float32)
                )

        synchronize_device(dev)
        t_end = time.perf_counter()
        model.reset_kv_cache()

        bar_step_counts[16] = prompt_steps + gen_steps
        bar_step_counts[12] = max(bar_step_counts[8], bar_step_counts[12])
        traj = tuple((b, self.active_kv_mib(bar_step_counts[b])) for b in (4, 8, 12, 16))

        output_16bar_midi.parent.mkdir(parents=True, exist_ok=True)
        meta = slot_codes_to_roll_meta(segment.slot_codes)
        sorted_tuples = sorted(
            (int(n[0]), int(n[1]), int(n[2]), int(n[3]), int(n[4]))
            for n in global_notes
            if int(n[2]) > int(n[1])
        )
        _notes_to_midi(sorted_tuples, meta, output_16bar_midi)

        gen_bars = float(segment.num_chunks - prompt_chunks) / float(CHUNKS_PER_BAR)
        return RolloutTelemetry(
            prompt_steps=prompt_steps,
            generated_steps=gen_steps,
            total_steps=prompt_steps + gen_steps,
            prefill_ms=(t_prefill - t0) * 1000.0,
            decode_ms=(t_end - t_prefill) * 1000.0,
            total_ms=(t_end - t0) * 1000.0,
            generated_bars=gen_bars,
            peak_active_kv_mib=self.active_kv_mib(prompt_steps + gen_steps),
            kv_bar_trajectory_mib=traj,
        )


class RemiFsmCanonicalStrategy(ParadigmEvaluationStrategy):
    """
    Strategy for `Multitrack-REMI` (`REMITransformer`) enforcing FSM grammar transitions,
    active-slot masks, and strict canonical `(slot, pitch)` chord monotonicity (§5.2.B).
    """

    def __init__(
        self,
        spec: ModelCandidateSpec,
        model: REMITransformer,
        device: torch.device,
    ) -> None:
        super().__init__(spec=spec, model=model, device=device)
        self._typed_model = model
        self._tok = model.get_symbolic_tokenizer()
        self._dur_lut = self._tok.grid_spec.build_dur_lut()
        vel_bins = int(self._tok.grid_spec.velocity_bins)
        self._vel_lut = self._tok.grid_spec.dequantize_vel_np(np.arange(vel_bins, dtype=np.int32))

    def count_16bar_ar_steps_and_bar_checkpoints(
        self,
        segment: TestSegmentItem,
    ) -> Tuple[int, Tuple[Tuple[int, float], ...]]:
        prefix_len = 1 + (4 if self._typed_model.use_instrument_tokens else 0)
        canon = segment.canonical_notes_view()
        traj: List[Tuple[int, float]] = []
        for bar_mark in (4, 8, 12, 16):
            sub_canon = canon[canon[:, 0] < bar_mark]
            body = self._tok.encode_canonical_segment_notes(
                sub_canon, num_chunks=bar_mark * CHUNKS_PER_BAR
            )
            steps_at_bar = prefix_len + int(len(body)) + (1 if bar_mark == 16 else 0)
            traj.append((bar_mark, self.active_kv_mib(steps_at_bar - 1)))
        full_body = self._tok.encode_canonical_segment_notes(canon, num_chunks=segment.num_chunks)
        total_s = prefix_len + int(len(full_body)) + 1
        return total_s, tuple(traj)

    @torch.inference_mode()
    def profile_ground_truth_horizon_rollout(
        self,
        segment: TestSegmentItem,
        sampling: SamplingSpec,
        prompt_chunks: int = 32,
    ) -> RolloutTelemetry:
        model = self._typed_model
        dev = self._device
        canon = segment.canonical_notes_view()
        prompt_canon = extract_segment_canonical_notes(
            segment.prompt_5col_notes(prompt_chunks=prompt_chunks),
            chunk_start=0,
            chunk_end=prompt_chunks,
            max_bars=segment.num_bars,
        )
        prompt_u16 = self._tok.encode_canonical_segment_notes(prompt_canon, num_chunks=prompt_chunks)
        full_u16 = self._tok.encode_canonical_segment_notes(canon, num_chunks=segment.num_chunks)
        cont_u16 = (
            full_u16[len(prompt_u16) :]
            if len(full_u16) > len(prompt_u16)
            else np.asarray([model.bar_token_id], dtype=np.uint16)
        )
        cont_budget = max(1, int(len(cont_u16)))

        prefix_toks = model.build_prefix_tokens(
            slot_codes=segment.slot_codes, batch_size=1, device=dev
        )
        if len(prompt_u16) > 0:
            body_t = torch.from_numpy(prompt_u16.astype(np.int64)).to(device=dev).reshape(1, -1)
            prompt_seq = torch.cat([prefix_toks, body_t], dim=1)
        else:
            prompt_seq = prefix_toks

        prompt_steps = int(prompt_seq.size(1))
        cont_toks_dev = torch.from_numpy(cont_u16.astype(np.int64)).to(device=dev)

        model.eval()
        model.reset_kv_cache()
        synchronize_device(dev)
        t0 = time.perf_counter()

        prefill_mask = torch.ones_like(prompt_seq, dtype=torch.bool)
        logits = model.forward(
            prompt_seq, padding_mask=prefill_mask, is_causal=True, is_save_cache=True
        )
        _ = logits[:, -1, :]
        synchronize_device(dev)
        t_prefill = time.perf_counter()

        for idx in range(cont_budget):
            step_logits = model._forward_decode_step(cont_toks_dev[idx : idx + 1])
            _ = _BaseMORTM5Transformer._sample_from_logits(
                step_logits.float(),
                temperature=sampling.temperature,
                top_k=sampling.top_k,
                top_p=sampling.top_p,
                min_p=sampling.min_p,
            )

        synchronize_device(dev)
        t_end = time.perf_counter()
        model.reset_kv_cache()

        _, traj = self.count_16bar_ar_steps_and_bar_checkpoints(segment)
        gen_bars = float(segment.num_chunks - prompt_chunks) / float(CHUNKS_PER_BAR)
        return RolloutTelemetry(
            prompt_steps=prompt_steps,
            generated_steps=cont_budget,
            total_steps=prompt_steps + cont_budget + 1,
            prefill_ms=(t_prefill - t0) * 1000.0,
            decode_ms=(t_end - t_prefill) * 1000.0,
            total_ms=(t_end - t0) * 1000.0,
            generated_bars=gen_bars,
            peak_active_kv_mib=self.active_kv_mib(prompt_steps + cont_budget),
            kv_bar_trajectory_mib=traj,
        )

    @staticmethod
    def _sample_next_fsm_token(
        model: REMITransformer,
        raw_logits_row: Tensor,
        fsm_state: int,
        cur_bar: int,
        last_pos: int,
        pending_slot: int,
        prev_sp_in_pos: int,
        notes_gen: int,
        floor_bar: int,
        floor_pos: int,
        bar_limit: int,
        last_bar_pos_limit: int,
        min_bars_idx: int,
        active_slot_flags: Sequence[bool],
        has_any_active_slot: bool,
        sampling: SamplingSpec,
    ) -> Tuple[int, bool]:
        raw_row = _BaseMORTM5Transformer._sanitize_logits_f32(raw_logits_row, posinf_value=1.0e30)
        constrained = raw_row.new_full((raw_row.numel(),), float("-inf"))
        pos_size = int(model.pos_size)
        slot_size = int(model.slot_size)
        pitch_size = int(model.pitch_size)
        has_valid = False

        if fsm_state == model._STATE_BEFORE_BAR:
            if cur_bar + 1 < bar_limit:
                constrained[model.bar_token_id] = raw_row[model.bar_token_id]
                has_valid = True

        elif fsm_state == model._STATE_AFTER_BAR:
            if cur_bar + 1 < bar_limit:
                constrained[model.bar_token_id] = raw_row[model.bar_token_id]
                has_valid = True
            if floor_bar <= cur_bar < bar_limit:
                p_end = last_bar_pos_limit if cur_bar == (bar_limit - 1) else pos_size
                p_start = floor_pos if cur_bar == floor_bar else 0
                if p_start < p_end:
                    constrained[model.pos_offset + p_start : model.pos_offset + p_end] = raw_row[
                        model.pos_offset + p_start : model.pos_offset + p_end
                    ]
                    has_valid = True
            if (
                cur_bar >= floor_bar
                and cur_bar >= min_bars_idx
                and (notes_gen >= 1 or not has_any_active_slot)
            ):
                constrained[EOS_TOKEN_ID] = raw_row[EOS_TOKEN_ID]
                has_valid = True

        elif fsm_state == model._STATE_AFTER_POS:
            for s_idx in range(slot_size):
                if active_slot_flags[s_idx] and (s_idx * pitch_size + (pitch_size - 1)) > prev_sp_in_pos:
                    tok_i = model.slot_offset + s_idx
                    constrained[tok_i] = raw_row[tok_i]
                    has_valid = True

        elif fsm_state == model._STATE_AFTER_SLOT:
            min_pitch = (
                (prev_sp_in_pos % pitch_size) + 1
                if (prev_sp_in_pos >= 0 and pending_slot == (prev_sp_in_pos // pitch_size))
                else 0
            )
            if min_pitch < pitch_size:
                pitch_slice = raw_row[model.pitch_offset : model.pitch_offset + pitch_size].clone()
                if min_pitch > 0:
                    pitch_slice[:min_pitch] = float("-inf")
                constrained[model.pitch_offset : model.pitch_offset + pitch_size] = pitch_slice
                has_valid = True

        elif fsm_state == model._STATE_AFTER_PITCH:
            constrained[model.dur_offset : model.dur_offset + model.dur_size] = raw_row[
                model.dur_offset : model.dur_offset + model.dur_size
            ]
            has_valid = True

        elif fsm_state == model._STATE_AFTER_DUR:
            constrained[model.vel_offset : model.vel_offset + model.vel_size] = raw_row[
                model.vel_offset : model.vel_offset + model.vel_size
            ]
            has_valid = True

        elif fsm_state == model._STATE_AFTER_VEL:
            p_end = last_bar_pos_limit if cur_bar == (bar_limit - 1) else pos_size
            if (
                0 <= cur_bar < bar_limit
                and 0 <= last_pos < p_end
                and (cur_bar > floor_bar or (cur_bar == floor_bar and last_pos >= floor_pos))
            ):
                for s_idx in range(slot_size):
                    if (
                        active_slot_flags[s_idx]
                        and (s_idx * pitch_size + (pitch_size - 1)) > prev_sp_in_pos
                    ):
                        tok_i = model.slot_offset + s_idx
                        constrained[tok_i] = raw_row[tok_i]
                        has_valid = True
            if floor_bar <= cur_bar < bar_limit:
                min_p_bar = floor_pos if cur_bar == floor_bar else 0
                p_start = max(last_pos + 1, min_p_bar)
                if p_start < p_end:
                    constrained[model.pos_offset + p_start : model.pos_offset + p_end] = raw_row[
                        model.pos_offset + p_start : model.pos_offset + p_end
                    ]
                    has_valid = True
            if cur_bar + 1 < bar_limit:
                constrained[model.bar_token_id] = raw_row[model.bar_token_id]
                has_valid = True
            if (
                cur_bar >= floor_bar
                and cur_bar >= min_bars_idx
                and (notes_gen >= 1 or not has_any_active_slot)
            ):
                constrained[EOS_TOKEN_ID] = raw_row[EOS_TOKEN_ID]
                has_valid = True

        if not has_valid or not bool(torch.isfinite(constrained).any().item()):
            return EOS_TOKEN_ID, True

        tok = int(
            _BaseMORTM5Transformer._sample_from_logits(
                constrained.unsqueeze(0),
                temperature=sampling.temperature,
                top_k=sampling.top_k,
                top_p=sampling.top_p,
                min_p=sampling.min_p,
            )[0].item()
        )
        return tok, (tok == EOS_TOKEN_ID)

    @torch.inference_mode()
    def generate_continuation_to_midi(
        self,
        segment: TestSegmentItem,
        output_16bar_midi: Path,
        sampling: SamplingSpec,
        prompt_chunks: int = 32,
        seed: Optional[int] = None,
    ) -> RolloutTelemetry:
        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed) % (2**32 - 1))

        model = self._typed_model
        dev = self._device
        pos_size = int(model.pos_size)
        slot_size = int(model.slot_size)
        pitch_size = int(model.pitch_size)
        full_horizon_steps = segment.num_chunks * CHUNK_STEPS
        max_valid_step = full_horizon_steps - 1
        bar_limit = (max_valid_step // pos_size) + 1
        last_bar_pos_limit = (max_valid_step % pos_size) + 1
        min_local_step = prompt_chunks * CHUNK_STEPS
        floor_bar = min_local_step // pos_size
        floor_pos = min_local_step % pos_size
        min_bars_idx = bar_limit - 1

        prompt_notes = segment.prompt_5col_notes(prompt_chunks=prompt_chunks)
        global_notes: List[List[int]] = [
            [int(r[0]), int(r[1]), int(r[2]), int(r[3]), int(r[4])] for r in prompt_notes
        ]
        last_note_by_sp: Dict[Tuple[int, int], List[int]] = {
            (n[0], n[3]): n for n in global_notes
        }

        raw_active = [
            idx < len(segment.slot_codes) and int(segment.slot_codes[idx]) != SLOT_CODE_EMPTY
            for idx in range(slot_size)
        ]
        has_any_active = any(raw_active)
        active_flags = raw_active if has_any_active else [True] * slot_size

        seg_canonical = extract_segment_canonical_notes(
            prompt_notes,
            chunk_start=0,
            chunk_end=prompt_chunks,
            max_bars=segment.num_bars,
        )
        body_flat_u16 = self._tok.encode_canonical_segment_notes(
            seg_canonical, num_chunks=prompt_chunks
        )

        model.eval()
        model.reset_kv_cache()
        synchronize_device(dev)
        t0 = time.perf_counter()

        prefix_toks = model.build_prefix_tokens(
            slot_codes=segment.slot_codes, batch_size=1, device=dev
        )
        if len(body_flat_u16) > 0:
            body_toks = torch.from_numpy(body_flat_u16.astype(np.int64)).to(device=dev).reshape(1, -1)
            prompt_seq = torch.cat([prefix_toks, body_toks], dim=1)
        else:
            prompt_seq = prefix_toks

        fsm_state, cur_bar, last_pos, _, _ = model._parse_prompt_fsm_state(
            prompt_seq[0],
            prefix_len=int(prefix_toks.size(1)),
            initial_active_slots=active_flags,
        )
        prev_sp_in_pos = -1
        if (
            fsm_state == model._STATE_AFTER_VEL
            and len(seg_canonical) > 0
            and int(seg_canonical[-1, 0]) == cur_bar
            and int(seg_canonical[-1, 1]) == last_pos
        ):
            prev_sp_in_pos = int(seg_canonical[-1, 2]) * pitch_size + int(seg_canonical[-1, 3])

        prompt_steps = int(prompt_seq.size(1))
        prefill_mask = torch.ones_like(prompt_seq, dtype=torch.bool)
        logits = model.forward(
            prompt_seq, padding_mask=prefill_mask, is_causal=True, is_save_cache=True
        )
        next_logits_cpu = logits[:, -1, :].detach().to(device="cpu", dtype=torch.float32)
        synchronize_device(dev)
        t_prefill = time.perf_counter()

        pending_slot = 0
        pending_pitch = 60
        pending_dur_bin = 0
        notes_gen = 0
        gen_steps = 0
        max_gen_notes = 3072
        max_gen_tokens = max_gen_notes * 5 + bar_limit + 32
        vel_hi = int(model.vel_offset) + int(model.vel_size)
        bar_step_counts = {4: prompt_steps, 8: prompt_steps, 12: prompt_steps, 16: prompt_steps}

        for tok_step in range(max_gen_tokens):
            if notes_gen >= max_gen_notes and fsm_state in (
                model._STATE_BEFORE_BAR,
                model._STATE_AFTER_BAR,
                model._STATE_AFTER_VEL,
            ):
                break

            sampled_tok, is_eos = self._sample_next_fsm_token(
                model=model,
                raw_logits_row=next_logits_cpu[0],
                fsm_state=fsm_state,
                cur_bar=cur_bar,
                last_pos=last_pos,
                pending_slot=pending_slot,
                prev_sp_in_pos=prev_sp_in_pos,
                notes_gen=notes_gen,
                floor_bar=floor_bar,
                floor_pos=floor_pos,
                bar_limit=bar_limit,
                last_bar_pos_limit=last_bar_pos_limit,
                min_bars_idx=min_bars_idx,
                active_slot_flags=active_flags,
                has_any_active_slot=has_any_active,
                sampling=sampling,
            )
            gen_steps += 1
            if is_eos:
                break

            if sampled_tok == model.bar_token_id:
                cur_bar += 1
                last_pos = -1
                prev_sp_in_pos = -1
                fsm_state = model._STATE_AFTER_BAR
            elif model.pos_offset <= sampled_tok < model.slot_offset:
                if cur_bar < 0:
                    cur_bar = 0
                last_pos = sampled_tok - model.pos_offset
                prev_sp_in_pos = -1
                fsm_state = model._STATE_AFTER_POS
            elif model.slot_offset <= sampled_tok < model.pitch_offset:
                pending_slot = sampled_tok - model.slot_offset
                fsm_state = model._STATE_AFTER_SLOT
            elif model.pitch_offset <= sampled_tok < model.dur_offset:
                pending_pitch = sampled_tok - model.pitch_offset
                fsm_state = model._STATE_AFTER_PITCH
            elif model.dur_offset <= sampled_tok < model.vel_offset:
                pending_dur_bin = sampled_tok - model.dur_offset
                fsm_state = model._STATE_AFTER_DUR
            elif model.vel_offset <= sampled_tok < vel_hi:
                vel_bin = sampled_tok - model.vel_offset
                dur_steps = max(
                    1, int(self._dur_lut[min(max(0, pending_dur_bin), len(self._dur_lut) - 1)])
                )
                velocity = int(self._vel_lut[min(max(0, vel_bin), len(self._vel_lut) - 1)])
                global_start = cur_bar * pos_size + max(0, last_pos)
                global_end = min(full_horizon_steps, global_start + dur_steps)

                sp_key = (pending_slot, pending_pitch)
                prev_same = last_note_by_sp.get(sp_key)
                if prev_same is not None and prev_same[2] > global_start:
                    prev_same[2] = max(prev_same[1] + 1, global_start)

                if global_end > global_start:
                    new_note = [pending_slot, global_start, global_end, pending_pitch, velocity]
                    global_notes.append(new_note)
                    last_note_by_sp[sp_key] = new_note

                prev_sp_in_pos = pending_slot * pitch_size + pending_pitch
                notes_gen += 1
                fsm_state = model._STATE_AFTER_VEL

            cur_total_steps = prompt_steps + gen_steps
            for bar_mark in (8, 12, 16):
                if cur_bar < bar_mark:
                    bar_step_counts[bar_mark] = cur_total_steps

            if tok_step + 1 < max_gen_tokens:
                next_tok_dev = torch.tensor([sampled_tok], dtype=torch.long, device=dev)
                step_logits = model._forward_decode_step(next_tok_dev)
                next_logits_cpu = step_logits.detach().to(device="cpu", dtype=torch.float32)

        synchronize_device(dev)
        t_end = time.perf_counter()
        model.reset_kv_cache()

        bar_step_counts[16] = prompt_steps + gen_steps
        bar_step_counts[12] = max(bar_step_counts[8], bar_step_counts[12])
        traj = tuple((b, self.active_kv_mib(bar_step_counts[b])) for b in (4, 8, 12, 16))

        output_16bar_midi.parent.mkdir(parents=True, exist_ok=True)
        meta = slot_codes_to_roll_meta(segment.slot_codes)
        sorted_tuples = sorted(
            (int(n[0]), int(n[1]), int(n[2]), int(n[3]), int(n[4]))
            for n in global_notes
            if int(n[2]) > int(n[1])
        )
        _notes_to_midi(sorted_tuples, meta, output_16bar_midi)

        gen_bars = float(segment.num_chunks - prompt_chunks) / float(CHUNKS_PER_BAR)
        return RolloutTelemetry(
            prompt_steps=prompt_steps,
            generated_steps=gen_steps,
            total_steps=prompt_steps + gen_steps,
            prefill_ms=(t_prefill - t0) * 1000.0,
            decode_ms=(t_end - t_prefill) * 1000.0,
            total_ms=(t_end - t0) * 1000.0,
            generated_bars=gen_bars,
            peak_active_kv_mib=self.active_kv_mib(prompt_steps + gen_steps),
            kv_bar_trajectory_mib=traj,
        )


StrategyBuilderFn = Callable[[ModelCandidateSpec, torch.device], ParadigmEvaluationStrategy]


class ParadigmStrategyRegistry:
    """Open/Closed registry mapping paradigm keys to concrete `ParadigmEvaluationStrategy` builders."""

    def __init__(self) -> None:
        self._builders: Dict[str, StrategyBuilderFn] = {}

    def register(self, paradigm: str, builder: StrategyBuilderFn) -> None:
        self._builders[str(paradigm).strip().lower()] = builder

    def create(self, spec: ModelCandidateSpec, device: torch.device) -> ParadigmEvaluationStrategy:
        key = spec.paradigm.strip().lower()
        builder = self._builders.get(key)
        if builder is None:
            raise KeyError(
                f"Unsupported paradigm '{spec.paradigm}'. Registered: {sorted(self._builders.keys())}"
            )
        return builder(spec, device)


def _load_state_dict_into(module: torch.nn.Module, ckpt_path: str | Path) -> None:
    path_obj = Path(ckpt_path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path_obj}")
    raw = torch.load(str(path_obj), map_location="cpu", weights_only=False)
    sd = raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw
    module.load_state_dict(sd, strict=True)


def _build_symic_grouped_strategy(
    spec: ModelCandidateSpec, device: torch.device
) -> ParadigmEvaluationStrategy:
    m_args = MORTM5Args(spec.config_path)
    tok_key = m_args.tokenizer_key or "m4_g4l4"
    tok_info = TOKENIZER_CONFIGS[tok_key]
    vq_ckpt = spec.vq_checkpoint_path or tok_info["ckpt"]

    vq_args = MORTM5Args(tok_info["config"])
    vq_model = MidiVision(vq_args, sync_codebook=False).to(device).eval()
    _load_state_dict_into(vq_model, vq_ckpt)

    model = build_mortm5_ablation_model(m_args).to(device).eval()
    _load_state_dict_into(model, spec.checkpoint_path)
    if not isinstance(model, AblationTransformer2):
        raise TypeError(f"Expected AblationTransformer2 for {spec.key}, got {type(model).__name__}")
    return SyMIcGroupedStrategy(spec=spec, model=model, vq_model=vq_model, device=device)


def _build_octuple_strategy(
    spec: ModelCandidateSpec, device: torch.device
) -> ParadigmEvaluationStrategy:
    m_args = MORTM5Args(spec.config_path)
    model = build_mortm5_ablation_model(m_args).to(device).eval()
    _load_state_dict_into(model, spec.checkpoint_path)
    if not isinstance(model, OctupleTransformer):
        raise TypeError(f"Expected OctupleTransformer for {spec.key}, got {type(model).__name__}")
    return OctupleCanonicalStrategy(spec=spec, model=model, device=device)


def _build_remi_strategy(
    spec: ModelCandidateSpec, device: torch.device
) -> ParadigmEvaluationStrategy:
    m_args = MORTM5Args(spec.config_path)
    model = build_mortm5_ablation_model(m_args).to(device).eval()
    _load_state_dict_into(model, spec.checkpoint_path)
    if not isinstance(model, REMITransformer):
        raise TypeError(f"Expected REMITransformer for {spec.key}, got {type(model).__name__}")
    return RemiFsmCanonicalStrategy(spec=spec, model=model, device=device)


def create_default_strategy_registry() -> ParadigmStrategyRegistry:
    registry = ParadigmStrategyRegistry()
    registry.register("grouped", _build_symic_grouped_strategy)
    registry.register("octuple", _build_octuple_strategy)
    registry.register("remi", _build_remi_strategy)
    return registry
