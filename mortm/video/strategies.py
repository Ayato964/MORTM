from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterator, Sequence, Tuple

import torch

from .decoder_engine import CodebookDecomposition
from .domain import StageDescriptor


def _cubic_ease_in_out(t: float) -> float:
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    return 4.0 * t * t * t if t < 0.5 else 1.0 - ((-2.0 * t + 2.0) ** 3) / 2.0


@dataclass(frozen=True)
class WaypointSpec:
    """Declarative specification of one key stage in a latent space trajectory."""
    stage_id: str
    badge_text: str
    title_ja: str
    subtitle_ja: str
    latent: torch.Tensor
    delta_reference_latent: torch.Tensor | None
    active_layers_mask: Tuple[bool, bool, bool, bool]
    highlight_layer_index: int | None
    transition_seconds: float
    hold_seconds: float


@dataclass(frozen=True)
class LatentTrajectoryStep:
    """Single-frame latent point yielded lazily by a TrajectoryStrategy."""
    frame_index: int
    total_frames: int
    latent: torch.Tensor
    delta_reference_latent: torch.Tensor | None
    descriptor: StageDescriptor
    is_stage_hold_anchor: bool


class TrajectoryStrategy(ABC):
    """Strategy interface generating a lazy stream of latent vectors across RVQ stages."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def build_waypoints(self, decomp: CodebookDecomposition) -> Sequence[WaypointSpec]:
        ...

    def iter_steps(self, decomp: CodebookDecomposition, fps: int) -> Iterator[LatentTrajectoryStep]:
        waypoints = list(self.build_waypoints(decomp))
        if not waypoints:
            return

        # Precompute total frame count in O(W) where W ~ 6 waypoints
        segment_counts: list[tuple[int, int]] = []
        for idx, wp in enumerate(waypoints):
            trans_frames = 0 if idx == 0 else max(1, int(round(wp.transition_seconds * fps)))
            hold_frames = max(1, int(round(wp.hold_seconds * fps)))
            segment_counts.append((trans_frames, hold_frames))

        total_frames = sum(tr + hd for tr, hd in segment_counts)
        frame_cursor = 0

        for idx, wp in enumerate(waypoints):
            trans_frames, hold_frames = segment_counts[idx]
            prev_latent = waypoints[idx - 1].latent if idx > 0 else wp.latent
            stage_total = trans_frames + hold_frames

            for f in range(trans_frames):
                raw_t = (f + 1) / float(trans_frames)
                alpha = _cubic_ease_in_out(raw_t)
                z_curr = (1.0 - alpha) * prev_latent + alpha * wp.latent
                timeline_prog = frame_cursor / float(max(1, total_frames - 1))
                desc = StageDescriptor(
                    stage_id=wp.stage_id,
                    badge_text=wp.badge_text,
                    title_ja=wp.title_ja,
                    subtitle_ja=wp.subtitle_ja,
                    active_layers_mask=wp.active_layers_mask,
                    highlight_layer_index=wp.highlight_layer_index,
                    progress_in_stage=(f + 1) / float(stage_total),
                    timeline_progress=timeline_prog,
                )
                yield LatentTrajectoryStep(
                    frame_index=frame_cursor,
                    total_frames=total_frames,
                    latent=z_curr,
                    delta_reference_latent=wp.delta_reference_latent,
                    descriptor=desc,
                    is_stage_hold_anchor=False,
                )
                frame_cursor += 1

            for h in range(hold_frames):
                timeline_prog = frame_cursor / float(max(1, total_frames - 1))
                desc = StageDescriptor(
                    stage_id=wp.stage_id,
                    badge_text=wp.badge_text,
                    title_ja=wp.title_ja,
                    subtitle_ja=wp.subtitle_ja,
                    active_layers_mask=wp.active_layers_mask,
                    highlight_layer_index=wp.highlight_layer_index,
                    progress_in_stage=(trans_frames + h + 1) / float(stage_total),
                    timeline_progress=timeline_prog,
                )
                yield LatentTrajectoryStep(
                    frame_index=frame_cursor,
                    total_frames=total_frames,
                    latent=wp.latent,
                    delta_reference_latent=wp.delta_reference_latent,
                    descriptor=desc,
                    is_stage_hold_anchor=(h == 0),
                )
                frame_cursor += 1


class NoiseToCoarseToFineStrategy(TrajectoryStrategy):
    """Empty -> Unquantized Sphere Noise -> RVQ L1 -> L1+L2 -> L1+L2+L3 -> L1+L2+L3+L4."""

    @property
    def name(self) -> str:
        return "noise_to_fine"

    def build_waypoints(self, decomp: CodebookDecomposition) -> Sequence[WaypointSpec]:
        z_l1 = decomp.cumulative_latent(1)
        z_l2 = decomp.cumulative_latent(2)
        z_l3 = decomp.cumulative_latent(3)
        z_l4 = decomp.cumulative_latent(4)

        return (
            WaypointSpec(
                stage_id="00_empty",
                badge_text="無音",
                title_ja="初期状態 — 空白のピアノロール",
                subtitle_ja="音符がまだ1つも存在しない無音の潜在表現から開始します。",
                latent=decomp.z_silent,
                delta_reference_latent=decomp.z_silent,
                active_layers_mask=(False, False, False, False),
                highlight_layer_index=None,
                transition_seconds=0.0,
                hold_seconds=1.1,
            ),
            WaypointSpec(
                stage_id="01_noise",
                badge_text="ノイズ",
                title_ja="未量子化ノイズ — コードブック適用前",
                subtitle_ja="ランダムな潜在ベクトルをCNNでデコードすると無秩序な音群が現れます。",
                latent=decomp.z_noise,
                delta_reference_latent=decomp.z_silent,
                active_layers_mask=(False, False, False, False),
                highlight_layer_index=None,
                transition_seconds=1.5,
                hold_seconds=1.5,
            ),
            WaypointSpec(
                stage_id="02_layer1",
                badge_text="Layer 1",
                title_ja="Layer 1 — リズム骨格と主要和音",
                subtitle_ja="ノイズが一掃され、楽曲の主拍と基本コードの大枠が立ち現れます。",
                latent=z_l1,
                delta_reference_latent=decomp.z_noise,
                active_layers_mask=(True, False, False, False),
                highlight_layer_index=0,
                transition_seconds=1.8,
                hold_seconds=2.0,
            ),
            WaypointSpec(
                stage_id="03_layer1_2",
                badge_text="Layer 1–2",
                title_ja="Layer 1–2 — パート分離と旋律の確立",
                subtitle_ja="2層目の残差が加わり、主旋律と動くベースラインが明確に分離します。",
                latent=z_l2,
                delta_reference_latent=z_l1,
                active_layers_mask=(True, True, False, False),
                highlight_layer_index=1,
                transition_seconds=1.6,
                hold_seconds=2.0,
            ),
            WaypointSpec(
                stage_id="04_layer1_3",
                badge_text="Layer 1–3",
                title_ja="Layer 1–3 — 微細発音と音の伸びの補正",
                subtitle_ja="3層目が16分音符の細かな発音と、チャンクを跨ぐ音の伸びを繋ぎます。",
                latent=z_l3,
                delta_reference_latent=z_l2,
                active_layers_mask=(True, True, True, False),
                highlight_layer_index=2,
                transition_seconds=1.6,
                hold_seconds=2.0,
            ),
            WaypointSpec(
                stage_id="05_layer1_4",
                badge_text="Layer 1–4",
                title_ja="Layer 1–4 — 強弱表現と微細エッジの完成",
                subtitle_ja="4層目がベロシティの強弱と音の終端を整え、元のMIDIへ収束します。",
                latent=z_l4,
                delta_reference_latent=z_l3,
                active_layers_mask=(True, True, True, True),
                highlight_layer_index=3,
                transition_seconds=1.6,
                hold_seconds=2.8,
            ),
        )


class OriginToCoarseToFineStrategy(TrajectoryStrategy):
    """Empty -> Zero Origin (0) -> RVQ L1 -> L1+L2 -> L1+L2+L3 -> L1+L2+L3+L4."""

    @property
    def name(self) -> str:
        return "origin_to_fine"

    def build_waypoints(self, decomp: CodebookDecomposition) -> Sequence[WaypointSpec]:
        base = list(NoiseToCoarseToFineStrategy().build_waypoints(decomp))
        base[1] = WaypointSpec(
            stage_id="01_origin",
            badge_text="原点 z=0",
            title_ja="ゼロ潜在ベクトル — コードブック加算前",
            subtitle_ja="球面正規化の中心点 z=0 をデコードすると定常的なノイズが現れます。",
            latent=decomp.z_origin,
            delta_reference_latent=decomp.z_silent,
            active_layers_mask=(False, False, False, False),
            highlight_layer_index=None,
            transition_seconds=1.4,
            hold_seconds=1.5,
        )
        wp2 = base[2]
        base[2] = WaypointSpec(
            stage_id=wp2.stage_id,
            badge_text=wp2.badge_text,
            title_ja=wp2.title_ja,
            subtitle_ja=wp2.subtitle_ja,
            latent=wp2.latent,
            delta_reference_latent=decomp.z_origin,
            active_layers_mask=wp2.active_layers_mask,
            highlight_layer_index=wp2.highlight_layer_index,
            transition_seconds=wp2.transition_seconds,
            hold_seconds=wp2.hold_seconds,
        )
        return tuple(base)


class FineToCoarseStrategy(TrajectoryStrategy):
    """Empty -> Fine Residual L4 (noisy) -> L4+L3 -> L4+L3+L2 -> Full L4+L3+L2+L1."""

    @property
    def name(self) -> str:
        return "fine_to_coarse"

    def build_waypoints(self, decomp: CodebookDecomposition) -> Sequence[WaypointSpec]:
        r1 = decomp.reverse_cumulative_latent(1)
        r2 = decomp.reverse_cumulative_latent(2)
        r3 = decomp.reverse_cumulative_latent(3)
        r4 = decomp.reverse_cumulative_latent(4)
        return (
            WaypointSpec(
                stage_id="00_empty",
                badge_text="無音",
                title_ja="初期状態 — 空白のピアノロール",
                subtitle_ja="最下層の微細残差から逆順にコードブックを重ねていきます。",
                latent=decomp.z_silent,
                delta_reference_latent=decomp.z_silent,
                active_layers_mask=(False, False, False, False),
                highlight_layer_index=None,
                transition_seconds=0.0,
                hold_seconds=1.1,
            ),
            WaypointSpec(
                stage_id="01_l4_only",
                badge_text="Layer 4",
                title_ja="Layer 4 単体 — 最下層の微細残差のみ",
                subtitle_ja="微細な補正成分のみのため大局構造を持たず、ノイズ状に現れます。",
                latent=r1,
                delta_reference_latent=decomp.z_silent,
                active_layers_mask=(False, False, False, True),
                highlight_layer_index=3,
                transition_seconds=1.5,
                hold_seconds=1.8,
            ),
            WaypointSpec(
                stage_id="02_l3_l4",
                badge_text="Layer 3–4",
                title_ja="Layer 3–4 — 局所的なリズム断片の凝集",
                subtitle_ja="3層目が加わり装飾パターンが見え始めますが、全体像はまだ曖昧です。",
                latent=r2,
                delta_reference_latent=r1,
                active_layers_mask=(False, False, True, True),
                highlight_layer_index=2,
                transition_seconds=1.6,
                hold_seconds=1.8,
            ),
            WaypointSpec(
                stage_id="03_l2_l3_l4",
                badge_text="Layer 2–4",
                title_ja="Layer 2–4 — 中間構造の形成",
                subtitle_ja="2層目で旋律の輪郭が現れますが、1層目の土台がなく過剰発音が残ります。",
                latent=r3,
                delta_reference_latent=r2,
                active_layers_mask=(False, True, True, True),
                highlight_layer_index=1,
                transition_seconds=1.6,
                hold_seconds=1.8,
            ),
            WaypointSpec(
                stage_id="04_full",
                badge_text="全層統合",
                title_ja="Layer 1 合流 — 大局骨格と噛み合い完全復元",
                subtitle_ja="最後に1層目の土台が加わることで全残差が収束し、元のMIDIが現れます。",
                latent=r4,
                delta_reference_latent=r3,
                active_layers_mask=(True, True, True, True),
                highlight_layer_index=0,
                transition_seconds=1.8,
                hold_seconds=2.8,
            ),
        )


class IsolatedLayersStrategy(TrajectoryStrategy):
    """Inspects each individual codebook layer alone: Empty -> L1 -> L2 -> L3 -> L4 -> Full."""

    @property
    def name(self) -> str:
        return "isolated_layers"

    def build_waypoints(self, decomp: CodebookDecomposition) -> Sequence[WaypointSpec]:
        c = decomp.layer_latents
        full = decomp.cumulative_latent(4)
        return (
            WaypointSpec(
                stage_id="00_empty",
                badge_text="無音",
                title_ja="初期状態 — 各レイヤー単体の比較",
                subtitle_ja="1層目から4層目まで、各コードブック単体のデコード結果を順に確認します。",
                latent=decomp.z_silent,
                delta_reference_latent=decomp.z_silent,
                active_layers_mask=(False, False, False, False),
                highlight_layer_index=None,
                transition_seconds=0.0,
                hold_seconds=1.0,
            ),
            WaypointSpec(
                stage_id="01_iso_l1",
                badge_text="Layer 1 単体",
                title_ja="Layer 1 単体 — 大局骨格・主拍・基本和音",
                subtitle_ja="1層目単独でも楽曲の主要なリズムと和音配置が明確に保持されています。",
                latent=c[0],
                delta_reference_latent=decomp.z_silent,
                active_layers_mask=(True, False, False, False),
                highlight_layer_index=0,
                transition_seconds=1.5,
                hold_seconds=2.0,
            ),
            WaypointSpec(
                stage_id="02_iso_l2",
                badge_text="Layer 2 単体",
                title_ja="Layer 2 単体 — 第1残差（パート補正成分）",
                subtitle_ja="1層目からの差分ベクトルのため、単体では部分的な音群とノイズが混在します。",
                latent=c[1],
                delta_reference_latent=decomp.z_silent,
                active_layers_mask=(False, True, False, False),
                highlight_layer_index=1,
                transition_seconds=1.5,
                hold_seconds=1.8,
            ),
            WaypointSpec(
                stage_id="03_iso_l3",
                badge_text="Layer 3 単体",
                title_ja="Layer 3 単体 — 第2残差（微細発音補正）",
                subtitle_ja="さらに細かな高周波残差を保持しており、単体では粒状の応答を示します。",
                latent=c[2],
                delta_reference_latent=decomp.z_silent,
                active_layers_mask=(False, False, True, False),
                highlight_layer_index=2,
                transition_seconds=1.5,
                hold_seconds=1.8,
            ),
            WaypointSpec(
                stage_id="04_iso_l4",
                badge_text="Layer 4 単体",
                title_ja="Layer 4 単体 — 第3残差（強弱・終端微調整）",
                subtitle_ja="最小ノルムの微細補正項であり、単体では背景ノイズ状の分布となります。",
                latent=c[3],
                delta_reference_latent=decomp.z_silent,
                active_layers_mask=(False, False, False, True),
                highlight_layer_index=3,
                transition_seconds=1.5,
                hold_seconds=1.8,
            ),
            WaypointSpec(
                stage_id="05_full_sum",
                badge_text="全層統合",
                title_ja="全4レイヤー統合 — 線形加算による完全復元",
                subtitle_ja="4層のベクトルが潜在空間で足し合わされ、元の演奏が高精度に復元されます。",
                latent=full,
                delta_reference_latent=c[0],
                active_layers_mask=(True, True, True, True),
                highlight_layer_index=None,
                transition_seconds=1.8,
                hold_seconds=2.8,
            ),
        )


class TrajectoryStrategyRegistry:
    """Open/Closed registry mapping strategy names to TrajectoryStrategy implementations."""

    def __init__(self) -> None:
        self._strategies: Dict[str, TrajectoryStrategy] = {}

    def register(self, strategy: TrajectoryStrategy) -> None:
        self._strategies[strategy.name] = strategy

    def get(self, name: str) -> TrajectoryStrategy:
        strategy = self._strategies.get(name)
        if strategy is None:
            available = ", ".join(sorted(self._strategies.keys()))
            raise ValueError(f"Unknown strategy '{name}'. Available strategies: [{available}]")
        return strategy

    def available_names(self) -> Tuple[str, ...]:
        return tuple(sorted(self._strategies.keys()))


def build_default_strategy_registry() -> TrajectoryStrategyRegistry:
    registry = TrajectoryStrategyRegistry()
    registry.register(NoiseToCoarseToFineStrategy())
    registry.register(OriginToCoarseToFineStrategy())
    registry.register(FineToCoarseStrategy())
    registry.register(IsolatedLayersStrategy())
    return registry
