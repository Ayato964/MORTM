from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .decoder_engine import CodebookDecomposition, MidiSampleContext
from .domain import DecodedFrameState, PitchWindow, QuantizedNote, Resolution, RgbColor


# Strict 90/10 Color Budget: 90%+ Neutral Monotone + 1 Single-Hue Solid Blue Accent
BG_PAGE: RgbColor = (250, 250, 250)        # #FAFAFA (Neutral off-white canvas)
BG_ROLL: RgbColor = (255, 255, 255)        # #FFFFFF (Crisp white piano-roll surface)
BG_BLACK_KEY: RgbColor = (248, 249, 251)   # Subtle pitch-lane alternation
LINE_DIVIDER: RgbColor = (229, 229, 229)   # #E5E5E5 (1px structural hairline)
LINE_BORDER: RgbColor = (209, 213, 219)    # #D1D5DB (Roll frame & bar lines)
LINE_OCTAVE: RgbColor = (212, 212, 216)    # #D4D4D8 (1px octave C boundary line)
LINE_BEAT: RgbColor = (241, 245, 249)      # Subtle quarter-note beat guide
GT_REFERENCE: RgbColor = (212, 216, 224)   # #D4D8E0 (Clear neutral ground-truth silhouette)

TEXT_PRIMARY: RgbColor = (23, 23, 23)      # #171717 (WCAG AAA 17.4:1 contrast)
TEXT_SECONDARY: RgbColor = (82, 82, 91)    # #52525B (WCAG AA+ 7.2:1 contrast)
TEXT_TERTIARY: RgbColor = (115, 115, 115)  # #737373 (Readable captions & labels)
TEXT_INACTIVE: RgbColor = (163, 163, 163)  # #A3A3A3 (Inactive layer indicator)

# Single functional accent family (Solid Deep Blue — zero neon, zero gradient)
ACCENT_BLUE: RgbColor = (29, 78, 216)      # #1D4ED8 (Primary active state & notes)
ACCENT_BLUE_DARK: RgbColor = (30, 58, 138) # #1E3A8A (High-velocity solid note & onset edge)
ACCENT_BLUE_MID: RgbColor = (59, 130, 246) # #3B82F6 (Soft velocity / morphing state)

_BLACK_KEY_PCS = frozenset({1, 3, 6, 8, 10})

_LAYER_DESCRIPTIONS: Tuple[Tuple[str, str], ...] = (
    ("Layer 1", "リズム骨格・主要和音"),
    ("Layer 2", "パート分離・主旋律"),
    ("Layer 3", "16分発音・音の伸び"),
    ("Layer 4", "強弱表現・終端補正"),
)


class FontSet:
    """Clean system font stack prioritizing macOS Hiragino Sans & SF Mono."""

    def __init__(self, scale: float = 1.0) -> None:
        self.h1 = self._load_jp_bold(int(round(32 * scale)))
        self.body = self._load_jp_regular(int(round(18 * scale)))
        self.layer_title = self._load_jp_bold(int(round(17 * scale)))
        self.layer_desc = self._load_jp_regular(int(round(15 * scale)))
        self.axis = self._load_jp_regular(int(round(15 * scale)))
        self.overline = self._load_jp_regular(int(round(14 * scale)))
        self.metric_val = self._load_mono_bold(int(round(32 * scale)))
        self.metric_sub = self._load_mono_regular(int(round(18 * scale)))
        self.mono_sm = self._load_mono_regular(int(round(14 * scale)))

    @staticmethod
    def _try_load(
        candidates: Tuple[tuple[str, int], ...],
        size: int,
    ) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        for path_str, index in candidates:
            if Path(path_str).is_file():
                try:
                    return ImageFont.truetype(path_str, size=size, index=index)
                except OSError:
                    continue
        return ImageFont.load_default()

    @classmethod
    def _load_jp_bold(cls, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        return cls._try_load(
            (
                ("/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc", 0),
                ("/System/Library/Fonts/Hiragino Sans GB.ttc", 0),
                ("/System/Library/Fonts/Supplemental/NotoSansGothic-Regular.ttf", 0),
                ("/System/Library/Fonts/HelveticaNeue.ttc", 1),
            ),
            size,
        )

    @classmethod
    def _load_jp_regular(cls, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        return cls._try_load(
            (
                ("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc", 0),
                ("/System/Library/Fonts/Hiragino Sans GB.ttc", 0),
                ("/System/Library/Fonts/Supplemental/NotoSansGothic-Regular.ttf", 0),
                ("/System/Library/Fonts/HelveticaNeue.ttc", 0),
            ),
            size,
        )

    @classmethod
    def _load_mono_bold(cls, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        return cls._try_load(
            (
                ("/System/Library/Fonts/SFNSMono.ttf", 0),
                ("/System/Library/Fonts/Menlo.ttc", 1),
                ("/System/Library/Fonts/Menlo.ttc", 0),
            ),
            size,
        )

    @classmethod
    def _load_mono_regular(cls, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        return cls._try_load(
            (
                ("/System/Library/Fonts/SFNSMono.ttf", 0),
                ("/System/Library/Fonts/Menlo.ttc", 0),
                ("/System/Library/Fonts/Helvetica.ttc", 0),
            ),
            size,
        )


class FrameRenderer(ABC):
    """Abstract presentation interface rendering domain frame states into RGBA arrays."""

    @property
    @abstractmethod
    def resolution(self) -> Resolution:
        ...

    @abstractmethod
    def render_frame(self, state: DecodedFrameState) -> np.ndarray:
        """Return contiguous uint8 RGBA array [height, width, 4]."""
        ...


class PianoRollDashboardRenderer(FrameRenderer):
    """Low-cognitive-load, typography-driven piano-roll renderer adhering to strict 90/10 design rules."""

    def __init__(
        self,
        sample: MidiSampleContext,
        decomposition: CodebookDecomposition,
        resolution: Resolution = Resolution(1920, 1080),
        fps: int = 30,
    ) -> None:
        self._sample = sample
        self._decomp = decomposition
        self._res = resolution
        self._fps = max(1, fps)

        self._sx = resolution.width / 1920.0
        self._sy = resolution.height / 1080.0
        self._fonts = FontSet(scale=min(self._sx, self._sy))

        # 8px-grid aligned margins and Negative Space layout (Zero Cardification)
        self._margin_x0 = self._px(80)
        self._margin_x1 = self._px(1840)

        self._plot_x0 = self._px(136)
        self._plot_x1 = self._px(1840)
        self._plot_w = self._plot_x1 - self._plot_x0

        self._plot_y0 = self._py(240)
        self._plot_y1 = self._py(832)
        self._plot_h = self._plot_y1 - self._plot_y0

        self._base_grid_rgb = self._build_clean_grid_with_gt_silhouette()
        self._static_canvas = self._build_static_layout()

    @property
    def resolution(self) -> Resolution:
        return self._res

    def _px(self, x_1920: float) -> int:
        return int(round(x_1920 * self._sx))

    def _py(self, y_1080: float) -> int:
        return int(round(y_1080 * self._sy))

    def _active_instruments_summary(self) -> str:
        active = [spec.display_label.split(" (")[0] for spec in self._sample.slot_specs if spec.is_active]
        return " / ".join(active) if active else "None"

    def _build_clean_grid_with_gt_silhouette(self) -> np.ndarray:
        """Precompute [pitch_span, total_steps, 3] float32 white piano-roll grid in [0, 1]."""
        pw: PitchWindow = self._sample.pitch_window
        steps = self._sample.bar_window.total_steps
        steps_per_bar = self._sample.bar_window.steps_per_bar
        steps_per_beat = steps_per_bar // 4

        white = np.asarray(BG_ROLL, dtype=np.float32) / 255.0
        black_row = np.asarray(BG_BLACK_KEY, dtype=np.float32) / 255.0
        beat_line = np.asarray(LINE_BEAT, dtype=np.float32) / 255.0
        bar_line = np.asarray(LINE_BORDER, dtype=np.float32) / 255.0
        gt_col = np.asarray(GT_REFERENCE, dtype=np.float32) / 255.0

        grid = np.empty((pw.span, steps, 3), dtype=np.float32)
        for row_idx in range(pw.span):
            pitch = pw.max_pitch - row_idx
            if (pitch % 12) in _BLACK_KEY_PCS:
                grid[row_idx, :] = black_row
            else:
                grid[row_idx, :] = white

        # Subtle quarter-note beat lines & crisp bar lines
        for st in range(steps_per_beat, steps, steps_per_beat):
            if st % steps_per_bar != 0:
                grid[:, st : st + 2] = np.minimum(grid[:, st : st + 2], beat_line)
        for st in range(steps_per_bar, steps, steps_per_bar):
            grid[:, st : st + 3] = bar_line

        # Quiet monotone silhouette of ground-truth MIDI notes
        for note in self._sample.gt_notes:
            if pw.min_pitch <= note.pitch <= pw.max_pitch:
                r = pw.max_pitch - note.pitch
                s0 = max(0, min(steps, note.start_step))
                s1 = max(0, min(steps, note.end_step))
                if s1 > s0:
                    grid[r, s0:s1] = gt_col

        return grid

    def _build_static_layout(self) -> Image.Image:
        """Render invariant typography, hairlines, and axis labels using pure negative space."""
        img = Image.new("RGBA", (self._res.width, self._res.height), (*BG_PAGE, 255))
        draw = ImageDraw.Draw(img)

        # Overline metadata (left) & instrumentation summary (right)
        draw.text(
            (self._margin_x0, self._py(40)),
            "MidiVision M4 (Deep-RVQ: G=4, L=4, D=128, K=2048)  /  冒頭8小節 CNNデコード可視化",
            font=self._fonts.overline,
            fill=TEXT_TERTIARY,
        )
        inst_text = f"編成: {self._active_instruments_summary()}"
        inst_bbox = draw.textbbox((0, 0), inst_text, font=self._fonts.overline)
        draw.text(
            (self._margin_x1 - (inst_bbox[2] - inst_bbox[0]), self._py(40)),
            inst_text,
            font=self._fonts.overline,
            fill=TEXT_TERTIARY,
        )

        # Top structural 1px divider
        draw.line(
            (self._margin_x0, self._py(184), self._margin_x1, self._py(184)),
            fill=LINE_DIVIDER,
            width=1,
        )

        # Minimal 2-item legend flush-right with piano roll edge (solid geometric swatches, no glyph symbols)
        sw_size = self._px(12)
        leg_y = self._py(206)
        gt_label = "元のMIDI（正解）"
        dec_label = "CNNデコード結果"
        gt_box = draw.textbbox((0, 0), gt_label, font=self._fonts.axis)
        dec_box = draw.textbbox((0, 0), dec_label, font=self._fonts.axis)
        gt_text_w = gt_box[2] - gt_box[0]
        dec_text_w = dec_box[2] - dec_box[0]

        lx_gt = self._plot_x1 - gt_text_w - self._px(8) - sw_size
        draw.rectangle(
            (lx_gt, leg_y + self._py(3), lx_gt + sw_size, leg_y + self._py(3) + sw_size),
            fill=GT_REFERENCE,
            outline=LINE_BORDER,
            width=1,
        )
        draw.text((lx_gt + sw_size + self._px(8), leg_y), gt_label, font=self._fonts.axis, fill=TEXT_TERTIARY)

        lx_dec = lx_gt - self._px(32) - dec_text_w - self._px(8) - sw_size
        draw.rectangle((lx_dec, leg_y + self._py(3), lx_dec + sw_size, leg_y + self._py(3) + sw_size), fill=ACCENT_BLUE)
        draw.text((lx_dec + sw_size + self._px(8), leg_y), dec_label, font=self._fonts.axis, fill=TEXT_SECONDARY)

        # Y-axis pitch octave labels (large, high-contrast, uncluttered)
        pw = self._sample.pitch_window
        for pitch in range(pw.min_pitch, pw.max_pitch + 1):
            if pitch % 12 == 0:
                octave = (pitch // 12) - 1
                row_ratio = (pw.max_pitch - pitch + 0.5) / float(pw.span)
                py_pos = self._plot_y0 + int(round(row_ratio * self._plot_h))
                label = f"C{octave}"
                lb_box = draw.textbbox((0, 0), label, font=self._fonts.mono_sm)
                lw = lb_box[2] - lb_box[0]
                draw.text(
                    (self._plot_x0 - lw - self._px(14), py_pos - self._py(8)),
                    label,
                    font=self._fonts.mono_sm,
                    fill=TEXT_SECONDARY,
                )

        # X-axis bar labels
        num_bars = self._sample.bar_window.num_bars
        for b in range(num_bars):
            bx = self._plot_x0 + int(round((b / float(num_bars)) * self._plot_w))
            draw.text(
                (bx + self._px(4), self._plot_y1 + self._py(10)),
                f"{b + 1}小節",
                font=self._fonts.axis,
                fill=TEXT_SECONDARY,
            )

        # Bottom structural 1px divider separating piano roll and 4-column layer strip
        draw.line(
            (self._margin_x0, self._py(888), self._margin_x1, self._py(888)),
            fill=LINE_DIVIDER,
            width=1,
        )

        return img

    def _rasterize_piano_roll(
        self,
        probs_flat: np.ndarray,
        notes: Tuple[QuantizedNote, ...],
    ) -> Image.Image:
        """Render crisp single-hue piano roll without neon glow or multi-color patchwork."""
        pw = self._sample.pitch_window
        steps = self._sample.bar_window.total_steps
        canvas = self._base_grid_rgb.copy()

        p_lo, p_hi = pw.min_pitch, pw.max_pitch + 1
        col_mid = np.asarray(ACCENT_BLUE_MID, dtype=np.float32) / 255.0
        col_solid = np.asarray(ACCENT_BLUE, dtype=np.float32) / 255.0
        col_dark = np.asarray(ACCENT_BLUE_DARK, dtype=np.float32) / 255.0

        # 1. Smooth morphing density during latent transitions (solid blue opacity blend on white, zero glow)
        sustain_max = probs_flat[1::2, :, p_lo:p_hi].max(axis=0).T[::-1]  # [P, steps]
        onset_max = probs_flat[0::2, :, p_lo:p_hi].max(axis=0).T[::-1]
        presence = np.clip((np.maximum(sustain_max, onset_max) - 0.22) / 0.55, 0.0, 0.45)[..., None]
        canvas = (1.0 - presence) * canvas + presence * col_mid

        # 2. Crisp decoded MIDI note bars in disciplined single-hue blue scale
        for note in notes:
            if not (pw.min_pitch <= note.pitch <= pw.max_pitch):
                continue
            r = pw.max_pitch - note.pitch
            s0 = max(0, min(steps, note.start_step))
            s1 = max(0, min(steps, note.end_step))
            if s1 <= s0:
                continue

            v_t = note.velocity_norm
            note_rgb = (1.0 - v_t) * col_mid + v_t * col_solid
            canvas[r, s0:s1] = note_rgb

            # Distinct note-onset edge (first 2 steps) so repeated notes never visually merge
            head_end = min(s1, s0 + 2)
            canvas[r, s0:head_end] = col_dark

        u8 = np.rint(np.clip(canvas, 0.0, 1.0) * 255.0).astype(np.uint8)
        return self._upscale_roll(u8, self._plot_w, self._plot_h, pw.max_pitch)

    @staticmethod
    def _upscale_roll(u8_grid: np.ndarray, target_w: int, target_h: int, max_pitch: int) -> Image.Image:
        """Horizontal BOX resize preserves thin attacks; clean 1px pitch-lane & octave hairlines keep chords legible."""
        h_pitches, _, _ = u8_grid.shape
        pil_raw = Image.fromarray(u8_grid, mode="RGB")
        horiz = pil_raw.resize((target_w, h_pitches), resample=Image.Resampling.BOX)
        full = horiz.resize((target_w, target_h), resample=Image.Resampling.NEAREST)

        if target_h // h_pitches >= 6:
            arr = np.asarray(full).copy()
            pitch_line = np.array([244, 245, 247], dtype=np.uint16)
            octave_line = np.asarray(LINE_OCTAVE, dtype=np.uint16)
            for r in range(1, h_pitches):
                y_line = int(round(r * target_h / float(h_pitches)))
                if 0 <= y_line < target_h:
                    upper_pitch = max_pitch - (r - 1)
                    if upper_pitch % 12 == 0:
                        # Crisp 1px octave boundary directly below C
                        arr[y_line, :] = ((arr[y_line, :].astype(np.uint16) * 2 + octave_line * 2) // 4).astype(
                            np.uint8
                        )
                    else:
                        arr[y_line, :] = ((arr[y_line, :].astype(np.uint16) * 3 + pitch_line) // 4).astype(np.uint8)
            return Image.fromarray(arr, mode="RGB")
        return full

    def render_frame(self, state: DecodedFrameState) -> np.ndarray:
        frame_img = self._static_canvas.copy()
        draw = ImageDraw.Draw(frame_img)

        # 1. Main Piano Roll with crisp 1px border (no drop shadows)
        roll_img = self._rasterize_piano_roll(state.probs_flat, state.notes)
        frame_img.paste(roll_img, (self._plot_x0, self._plot_y0))
        draw.rectangle(
            (self._plot_x0 - 1, self._plot_y0 - 1, self._plot_x1, self._plot_y1),
            outline=LINE_BORDER,
            width=1,
        )

        # 2. Header Typography & Core 3 Metrics (Negative Space grouping, zero boxes)
        self._draw_header(draw, state)

        # 3. Bottom 4-Column Codebook Layer Strip (Hairline indicator bars, zero card clutter)
        self._draw_layer_columns(draw, state)

        # 4. Subtle bottom timeline hairline
        self._draw_timeline_hairline(draw, state)

        return np.ascontiguousarray(np.asarray(frame_img, dtype=np.uint8))

    def _draw_header(self, draw: ImageDraw.ImageDraw, state: DecodedFrameState) -> None:
        desc = state.descriptor
        m = state.metrics

        # Left: Stage H1 & concise 30-38 char Japanese explanation
        draw.text(
            (self._margin_x0, self._py(74)),
            desc.title_ja,
            font=self._fonts.h1,
            fill=TEXT_PRIMARY,
        )
        draw.text(
            (self._margin_x0, self._py(126)),
            desc.subtitle_ja,
            font=self._fonts.body,
            fill=TEXT_SECONDARY,
        )

        # Right: 3 flush-right tabular metrics aligned to x=1840 on an 8px grid (zero jitter, zero boxes)
        metrics_specs = (
            ("Note F1", f"{m.note_f1 * 100:.1f}%", None),
            ("Onset F1", f"{m.onset_f1 * 100:.1f}%", None),
            ("検出音数", f"{m.decoded_note_count}", f"/ {len(self._sample.gt_notes)}"),
        )

        col_right_edges = (self._px(1360), self._px(1600), self._margin_x1)
        label_y = self._py(76)
        val_y = self._py(104)

        for (label, main_val, sub_val), rx in zip(metrics_specs, col_right_edges):
            lbox = draw.textbbox((0, 0), label, font=self._fonts.overline)
            lw = lbox[2] - lbox[0]
            draw.text((rx - lw, label_y), label, font=self._fonts.overline, fill=TEXT_TERTIARY)

            if sub_val is None:
                vbox = draw.textbbox((0, 0), main_val, font=self._fonts.metric_val)
                vw = vbox[2] - vbox[0]
                draw.text((rx - vw, val_y), main_val, font=self._fonts.metric_val, fill=TEXT_PRIMARY)
            else:
                sbox = draw.textbbox((0, 0), sub_val, font=self._fonts.metric_sub)
                sw = sbox[2] - sbox[0]
                draw.text((rx - sw, val_y + self._py(12)), sub_val, font=self._fonts.metric_sub, fill=TEXT_TERTIARY)

                vbox = draw.textbbox((0, 0), main_val, font=self._fonts.metric_val)
                vw = vbox[2] - vbox[0]
                draw.text(
                    (rx - sw - self._px(8) - vw, val_y),
                    main_val,
                    font=self._fonts.metric_val,
                    fill=TEXT_PRIMARY,
                )

    def _draw_layer_columns(self, draw: ImageDraw.ImageDraw, state: DecodedFrameState) -> None:
        desc = state.descriptor
        norms = self._decomp.layer_mean_norms

        total_w = self._margin_x1 - self._margin_x0
        gutter = self._px(48)
        col_w = (total_w - gutter * 3) // 4
        bar_y = self._py(912)
        bar_h = max(3, self._py(4))

        for l_idx in range(4):
            cx0 = self._margin_x0 + l_idx * (col_w + gutter)
            cx1 = cx0 + col_w
            is_on = desc.active_layers_mask[l_idx]
            is_hl = (desc.highlight_layer_index == l_idx)

            # Top 4px indicator bar: solid blue when active, quiet neutral gray when inactive
            bar_color = ACCENT_BLUE if is_on else LINE_DIVIDER
            draw.rectangle((cx0, bar_y, cx1, bar_y + bar_h), fill=bar_color)

            l_name, l_role = _LAYER_DESCRIPTIONS[l_idx]
            title_col = ACCENT_BLUE if is_hl else (TEXT_PRIMARY if is_on else TEXT_INACTIVE)
            desc_col = TEXT_SECONDARY if is_on else TEXT_INACTIVE
            norm_col = TEXT_TERTIARY if is_on else TEXT_INACTIVE

            draw.text((cx0, self._py(928)), l_name, font=self._fonts.layer_title, fill=title_col)

            norm_val = norms[l_idx] if l_idx < len(norms) else 0.0
            norm_str = f"||c{l_idx + 1}|| = {norm_val:4.2f}"
            nb = draw.textbbox((0, 0), norm_str, font=self._fonts.mono_sm)
            draw.text(
                (cx1 - (nb[2] - nb[0]), self._py(931)),
                norm_str,
                font=self._fonts.mono_sm,
                fill=norm_col,
            )

            draw.text((cx0, self._py(958)), l_role, font=self._fonts.layer_desc, fill=desc_col)

    def _draw_timeline_hairline(self, draw: ImageDraw.ImageDraw, state: DecodedFrameState) -> None:
        y_line = self._py(1040)
        x0 = self._margin_x0
        x1 = self._px(1720)
        draw.line((x0, y_line, x1, y_line), fill=LINE_DIVIDER, width=max(2, self._py(2)))

        prog_x = x0 + int(round(state.descriptor.timeline_progress * (x1 - x0)))
        if prog_x > x0:
            draw.line((x0, y_line, prog_x, y_line), fill=ACCENT_BLUE, width=max(2, self._py(2)))

        t_curr = state.frame_index / float(self._fps)
        t_tot = state.total_frames / float(self._fps)
        time_str = f"{t_curr:4.1f}s / {t_tot:4.1f}s"
        tb = draw.textbbox((0, 0), time_str, font=self._fonts.mono_sm)
        draw.text(
            (self._margin_x1 - (tb[2] - tb[0]), y_line - self._py(8)),
            time_str,
            font=self._fonts.mono_sm,
            fill=TEXT_TERTIARY,
        )
