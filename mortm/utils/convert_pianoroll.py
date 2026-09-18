from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import mido
import numpy as np


STEPS_PER_BAR = 192
BEATS_PER_BAR = 4
MAX_TRACKS = 4
FEATURES_PER_TRACK = 2
TOTAL_CHANNELS = MAX_TRACKS * FEATURES_PER_TRACK
CHUNK_STEPS = 24
CHUNKS_PER_BAR = STEPS_PER_BAR // CHUNK_STEPS

if STEPS_PER_BAR % CHUNK_STEPS != 0:
    raise ValueError("STEPS_PER_BAR must be divisible by CHUNK_STEPS.")
DRUM_MIDI_CHANNEL = 9  # MIDI channel 10 in 1-based notation


@dataclass(frozen=True)
class Note:
    start_tick: int
    end_tick: int
    pitch: int
    velocity: int
    channel: int
    program: int
    source_track: int
    track_name: str
    is_drum: bool


@dataclass
class SlotInfo:
    slot: int
    is_drum: bool
    program: int
    name: str
    source_track: int | None
    source_channel: int | None
    note_count: int


@dataclass
class RollMeta:
    ticks_per_beat: int
    steps_per_bar: int
    beats_per_bar: int
    max_tracks: int
    features_per_track: int
    slots: list[SlotInfo]

    @property
    def ticks_per_step(self) -> float:
        return (self.ticks_per_beat * self.beats_per_bar) / self.steps_per_bar


def _validate_four_four(mid: mido.MidiFile) -> None:
    """Validate 4/4 time signature (disabled: universal time signature support enabled)."""
    pass


def extract_notes(mid: mido.MidiFile) -> list[Note]:
    """Extract effective audible notes in absolute ticks, applying CC64 sustain.

    CC64 semantics for non-drum channels:
      - value >= 64: pedal down
      - value < 64: pedal up
      - a note_off while pedal is down is deferred until pedal-up
      - if the same pitch is re-attacked while an older released note is being
        held by the pedal, the older effective note ends at the new onset. This
        canonicalizes same-pitch pedal overlap into a form representable by the
        onset_velocity + note_active piano roll.

    Drums ignore CC64.
    """
    notes: list[Note] = []

    def append_note(
        start_tick: int,
        end_tick: int,
        pitch: int,
        velocity: int,
        channel: int,
        program: int,
        source_track: int,
        track_name: str,
    ) -> None:
        if end_tick <= start_tick:
            end_tick = start_tick + 1
        notes.append(
            Note(
                start_tick=start_tick,
                end_tick=end_tick,
                pitch=pitch,
                velocity=velocity,
                channel=channel,
                program=program,
                source_track=source_track,
                track_name=track_name,
                is_drum=(channel == DRUM_MIDI_CHANNEL),
            )
        )

    for track_idx, track in enumerate(mid.tracks):
        abs_tick = 0
        track_name = f"track_{track_idx}"
        programs = [0] * 16
        pedal_down = [False] * 16

        # Keys are (channel, pitch). A deque handles unusual overlapping note-ons.
        held: dict[tuple[int, int], deque[tuple[int, int, int]]] = defaultdict(deque)
        pedal_held: dict[tuple[int, int], deque[tuple[int, int, int]]] = defaultdict(deque)

        def release_pedal_notes(channel: int, end_tick: int, pitch: int | None = None) -> None:
            keys = list(pedal_held.keys())
            for key in keys:
                ch, p = key
                if ch != channel or (pitch is not None and p != pitch):
                    continue
                queue = pedal_held[key]
                while queue:
                    start_tick, velocity, program = queue.popleft()
                    append_note(
                        start_tick, end_tick, p, velocity, ch, program,
                        track_idx, track_name,
                    )

        for msg in track:
            abs_tick += msg.time

            if msg.type == "track_name":
                track_name = msg.name

            elif msg.type == "program_change":
                programs[msg.channel] = msg.program

            elif msg.type == "control_change" and msg.control == 64:
                channel = msg.channel
                if channel == DRUM_MIDI_CHANNEL:
                    continue
                new_down = msg.value >= 64
                if pedal_down[channel] and not new_down:
                    release_pedal_notes(channel, abs_tick)
                pedal_down[channel] = new_down

            elif msg.type == "note_on" and msg.velocity > 0:
                key = (msg.channel, msg.note)

                # A re-attack is an explicit new boundary. Any older instance of
                # this pitch that only survives because of CC64 ends here.
                if msg.channel != DRUM_MIDI_CHANNEL and pedal_held[key]:
                    release_pedal_notes(msg.channel, abs_tick, pitch=msg.note)

                held[key].append((abs_tick, msg.velocity, programs[msg.channel]))

            elif msg.type == "note_off" or (
                msg.type == "note_on" and msg.velocity == 0
            ):
                key = (msg.channel, msg.note)
                if held[key]:
                    start_tick, velocity, program = held[key].popleft()
                    if msg.channel != DRUM_MIDI_CHANNEL and pedal_down[msg.channel]:
                        pedal_held[key].append((start_tick, velocity, program))
                    else:
                        append_note(
                            start_tick, abs_tick, msg.note, velocity, msg.channel,
                            program, track_idx, track_name,
                        )

        # Close any hanging physical keys and pedal-held notes at track end.
        for (channel, pitch), queue in held.items():
            while queue:
                start_tick, velocity, program = queue.popleft()
                append_note(
                    start_tick, max(abs_tick, start_tick + 1), pitch, velocity,
                    channel, program, track_idx, track_name,
                )

        for (channel, pitch), queue in pedal_held.items():
            while queue:
                start_tick, velocity, program = queue.popleft()
                append_note(
                    start_tick, max(abs_tick, start_tick + 1), pitch, velocity,
                    channel, program, track_idx, track_name,
                )

    return notes


def _quantize_note(note: Note, ticks_per_step: float) -> tuple[int, int, int, int]:
    start = int(round(note.start_tick / ticks_per_step))
    end = int(round(note.end_tick / ticks_per_step))
    end = max(end, start + 1)
    return start, end, note.pitch, note.velocity


def _lane_key(note: Note) -> tuple[int, int, int]:
    # Program is part of the key so a source MIDI track that changes instruments
    # is treated as multiple candidate lanes.
    return note.source_track, note.channel, note.program


def _choose_slots(notes: list[Note]) -> tuple[list[Note], list[SlotInfo], dict[tuple[int, int, int], int]]:
    """
    slot 0 is always drums. Up to three non-drum lanes are selected by note count.

    Returns the notes that fit selected slots, slot metadata, and lane->slot mapping.
    """
    drum_notes = [n for n in notes if n.is_drum]
    melodic = [n for n in notes if not n.is_drum]

    lanes: dict[tuple[int, int, int], list[Note]] = defaultdict(list)
    for note in melodic:
        lanes[_lane_key(note)].append(note)

    ranked = sorted(
        lanes.items(),
        key=lambda kv: (-len(kv[1]), kv[0][0], kv[0][1], kv[0][2]),
    )
    chosen = ranked[: MAX_TRACKS - 1]

    slots = [
        SlotInfo(
            slot=0,
            is_drum=True,
            program=0,
            name="drums",
            source_track=None,
            source_channel=DRUM_MIDI_CHANNEL,
            note_count=len(drum_notes),
        )
    ]
    lane_to_slot: dict[tuple[int, int, int], int] = {}
    selected_notes = list(drum_notes)

    for slot, (key, lane_notes) in enumerate(chosen, start=1):
        source_track, source_channel, program = key
        exemplar = lane_notes[0]
        lane_to_slot[key] = slot
        selected_notes.extend(lane_notes)
        slots.append(
            SlotInfo(
                slot=slot,
                is_drum=False,
                program=program,
                name=exemplar.track_name or f"track_{slot}",
                source_track=source_track,
                source_channel=source_channel,
                note_count=len(lane_notes),
            )
        )

    # Pad metadata to exactly four slots.
    while len(slots) < MAX_TRACKS:
        slot = len(slots)
        slots.append(
            SlotInfo(
                slot=slot,
                is_drum=False,
                program=0,
                name=f"empty_{slot}",
                source_track=None,
                source_channel=None,
                note_count=0,
            )
        )

    return selected_notes, slots, lane_to_slot


def midi_to_roll(path: str | Path) -> tuple[np.ndarray, RollMeta]:
    """
    Encode MIDI to normalized float32 piano-roll.

    Shape: [bars, 8, 192, 128]

    1-based feature-channel meaning:
      1: drum onset velocity
      2: drum note-active mask
      3: track 2 onset velocity
      4: track 2 note-active mask
      5: track 3 onset velocity
      6: track 3 note-active mask
      7: track 4 onset velocity
      8: track 4 note-active mask

    onset_velocity: 0.0 for no onset, otherwise MIDI velocity / 127.0
    note_active:    0.0/1.0, active for every quantized audible step after CC64

    Therefore the entire tensor lies in [0, 1].
    """
    mid = mido.MidiFile(str(path))
    _validate_four_four(mid)
    notes = extract_notes(mid)

    ticks_per_step = (mid.ticks_per_beat * BEATS_PER_BAR) / STEPS_PER_BAR
    selected_notes, slots, lane_to_slot = _choose_slots(notes)

    encoded: list[tuple[int, int, int, int, int]] = []
    max_end = 0

    for note in selected_notes:
        if note.is_drum:
            slot = 0
        else:
            key = _lane_key(note)
            if key not in lane_to_slot:
                continue
            slot = lane_to_slot[key]

        start, end, pitch, velocity = _quantize_note(note, ticks_per_step)
        encoded.append((slot, start, end, pitch, velocity))
        max_end = max(max_end, end)

    num_bars = max(1, (max_end + STEPS_PER_BAR - 1) // STEPS_PER_BAR)
    # Build in flattened global time first.  Reshaping a transposed array can create
    # a copy, so we deliberately fill this contiguous buffer and convert afterwards.
    flat = np.zeros(
        (TOTAL_CHANNELS, num_bars * STEPS_PER_BAR, 128), dtype=np.float32
    )

    for slot, start, end, pitch, velocity in encoded:
        onset_ch = slot * 2
        sustain_ch = onset_ch + 1

        # Same-pitch simultaneous onsets are not separately representable in a piano roll.
        # Keep the stronger velocity, and use the union of activity.
        flat[onset_ch, start, pitch] = max(
            flat[onset_ch, start, pitch], np.float32(velocity / 127.0)
        )
        flat[sustain_ch, start:end, pitch] = 1

    roll = (
        flat.reshape(TOTAL_CHANNELS, num_bars, STEPS_PER_BAR, 128)
        .transpose(1, 0, 2, 3)
        .copy()
    )

    meta = RollMeta(
        ticks_per_beat=mid.ticks_per_beat,
        steps_per_bar=STEPS_PER_BAR,
        beats_per_bar=BEATS_PER_BAR,
        max_tracks=MAX_TRACKS,
        features_per_track=FEATURES_PER_TRACK,
        slots=slots,
    )
    return roll, meta


def roll_to_chunks(roll: np.ndarray) -> np.ndarray:
    """Split a bar-wise roll into chronological 24-step training chunks.

    Input shape:
        [num_bars, 8, 192, 128]

    Output shape:
        [num_bars * 8, 8, 24, 128]

    Chunk order is chronological:
        bar0/chunk0, bar0/chunk1, ..., bar0/chunk7,
        bar1/chunk0, ...
    """
    if roll.ndim != 4 or roll.shape[1:] != (TOTAL_CHANNELS, STEPS_PER_BAR, 128):
        raise ValueError(
            f"Expected [bars, {TOTAL_CHANNELS}, {STEPS_PER_BAR}, 128], got {roll.shape}."
        )

    num_bars = roll.shape[0]
    chunks = (
        roll.reshape(num_bars, TOTAL_CHANNELS, CHUNKS_PER_BAR, CHUNK_STEPS, 128)
        .transpose(0, 2, 1, 3, 4)
        .reshape(num_bars * CHUNKS_PER_BAR, TOTAL_CHANNELS, CHUNK_STEPS, 128)
        .copy()
    )
    return chunks.astype(np.float32, copy=False)


def chunks_to_roll(chunks: np.ndarray) -> np.ndarray:
    """Inverse of :func:`roll_to_chunks`.

    Input shape:
        [all_chunk_count, 8, 24, 128]

    Output shape:
        [num_bars, 8, 192, 128]

    ``all_chunk_count`` must be divisible by 8 because each 4/4 bar contains
    exactly eight 24-step chunks at 192 steps/bar.
    """
    if chunks.ndim != 4 or chunks.shape[1:] != (TOTAL_CHANNELS, CHUNK_STEPS, 128):
        raise ValueError(
            f"Expected [chunks, {TOTAL_CHANNELS}, {CHUNK_STEPS}, 128], got {chunks.shape}."
        )
    rem = chunks.shape[0] % CHUNKS_PER_BAR
    if rem != 0:
        pad_count = CHUNKS_PER_BAR - rem
        pad = np.zeros((pad_count, *chunks.shape[1:]), dtype=chunks.dtype)
        chunks = np.concatenate([chunks, pad], axis=0)

    num_bars = chunks.shape[0] // CHUNKS_PER_BAR
    roll = (
        chunks.reshape(num_bars, CHUNKS_PER_BAR, TOTAL_CHANNELS, CHUNK_STEPS, 128)
        .transpose(0, 2, 1, 3, 4)
        .reshape(num_bars, TOTAL_CHANNELS, STEPS_PER_BAR, 128)
        .copy()
    )
    return roll.astype(np.float32, copy=False)


def midi_to_chunks(path: str | Path) -> tuple[np.ndarray, RollMeta]:
    """Encode MIDI directly to normalized 24-step training chunks.
    Supports arbitrary time signatures (4/4, 3/4, 5/4, 6/8, 7/8, etc.) by quantizing
    relative to quarter-note beats (1 beat = 48 steps, 1 chunk = 24 steps = 0.5 beat).

    Returns:
        chunks: float32 array [all_chunk_count, 8, 24, 128], values in [0, 1]
        meta:    RollMeta used for optional reconstruction
    """
    mid = mido.MidiFile(str(path))
    _validate_four_four(mid)
    notes = extract_notes(mid)

    ticks_per_step = (mid.ticks_per_beat * BEATS_PER_BAR) / STEPS_PER_BAR
    selected_notes, slots, lane_to_slot = _choose_slots(notes)

    encoded: list[tuple[int, int, int, int, int]] = []
    max_end = 0

    for note in selected_notes:
        if note.is_drum:
            slot = 0
        else:
            key = _lane_key(note)
            if key not in lane_to_slot:
                continue
            slot = lane_to_slot[key]

        start, end, pitch, velocity = _quantize_note(note, ticks_per_step)
        encoded.append((slot, start, end, pitch, velocity))
        max_end = max(max_end, end)

    num_chunks = max(1, (max_end + CHUNK_STEPS - 1) // CHUNK_STEPS)
    total_steps = num_chunks * CHUNK_STEPS
    flat = np.zeros(
        (TOTAL_CHANNELS, total_steps, 128), dtype=np.float32
    )

    for slot, start, end, pitch, velocity in encoded:
        onset_ch = slot * 2
        sustain_ch = onset_ch + 1

        flat[onset_ch, start, pitch] = max(
            flat[onset_ch, start, pitch], np.float32(velocity / 127.0)
        )
        flat[sustain_ch, start:end, pitch] = 1

    chunks = (
        flat.reshape(TOTAL_CHANNELS, num_chunks, CHUNK_STEPS, 128)
        .transpose(1, 0, 2, 3)
        .copy()
    )

    meta = RollMeta(
        ticks_per_beat=mid.ticks_per_beat,
        steps_per_bar=STEPS_PER_BAR,
        beats_per_bar=BEATS_PER_BAR,
        max_tracks=MAX_TRACKS,
        features_per_track=FEATURES_PER_TRACK,
        slots=slots,
    )
    return chunks.astype(np.float32, copy=False), meta


def _flat_to_quantized_notes(flat: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """Extract (slot, start, end, pitch, velocity) from [TOTAL_CHANNELS, total_steps, 128]."""
    total_steps = flat.shape[1]
    result: list[tuple[int, int, int, int, int]] = []

    for slot in range(MAX_TRACKS):
        onset = flat[slot * 2]
        sustain = flat[slot * 2 + 1]

        onset_steps, pitches = np.nonzero(onset)
        order = np.argsort(onset_steps * 128 + pitches)

        for idx in order:
            start = int(onset_steps[idx])
            pitch = int(pitches[idx])
            velocity = int(np.clip(np.rint(float(onset[start, pitch]) * 127.0), 1, 127))

            next_onsets = np.flatnonzero(onset[start + 1 :, pitch])
            next_onset = (
                start + 1 + int(next_onsets[0]) if len(next_onsets) else total_steps
            )

            end = start + 1
            while end < next_onset and sustain[end, pitch] != 0:
                end += 1

            result.append((slot, start, end, pitch, velocity))

    return sorted(result)


def _roll_to_quantized_notes(roll: np.ndarray, meta: RollMeta) -> list[tuple[int, int, int, int, int]]:
    """Return (slot, start_step, end_step, pitch, velocity) tuples."""
    if roll.ndim != 4 or roll.shape[1:] != (TOTAL_CHANNELS, STEPS_PER_BAR, 128):
        raise ValueError(
            f"Expected [bars, {TOTAL_CHANNELS}, {STEPS_PER_BAR}, 128], got {roll.shape}."
        )

    flat = roll.transpose(1, 0, 2, 3).reshape(TOTAL_CHANNELS, -1, 128)
    return _flat_to_quantized_notes(flat)


def _notes_to_midi(notes: list[tuple[int, int, int, int, int]], meta: RollMeta, output_path: str | Path) -> None:
    tps = meta.ticks_per_step
    mid = mido.MidiFile(type=1, ticks_per_beat=meta.ticks_per_beat)

    conductor = mido.MidiTrack()
    conductor.append(mido.MetaMessage("track_name", name="conductor", time=0))
    conductor.append(
        mido.MetaMessage(
            "time_signature",
            numerator=4,
            denominator=4,
            clocks_per_click=24,
            notated_32nd_notes_per_beat=8,
            time=0,
        )
    )
    conductor.append(mido.MetaMessage("end_of_track", time=0))
    mid.tracks.append(conductor)

    melodic_channels = [0, 1, 2]
    by_slot: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
    for slot, start, end, pitch, velocity in notes:
        by_slot[slot].append((start, end, pitch, velocity))

    for slot in range(MAX_TRACKS):
        slot_info = meta.slots[slot]
        if slot != 0 and not by_slot.get(slot) and slot_info.note_count == 0:
            continue

        track = mido.MidiTrack()
        track.append(mido.MetaMessage("track_name", name=slot_info.name, time=0))

        if slot == 0:
            channel = DRUM_MIDI_CHANNEL
        else:
            channel = melodic_channels[slot - 1]
            track.append(
                mido.Message(
                    "program_change",
                    channel=channel,
                    program=int(slot_info.program),
                    time=0,
                )
            )

        events: list[tuple[int, int, mido.Message]] = []
        for start, end, pitch, velocity in by_slot.get(slot, []):
            start_tick = int(round(start * tps))
            end_tick = int(round(end * tps))
            events.append(
                (
                    start_tick,
                    1,
                    mido.Message(
                        "note_on", channel=channel, note=pitch, velocity=velocity, time=0
                    ),
                )
            )
            events.append(
                (
                    end_tick,
                    0,
                    mido.Message(
                        "note_off", channel=channel, note=pitch, velocity=0, time=0
                    ),
                )
            )

        events.sort(key=lambda x: (x[0], x[1], x[2].note))
        prev_tick = 0
        for abs_tick, _, msg in events:
            msg.time = abs_tick - prev_tick
            track.append(msg)
            prev_tick = abs_tick

        track.append(mido.MetaMessage("end_of_track", time=0))
        mid.tracks.append(track)

    mid.save(str(output_path))


def roll_to_midi(roll: np.ndarray, meta: RollMeta, output_path: str | Path) -> None:
    """Decode a roll back to a quantized 4/4 MIDI file."""
    notes = _roll_to_quantized_notes(roll, meta)
    _notes_to_midi(notes, meta, output_path)


def chunks_to_midi(chunks: np.ndarray, meta: RollMeta, output_path: str | Path) -> None:
    """Decode training chunks directly back to a quantized MIDI file (supports any chunk count / time signature)."""
    if chunks.ndim != 4 or chunks.shape[1:] != (TOTAL_CHANNELS, CHUNK_STEPS, 128):
        raise ValueError(
            f"Expected [chunks, {TOTAL_CHANNELS}, {CHUNK_STEPS}, 128], got {chunks.shape}."
        )
    flat = chunks.transpose(1, 0, 2, 3).reshape(TOTAL_CHANNELS, -1, 128)
    notes = _flat_to_quantized_notes(flat)
    _notes_to_midi(notes, meta, output_path)


def save_npz(path: str | Path, roll: np.ndarray, meta: RollMeta) -> None:
    meta_dict = asdict(meta)
    np.savez_compressed(path, roll=roll, meta=json.dumps(meta_dict, ensure_ascii=False))


def load_npz(path: str | Path) -> tuple[np.ndarray, RollMeta]:
    obj = np.load(path, allow_pickle=False)
    meta_dict = json.loads(str(obj["meta"]))
    meta_dict["slots"] = [SlotInfo(**x) for x in meta_dict["slots"]]
    return obj["roll"], RollMeta(**meta_dict)


def save_chunks_npz(path: str | Path, chunks: np.ndarray, meta: RollMeta) -> None:
    """Save training chunks plus reconstruction metadata."""
    meta_dict = asdict(meta)
    np.savez_compressed(
        path, chunks=chunks.astype(np.float32, copy=False),
        meta=json.dumps(meta_dict, ensure_ascii=False)
    )


def load_chunks_npz(path: str | Path) -> tuple[np.ndarray, RollMeta]:
    obj = np.load(path, allow_pickle=False)
    meta_dict = json.loads(str(obj["meta"]))
    meta_dict["slots"] = [SlotInfo(**x) for x in meta_dict["slots"]]
    chunks = obj["chunks"].astype(np.float32, copy=False)
    return chunks, RollMeta(**meta_dict)


def quantized_selected_notes(path: str | Path) -> list[tuple[int, int, int, int, int]]:
    """Ground-truth selected notes after applying the encoder's quantizer/slot policy."""
    mid = mido.MidiFile(str(path))
    _validate_four_four(mid)
    notes = extract_notes(mid)
    tps = (mid.ticks_per_beat * BEATS_PER_BAR) / STEPS_PER_BAR
    selected, _, lane_to_slot = _choose_slots(notes)

    out = []
    for n in selected:
        slot = 0 if n.is_drum else lane_to_slot.get(_lane_key(n))
        if slot is None:
            continue
        s, e, p, v = _quantize_note(n, tps)
        out.append((slot, s, e, p, v))
    return sorted(out)


def verify_roundtrip(input_midi: str | Path, decoded_midi: str | Path) -> None:
    """Assert encode/decode preserves every representable quantized note."""
    roll, meta = midi_to_roll(input_midi)
    expected = quantized_selected_notes(input_midi)
    represented = _roll_to_quantized_notes(roll, meta)

    if represented != expected:
        missing = [x for x in expected if x not in represented]
        extra = [x for x in represented if x not in expected]
        raise AssertionError(
            "The source contains collisions not representable by this piano roll. "
            f"Missing={missing[:10]}, extra={extra[:10]}"
        )

    roll_to_midi(roll, meta, decoded_midi)
    roll2, _ = midi_to_roll(decoded_midi)

    if not np.array_equal(roll, roll2):
        diff = int(np.count_nonzero(roll != roll2))
        raise AssertionError(f"Round-trip roll mismatch: {diff} cells differ.")


def _add_note(events: list[tuple[int, int, mido.Message]], start: int, end: int,
              channel: int, pitch: int, velocity: int) -> None:
    events.append((start, 1, mido.Message("note_on", channel=channel, note=pitch,
                                         velocity=velocity, time=0)))
    events.append((end, 0, mido.Message("note_off", channel=channel, note=pitch,
                                        velocity=0, time=0)))


def make_synthetic_test_midi(path: str | Path) -> None:
    """Create a test MIDI with drums + 3 melodic tracks and cross-bar notes."""
    ppq = 480
    mid = mido.MidiFile(type=1, ticks_per_beat=ppq)

    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("track_name", name="conductor", time=0))
    meta.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))
    mid.tracks.append(meta)

    specs = [
        ("drums", DRUM_MIDI_CHANNEL, 0),
        ("piano", 0, 0),
        ("bass", 1, 33),
        ("strings", 2, 48),
    ]

    for name, channel, program in specs:
        tr = mido.MidiTrack()
        tr.append(mido.MetaMessage("track_name", name=name, time=0))
        if channel != DRUM_MIDI_CHANNEL:
            tr.append(mido.Message("program_change", channel=channel,
                                   program=program, time=0))

        events: list[tuple[int, int, mido.Message]] = []
        if name == "drums":
            # Kick/snare/hat spread over more than two bars.
            for beat in range(10):
                t = beat * ppq
                _add_note(events, t, t + 120, channel, 36 if beat % 2 == 0 else 38,
                          100 - (beat % 3) * 7)
                _add_note(events, t + 240, t + 330, channel, 42, 72)
        elif name == "piano":
            _add_note(events, 0, 960, channel, 60, 96)
            _add_note(events, 960, 1800, channel, 64, 88)
            # Cross the bar boundary at 1920 ticks.
            _add_note(events, 1800, 2520, channel, 67, 103)
            # Adjacent repeated same pitch: boundary must be preserved by onset channel.
            _add_note(events, 2520, 2760, channel, 67, 81)
        elif name == "bass":
            for i, p in enumerate([36, 38, 41, 43, 36, 43]):
                s = i * 720
                _add_note(events, s, s + 480, channel, p, 84 + i)
        else:
            _add_note(events, 480, 2100, channel, 55, 75)
            _add_note(events, 2100, 3900, channel, 57, 79)

        events.sort(key=lambda x: (x[0], x[1], x[2].note))
        prev = 0
        for tick, _, msg in events:
            msg.time = tick - prev
            tr.append(msg)
            prev = tick
        mid.tracks.append(tr)

    mid.save(str(path))



