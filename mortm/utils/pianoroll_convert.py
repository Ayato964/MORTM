import pretty_midi
import numpy as np
import math

def get_pianoroll(midi_path: str, ticks_per_measure: int, instrument_programs: list[int]):
    """
    Converts a MIDI file to a piano roll representation.

    Args:
        midi_path (str): Path to the MIDI file.
        ticks_per_measure (int): The number of time steps (ticks) per measure in the output piano roll.
        instrument_programs (list[int]): A list of MIDI program numbers for the instruments to include.

    Returns:
        numpy.ndarray: A NumPy array representing the piano roll.
                       The shape is (total_ticks, 128, num_instruments * 2).
                       For each instrument, there are two channels:
                       1. A binary pitch channel (note on/off).
                       2. A velocity channel, with values normalized to the range [0, 1].
                       Returns an empty array if the MIDI file cannot be processed or has no matching instruments.
    """
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        print(f"Error loading MIDI file: {e}")
        return np.array([])

    # Assuming 4/4 time signature for measure calculation if none are present.
    try:
        # Get downbeats to count measures.
        downbeats = pm.get_downbeats()
        if len(downbeats) == 0:
            beats = pm.get_beats()
            ts = pm.time_signature_changes[0] if pm.time_signature_changes else pretty_midi.TimeSignature(4, 4, 0)
            beats_per_measure = ts.numerator
            num_measures = math.ceil(len(beats) / beats_per_measure) if beats_per_measure > 0 else 0
        else:
            num_measures = len(downbeats)
        
        if num_measures == 0:
            num_measures = 1

    except Exception:
        end_time = pm.get_end_time()
        tempo = pm.get_tempo_changes()[1][0] if len(pm.get_tempo_changes()[1]) > 0 else 120.0
        beats_per_minute = tempo
        beats_per_second = beats_per_minute / 60.0
        total_beats = end_time * beats_per_second
        beats_per_measure = 4 # Assume 4/4
        num_measures = math.ceil(total_beats / beats_per_measure)
        if num_measures == 0:
            num_measures = 1

    total_ticks = num_measures * ticks_per_measure
    end_time = pm.get_end_time()
    
    num_channels = len(instrument_programs) * 2

    if end_time == 0:
        return np.zeros((total_ticks, 128, num_channels), dtype=np.float32)

    fs = total_ticks / end_time if end_time > 0 else 0

    all_instrument_rolls = []

    for program in instrument_programs:
        instrument_velocity_roll = np.zeros((total_ticks, 128), dtype=np.float32)
        
        matching_instruments = [inst for inst in pm.instruments if inst.program == program and not inst.is_drum]
        
        if 112 <= program <= 127:
             matching_instruments.extend([inst for inst in pm.instruments if inst.is_drum and inst not in matching_instruments])

        for instrument in matching_instruments:
            if fs > 0:
                velocity_roll = instrument.get_piano_roll(fs=fs).astype(np.float32)
                velocity_roll = velocity_roll.T
            else:
                velocity_roll = np.zeros((0, 128), dtype=np.float32)
            
            current_len = velocity_roll.shape[0]
            if current_len < total_ticks:
                padded_roll = np.zeros((total_ticks, 128), dtype=np.float32)
                padded_roll[:current_len, :] = velocity_roll
                velocity_roll = padded_roll
            elif current_len > total_ticks:
                velocity_roll = velocity_roll[:total_ticks, :]
            
            instrument_velocity_roll = np.maximum(instrument_velocity_roll, velocity_roll)

        instrument_pitch_roll = (instrument_velocity_roll > 0).astype(np.float32)
        
        normalized_velocity_roll = instrument_velocity_roll / 127.0
        
        all_instrument_rolls.append(instrument_pitch_roll)
        all_instrument_rolls.append(normalized_velocity_roll)

    if not all_instrument_rolls:
        return np.zeros((total_ticks, 128, num_channels), dtype=np.float32)

    final_pianoroll = np.stack(all_instrument_rolls, axis=-1)

    return final_pianoroll


def pianoroll_to_midi(pianoroll: np.ndarray, output_midi_path: str, instrument_programs: list[int], ticks_per_measure: int, tempo: int = 120):
    """
    Converts a piano roll representation back to a MIDI file.

    Args:
        pianoroll (np.ndarray): A NumPy array representing the piano roll.
                                Shape should be (total_ticks, 128, num_instruments * 2).
        output_midi_path (str): Path to save the output MIDI file.
        instrument_programs (list[int]): A list of MIDI program numbers for the instruments.
        ticks_per_measure (int): The number of time steps (ticks) per measure used to generate the pianoroll.
        tempo (int, optional): The tempo (beats per minute) for the output MIDI. Defaults to 120.
    """
    if pianoroll.ndim != 3 or pianoroll.shape[2] != len(instrument_programs) * 2:
        print("Error: Invalid pianoroll shape.")
        return

    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    total_ticks = pianoroll.shape[0]
    
    # Assuming 4/4 time signature
    seconds_per_beat = 60.0 / tempo
    seconds_per_tick = (seconds_per_beat * 4) / ticks_per_measure

    for i, program in enumerate(instrument_programs):
        is_drum = 112 <= program <= 127
        instrument = pretty_midi.Instrument(program=program, is_drum=is_drum)
        
        pitch_roll = pianoroll[:, :, i * 2]
        velo_roll = pianoroll[:, :, i * 2 + 1]

        for pitch in range(128):
            note_start_tick = -1
            velocity_val = 0

            for tick in range(total_ticks):
                is_on = pitch_roll[tick, pitch] > 0
                
                if is_on and note_start_tick == -1:
                    note_start_tick = tick
                    raw_velocity = int(velo_roll[tick, pitch] * 127)
                    velocity_val = max(1, raw_velocity) # Velocity must be > 0

                elif not is_on and note_start_tick != -1:
                    end_tick = tick
                    
                    start_time = note_start_tick * seconds_per_tick
                    end_time = end_tick * seconds_per_tick
                    
                    note = pretty_midi.Note(
                        velocity=velocity_val,
                        pitch=pitch,
                        start=start_time,
                        end=end_time
                    )
                    instrument.notes.append(note)
                    note_start_tick = -1

            # If a note is still on at the end of the pianoroll
            if note_start_tick != -1:
                end_tick = total_ticks
                start_time = note_start_tick * seconds_per_tick
                end_time = end_tick * seconds_per_tick
                note = pretty_midi.Note(
                    velocity=velocity_val,
                    pitch=pitch,
                    start=start_time,
                    end=end_time
                )
                instrument.notes.append(note)

        pm.instruments.append(instrument)

    try:
        pm.write(output_midi_path)
        print(f"MIDI file saved to {output_midi_path}")
    except Exception as e:
        print(f"Error writing MIDI file: {e}")
