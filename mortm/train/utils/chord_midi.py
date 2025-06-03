from typing import List

class Chord:
    def __init__(self, chord: str, time_stamp):
        self.chord: str = chord
        self.time_stamp = time_stamp
        self.is_called = False


class ChordMidi:

    def __len__(self):
        return len(self.chords)

    def __getitem__(self, item):
        return self.chords[item]

    def __init__(self, chords: List[str], time_stamps: List[float]):
        self.chords: List[Chord] = list()
        for chord, time_stamp in zip(chords, time_stamps):
            self.chords.append(Chord(chord, time_stamp))

        self.chords = sorted(self.chords, key=lambda x: x.time_stamp)

    def get_chord(self, time: float, is_final_search=False):
        for i in range(len(self.chords) - 1):
            if self.chords[i].time_stamp <= time < self.chords[i + 1].time_stamp:
                return self.chords[i], i

        if self.chords[-1].time_stamp <= time:
            return self.chords[-1], len(self.chords) - 1

        return None

    def get_chord_by_index(self, index: int):
        if 0 <= index < len(self.chords):
            return self.chords[index]
        return None

    def reset(self):
        for chord in self.chords:
            chord.is_called = False

    def sort(self, time: int):
        for c in self.chords:
            c.time_stamp -= time

