from typing import List

class Chord:
    def __init__(self, chord: str, time_stamp):
        self.chord: str = chord
        self.time_stamp = time_stamp
        self.is_called = False

class ChordMidi:
    def __init__(self, chords: List[str], time_stamps: List[float]):
        self.chords: List[Chord] = list()
        for chord, time_stamp in zip(chords, time_stamps):
            self.chords.append(Chord(chord, time_stamp))

        self.chords = sorted(self.chords, key=lambda x: x.time_stamp)

    def get_chord(self, time: float, is_final_search=False):
        for i in range(len(self.chords) - 1):
            if self.chords[i].time_stamp <= time < self.chords[i + 1].time_stamp:
                if self.chords[i].is_called:
                    return None
                else:
                    self.chords[i].is_called = is_final_search
                    return self.chords[i]

        if self.chords[-1].time_stamp <= time:
            if self.chords[-1].is_called:
                return None
            else:
                self.chords[-1].is_called = is_final_search
                return self.chords[-1]
        return None

    def reset(self):
        for chord in self.chords:
            chord.is_called = False

