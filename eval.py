from mortm.mortm import MORTM
from abc import abstractmethod


class AbstractEval:
    def __init__(self, mortm:MORTM):
        self.mortm = mortm

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class EvalPianoRoll(AbstractEval):
    def __call__(self, *args, **kwargs):

        pass