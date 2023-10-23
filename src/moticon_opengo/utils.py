import os
from enum import Enum
from typing import List, Optional

import numpy as np


class Side(Enum):
    LEFT = 1
    RIGHT = 2

    @property
    def other(self):
        if self == Side.LEFT:
            return Side.RIGHT

        return Side.LEFT

    @staticmethod
    def from_string(side_string: str):
        if side_string.lower() == "left":
            return Side.LEFT
        elif side_string.lower() == "right":
            return Side.RIGHT

        return None


class FileIterator:
    def __init__(self, folder: str, filename_extension: str = ".txt"):
        self.folder: str = folder
        self.filename_extension: str = filename_extension

        self.files: List[str] = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.endswith(self.filename_extension)
        ]
        self.index: int = 0
        self._check_files()
        self._sort_files()

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.files):
            self.index += 1
            return os.path.join(self.folder, self.files[self.index - 1])

        raise StopIteration

    def _check_files(self):
        """
        Plausibility check removing non-matching files
        """
        pass

    def _sort_files(self):
        self.files.sort()


class SideData(object):
    def __init__(self, side: Side):
        self.side: Side = side
        self.pressure: Optional[np.ndarray] = None
        self.acceleration: Optional[np.ndarray] = None
        self.angular: Optional[np.ndarray] = None
        self.total_force: Optional[np.ndarray] = None
        self.cop: Optional[np.ndarray] = None
        self.cop_velocity: Optional[np.ndarray] = None


class Step(object):
    def __init__(self, side: Side):
        self.side: Optional[Side] = side


class StepIterator(object):
    def __init__(self):
        self.index: int = 0
        self.steps = list()

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.steps):
            self.index += 1
            return self.steps[self.index - 1]

        raise StopIteration
