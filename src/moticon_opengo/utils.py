import os
from enum import Enum
from typing import List, Optional, Union

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


def is_not_null(value) -> bool:
    """
    Return True if the [value] is a proper number, not None or NAN.
    """
    if value is None:
        return False

    if (
        not isinstance(value, float)
        and not isinstance(value, int)
        and not isinstance(value, np.float32)
        and not isinstance(value, np.float64)
    ):
        return False

    if np.isnan(value):
        return False

    return True


def average_null_safe(
    a: List[Union[float, None, np.float32, np.float64]],
    weights: Optional[List[Union[float, None, np.float32, np.float64]]] = None,
) -> Optional[float]:
    """
    Average a list of values, while excluding None values and Numpy nan values.
    Optionally, compute weighted average using weights.
    """
    non_null, non_null_weights = list(), list()

    if weights is None:
        weights = np.ones(shape=[len(a)])

    for x, w in zip(a, weights):
        if x is None:
            continue

        if np.isnan(x):
            continue

        non_null.append(x)
        non_null_weights.append(w)

    if non_null:
        # Note: In np.average(), sum(weights) is rescaled to 1. Consequently,
        # if a value in 'a' is Null, the weights of the remaining values will
        # sum up to 1 (as desired). No need to handle this manually here.
        return np.average(non_null, weights=non_null_weights)

    return None
