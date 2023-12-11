import re
from datetime import datetime
from typing import Dict, List, Optional, Union

import numpy as np

from moticon_opengo.utils import FileIterator, Side, SideData, Step


class TxtExportFileIterator(FileIterator):
    def __init__(self, folder: str):
        super().__init__(folder, ".txt")

    def _check_files(self):
        """
        Plausibility check removing files which are obviously no OpenGo text export
        files.
        """
        for fname in self.files.copy():
            comment_fields: List[str] = []

            with open(fname, "r") as f:
                for line in f:
                    if not line.startswith("#"):
                        break

                    splits: List[str] = line[1:].strip().split(":", 1)

                    if len(splits) > 1:
                        comment_fields.append(splits[0])

            if (
                "Start time" not in comment_fields
                or "Duration" not in comment_fields
                or "Sensor insoles" not in comment_fields
            ):
                self.files.remove(fname)


class TxtExportStep(Step):
    def __init__(self, side: Side, heel_strike_idx: int, toe_off_idx: int):
        super().__init__(side)

        self.heel_strike_idx: Optional[int] = heel_strike_idx
        self.toe_off_idx: Optional[int] = toe_off_idx


class Measurement(object):
    def __init__(self, fname: str, trim: bool = True, fill_method: str = "duplicate"):
        """
        Represents a sensor insole measurement constructed from a text export
        file [fname].
        In general, the left/right data will not start/stop at the same point in
        time; if [trim], the resulting single-side data at the begin and end of
        the data is removed, such that there is only bipedal data. The left and
        right side data is merged by the text export to a common time axis,
        which may result in sporadic rows only having data on one side; with
        [fill_method] (duplicate/drop/keep) equal to "duplicate", missing data
        is filled by duplicating the previous sample on the corresponding side;
        with "drop", the entire sample is dropped on both sides; "keep" simply
        keeps the original data including gaps. Caution: The "drop" fill method
        may also drop step detection events, which affects the [steps] list.
        """
        self.fname: str = fname
        self.trim: bool = trim
        self.fill_method: str = fill_method

        self.start_time: Optional[datetime] = None
        self.duration: Optional[str] = None
        self.serial_numbers: List[str] = list()
        self.sensor_insole_size: Optional[int] = None
        self.recording_type: Optional[str] = None
        self.name: Optional[str] = None
        self.comment: Optional[str] = None
        self.tag: Optional[str] = None
        self.channel_names: List[str] = list()

        self.data: Optional[np.ndarray] = None
        self.time: Optional[np.ndarray] = None
        self.extra_data: Dict[str, np.ndarray] = dict()
        self.sides: List[Side] = list()
        self.side_data: Dict[Side, SideData] = dict()
        self.steps: List[Step] = list()
        self._contains_steps_channel: bool = False
        self._first_full_data_column: Dict[Side, int] = dict()
        self._full_data_columns: Dict[Side, List[int]] = dict()

        self._load_data()

    def _load_data(self):
        self._parse_header()
        self.data = np.genfromtxt(self.fname, delimiter="\t", dtype=None, comments="#")
        self._apply_trim()
        self._apply_fill_method()
        self._load_channel_data()

    def _parse_header(self):
        with open(self.fname, "r") as f:
            for line in f:
                if not line.startswith("#"):
                    break

                line = line[2:].strip()

                def strip_name(line):
                    return line.split(":", maxsplit=1)[-1].strip()

                if line.startswith("Start time:"):
                    self.start_time = datetime.strptime(
                        line[12:], "%d.%m.%Y %H:%M:%S.%f"
                    )
                elif line.startswith("Duration:"):
                    self.duration = strip_name(line)
                elif line.startswith("Sensor insoles:"):
                    self.serial_numbers = strip_name(line).split(", ")
                elif line.startswith("Size:"):
                    self.sensor_insole_size = int(strip_name(line))
                elif line.startswith("Recording type:"):
                    self.recording_type = strip_name(line)
                elif line.startswith("Name:"):
                    self.name = strip_name(line)
                elif line.startswith("Notes:"):
                    self.comment = strip_name(line)
                elif line.startswith("Tag:"):
                    self.tag = strip_name(line)
                else:
                    self.channel_names = line.split("\t")

        # Core sensor insole data channel columns always exported before sparse event
        # channels. Later cannot be used for trimming/filling, as there are always many
        # nan entries.
        for side in Side:
            col_idx: List[int] = [
                i
                for i, n in enumerate(self.channel_names)
                if n.startswith(f"{side.name.lower()} ")
            ]

            if col_idx:
                self._first_full_data_column[side] = col_idx[0]
                self._full_data_columns[side] = col_idx
                self.sides.append(side)

    def _apply_trim(self):
        if not self.trim:
            return

        if self.data is None:
            return

        nan_rows: np.ndarray = np.isnan(
            self.data[:, list(self._first_full_data_column.values())]
        ).any(axis=1)

        idx_start: int = 0
        idx_stop: int = self.data.shape[0] - 1

        for idx_start in range(self.data.shape[0]):
            if not nan_rows[idx_start]:
                break

        for idx_stop in range(self.data.shape[0] - 1, -1, -1):
            if not nan_rows[idx_stop]:
                break

        # No data remains.
        if idx_start > idx_stop:
            self.data = None
            return

        self.data = self.data[idx_start : idx_stop + 1]

    def _apply_fill_method(self):
        if self.data is None or self.fill_method == "keep":
            return

        if self.fill_method == "drop":
            nan_rows: np.ndarray = np.isnan(
                self.data[:, list(self._first_full_data_column.values())]
            ).any(axis=1)
            self.data = self.data[~nan_rows]

        elif self.fill_method == "duplicate":
            for side in self.sides:
                latest_non_nan_row: Optional[int] = None
                nan_rows: np.ndarray = np.isnan(
                    self.data[:, self._first_full_data_column[side]]
                )

                for row_idx in range(self.data.shape[0]):
                    if not nan_rows[row_idx]:
                        latest_non_nan_row = row_idx
                    elif latest_non_nan_row is not None:
                        for col_idx in self._full_data_columns[side]:
                            self.data[row_idx, col_idx] = self.data[
                                latest_non_nan_row, col_idx
                            ]

    def _load_channel_data(self):
        if self.data is None:
            return

        channel_groups = [
            x[: x.find("[")] if x.find("[") != -1 else x for x in self.channel_names
        ]
        channel_groups: List[str] = [
            re.sub(r"\s(X|Y|Z|\d+)\s*$", "", x) for x in channel_groups
        ]
        group_selectors: List[List[Union[str, Optional[int]]]] = [
            [channel_groups[0], 0, None]
        ]

        for i in range(1, len(channel_groups)):
            if channel_groups[i] != channel_groups[i - 1]:
                group_selectors[-1][2] = i
                group_selectors.append([channel_groups[i], i, None])

        group_selectors[-1][2] = len(channel_groups)

        if any([x.startswith("left ") for x in channel_groups]):
            self.side_data[Side.LEFT] = SideData(Side.LEFT)

        if any([x.startswith("right ") for x in channel_groups]):
            self.side_data[Side.RIGHT] = SideData(Side.RIGHT)

        for sel in group_selectors:
            column_title: str = sel[0]
            side: Side = Side.from_string(column_title.split()[0])
            group = " ".join(column_title.split()[1:])

            if column_title == "time":
                self.time = self.data[:, sel[1]]
            elif group == "pressure":
                self.side_data[side].pressure = self.data[:, sel[1] : sel[2]]
            elif group == "acceleration":
                self.side_data[side].acceleration = self.data[:, sel[1] : sel[2]]
            elif group == "angular":
                self.side_data[side].angular = self.data[:, sel[1] : sel[2]]
            elif group == "total force":
                self.side_data[side].total_force = self.data[:, sel[1]]
            elif group == "center of pressure":
                self.side_data[side].cop = self.data[:, sel[1] : sel[2]]
            elif group == "COP velocity":
                self.side_data[side].cop_velocity = self.data[:, sel[1]]
            elif group == "steps":
                self._contains_steps_channel = True
                self.extract_step_events(side, self.data[:, sel[1]])
            else:
                if sel[1] == sel[2] - 1:
                    self.extra_data[column_title] = self.data[:, sel[1]]
                else:
                    self.extra_data[column_title] = self.data[:, sel[1] : sel[2]]

    def extract_step_events(self, side: Side, channel_data: np.ndarray):
        heel_strike_value: int = 11 if side == Side.LEFT else 21
        toe_off_value: int = 15 if side == Side.LEFT else 25
        heel_strike_idx: np.ndarray = np.where(channel_data == heel_strike_value)[0]
        toe_off_idx: np.ndarray = np.where(channel_data == toe_off_value)[0]

        for h, t in zip(heel_strike_idx, toe_off_idx):
            self.steps.append(TxtExportStep(side, h, t))

    @property
    def has_steps(self) -> bool:
        return len(self.steps.steps) > 0

    @property
    def has_step_channel(self) -> bool:
        return self._contains_steps_channel

    def __str__(self) -> str:
        return f"OpenGo Text Export <{self.fname}> {self.name} ({self.tag})"
