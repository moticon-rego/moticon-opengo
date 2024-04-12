import re
from dataclasses import dataclass
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


@dataclass
class TxtExportEvent(object):
    """
    Represents an event which was set in the the OpenGo software, and which got
    exported by the text export. The [time] is the relative time (corresponds to the
    "time" column), the [index] is the row index according to the numeric
    measurement data (i.e. you can use this [index] for slicing data).
    """

    group_name: str
    event_name: str
    value: int
    time: float
    index: int


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
        self._event_label_column_idx: List[int] = list()  # Each followed by value col.
        self._value_column_idx: List[int] = list()
        self.events: List[TxtExportEvent] = list()

        self._load_data()

    def _load_data(self):
        self._parse_header()
        self._load_from_file()
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

        self._detect_event_columns()

        # Core sensor insole data channel columns always exported before sparse event
        # channels. Later cannot be used for trimming/filling, as there are always many
        # nan entries.
        for side in Side:
            col_idx: List[int] = [
                i
                for i, n in enumerate(self.channel_names)
                if n.startswith(f"{side.name.lower()} ")
                and i not in self._event_label_column_idx
                and i not in self._value_column_idx
            ]

            if col_idx:
                self._first_full_data_column[side] = col_idx[0]
                self._full_data_columns[side] = col_idx
                self.sides.append(side)

    def _detect_event_columns(self):
        # Events are stored by two columns: An event label column named e.g.
        # "My Events label[]", followed by an event value column named "My Events[]".
        self._event_label_column_idx = [
            i
            for i, (name, next_name) in enumerate(
                zip(self.channel_names[:-1], self.channel_names[1:])
            )
            if name[:-2] == next_name[:-2] + " label"
        ]

        self._value_column_idx = [x + 1 for x in self._event_label_column_idx]

    def _load_from_file(self):
        # Load full file including event columns.
        dtype = list()

        for i, name in enumerate(self.channel_names):
            if i not in self._event_label_column_idx:
                dtype.append((name, float))
            else:
                dtype.append((name, "U100"))

        all_data = np.genfromtxt(self.fname, delimiter="\t", dtype=dtype, comments="#")

        # Extract the contained events, and sort by time.
        for col_idx in self._event_label_column_idx:
            for i, data_line in enumerate(all_data):
                if data_line[col_idx]:
                    self.events.append(
                        TxtExportEvent(
                            group_name=self.channel_names[col_idx + 1][:-2],
                            event_name=data_line[col_idx],
                            value=data_line[col_idx + 1],
                            time=data_line[0],
                            index=i,
                        )
                    )

        self.events = sorted(self.events, key=lambda x: x.time)

        # Load file again, this time without event columns.
        num_cols: int = len(self.channel_names)
        event_col_idx: List[int] = sorted(
            self._event_label_column_idx + self._value_column_idx
        )
        usecols: List[int] = [i for i in range(num_cols) if i not in event_col_idx]

        self.data = np.genfromtxt(
            self.fname, delimiter="\t", usecols=usecols, comments="#"
        )

        # Post-fix column indexes, since [self.data] no longer contains event columns.
        self.channel_names = [self.channel_names[i] for i in usecols]
        map: Dict[int, int] = {idx: i for i, idx in enumerate(usecols)}

        for side in Side:
            self._first_full_data_column[side] = map[self._first_full_data_column[side]]
            self._full_data_columns[side] = [
                map[i] for i in self._full_data_columns[side]
            ]

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
            self.events = list()
            return

        self.data = self.data[idx_start : idx_stop + 1]

        self.events = [
            x for x in self.events if x.index >= idx_start and x.index <= idx_stop
        ]

    def _apply_fill_method(self):
        if self.data is None or self.fill_method == "keep":
            return

        if self.fill_method == "drop":
            nan_rows: np.ndarray = np.isnan(
                self.data[:, list(self._first_full_data_column.values())]
            ).any(axis=1)
            self.data = self.data[~nan_rows]

            # This will keep all events, but adjust the event's [index] value so it can
            # still be used for data slicing. However, the event's [time] is kept
            # unchanged. This may result in multiple events having the same [index] but
            # different [time].
            dropped_row_idx = [i for i, x in enumerate(nan_rows) if x]

            for event in self.events:
                dropped_rows_prior_event = len(
                    [i for i in dropped_row_idx if i <= event.index]
                )
                event.index = max(0, event.index - dropped_rows_prior_event)

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

        channel_groups: List[str] = [
            x[: x.find("[")] if x.find("[") != -1 else x for x in self.channel_names
        ]
        channel_groups = [re.sub(r"\s(X|Y|Z|\d+)\s*$", "", x) for x in channel_groups]
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
            group: str = " ".join(column_title.split()[1:])

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

    @property
    def event_groups(self) -> List[str]:
        return sorted(list(set([x.group_name for x in self.events])))

    def get_events(
        self,
        group_names: List[str] = list(),
        event_names: List[str] = list(),
        values: List[int] = list(),
    ) -> List[TxtExportEvent]:
        """
        Return the events which, if given, match the [group_names], [event_names], and
        [values].
        """
        return [
            x
            for x in self.events
            if (not group_names or x.group_name in group_names)
            and (not event_names or x.event_name in event_names)
            and (not values or x.value in values)
        ]

    def get_event_pairs(
        self,
        start_group_names: List[str] = list(),
        start_event_names: List[str] = list(),
        start_values: List[int] = list(),
        stop_group_names: List[str] = list(),
        stop_event_names: List[str] = list(),
        stop_values: List[int] = list(),
    ) -> List[List[TxtExportEvent]]:
        """
        Return pairs of events for which, if given, the first event matches the
        [start_group_names], [start_event_names], and [start_values], and the second
        event matches the [stop_group_names], [stop_event_names], and [stop_values].
        This is a greedy algorithm, which simply takes the next matching event beginning
        with the first start event, skipping potential orphan events.
        """
        start_events: List[TxtExportEvent] = self.get_events(
            start_group_names, start_event_names, start_values
        )

        stop_events: List[TxtExportEvent] = self.get_events(
            stop_group_names, stop_event_names, stop_values
        )

        if not start_events or not stop_events:
            return list()

        event_pairs: List[List[TxtExportEvent]] = list()
        prev_time = -np.inf

        # We can assume that the events are sorted in time.
        while True:
            while start_events and start_events[0].time <= prev_time:
                del start_events[0]

            if not start_events:
                break

            prev_time = start_events[0].time

            while stop_events and stop_events[0].time <= prev_time:
                del stop_events[0]

            if not stop_events:
                break

            prev_time = stop_events[0].time
            event_pairs.append([start_events[0], stop_events[0]])

        return event_pairs

    def __str__(self) -> str:
        return f"OpenGo Text Export <{self.fname}> {self.name} ({self.tag})"
