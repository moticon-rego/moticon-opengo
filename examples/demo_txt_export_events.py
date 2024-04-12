import os

import matplotlib.pyplot as plt
import numpy as np

from moticon_opengo.txt_export import Measurement
from moticon_opengo.utils import Side


def cop_trace_length(cop: np.array) -> np.float64:
    # Remove zero-valued COPs (low force).
    cop_ = np.empty(shape=(0, 2))

    for i in range(cop.shape[0]):
        if cop[i, 0] != 0 or cop[i, 1] != 0:
            cop_ = np.append(cop_, [cop[i, :]], axis=0)

    return np.sum(np.sqrt(np.sum(np.diff(cop_, axis=0) ** 2, axis=1)))


def main():
    meas: Measurement = Measurement(
        os.path.join(
            os.path.dirname(__file__),
            "../tests/testdata/walk_data_with_events/text_export.txt",
        )
    )

    for side, color in zip(Side, ["blue", "brown"]):
        group_name: str = f"{side.name.capitalize()} Events"
        plt.plot(meas.time, meas.side_data[side].total_force, color=color)

        for event_pair in meas.get_event_pairs(
            start_group_names=[group_name], stop_group_names=[group_name]
        ):
            cop = meas.side_data[side].cop[event_pair[0].index : event_pair[1].index, :]
            print(f"{side.name.capitalize()} COP path length: {cop_trace_length(cop)}")

            plt.axvspan(
                xmin=event_pair[0].time,
                xmax=event_pair[1].time,
                facecolor=color,
                alpha=0.3,
            )

    plt.xlabel("Time (s)")
    plt.ylabel("Total force (N)")
    plt.show()


if __name__ == "__main__":
    main()
