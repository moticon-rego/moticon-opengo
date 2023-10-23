import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from moticon_opengo.txt_export import Measurement, TxtExportFileIterator


def main():
    testdata_dir: str = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "tests", "testdata")
    )
    colors: List[str] = [c["color"] for c in plt.rcParams["axes.prop_cycle"]]

    for i, fname in enumerate(TxtExportFileIterator(testdata_dir)):
        meas: Measurement = Measurement(fname)

        for step in meas.steps:
            h: int = step.heel_strike_idx - 5
            t: int = step.toe_off_idx + 5
            time: np.array = meas.time[h:t] - meas.time[h]
            total_force: np.array = meas.side_data[step.side].total_force[h:t]
            plt.plot(time, total_force, label=meas.tag, color=colors[i])

    plt.title("Total force curve of all detected steps")
    plt.xlabel("Time (s)")
    plt.ylabel("Total force (N)")
    plt.grid(True)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())  # Avoid legend entry per curve
    plt.show()


if __name__ == "__main__":
    main()
