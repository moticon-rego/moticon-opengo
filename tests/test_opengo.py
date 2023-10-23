import os
import unittest
from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from moticon_opengo.txt_export import Measurement, TxtExportFileIterator

# import filecmp

dir = os.path.dirname(os.path.abspath(__file__))


class TestOpenGo(unittest.TestCase):
    @property
    def testdir(self):
        return os.path.join(dir, "testdata")

    def test_import_data(self):
        meas: Measurement = Measurement(os.path.join(self.testdir, "230927_102114.txt"))
        self.assertEqual(
            meas.start_time,
            datetime(
                year=2023,
                month=9,
                day=27,
                hour=10,
                minute=21,
                second=14,
                microsecond=922000,
            ),
        )
        self.assertEqual(meas.duration, "00:17.496")
        self.assertEqual(meas.recording_type, "normal")
        self.assertEqual(meas.tag, "normal")
        self.assertEqual(meas.comment, "Some walking steps")
        self.assertEqual(meas.name, "230927_10:21:14")
        self.assertEqual(meas.sensor_insole_size, 7)
        self.assertListEqual(meas.serial_numbers, ["Left SN5968", "Right SN9171"])

    def test_import_and_plot_steps(self):
        colors: List[str] = [c["color"] for c in plt.rcParams["axes.prop_cycle"]]

        for i, fname in enumerate(TxtExportFileIterator(os.path.join(dir, "testdata"))):
            meas: Measurement = Measurement(fname)

            if not meas.has_step_channel:
                print(f"No gait report step detection in {fname}!")

            for step in meas.steps:
                h: int = step.heel_strike_idx - 5
                t: int = step.toe_off_idx + 5
                time: np.ndarray = meas.time[h:t] - meas.time[h]
                total_force: np.ndarray = meas.side_data[step.side].total_force[h:t]
                plt.plot(time, total_force, label=meas.tag, color=colors[i])

        plt.title("Total force curve of all detected steps")
        plt.xlabel("Time (s)")
        plt.ylabel("Total force (N)")
        plt.grid(True)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())  # Avoid legend entry per curve
        outfile = os.path.join(self.testdir, "gait_analysis.png")
        golden_result = os.path.join(self.testdir, "golden_gait_analysis.png")
        plt.savefig(outfile)
        print(outfile)
        print(golden_result)

        self.assertTrue(os.path.exists(outfile))
        self.assertTrue(os.path.exists(golden_result))

        # filecmp did not work in github actions
        # self.assertTrue(filecmp.cmp(outfile, golden_result))


if __name__ == "__main__":
    unittest.main()
