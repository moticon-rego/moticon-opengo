import os

from moticon_opengo.gait_report import GaitReport, GaitReportFileIterator


def main():
    testdata_dir: str = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "tests", "testdata")
    )

    for fname in GaitReportFileIterator(testdata_dir):
        rep = GaitReport(fname)
        print(rep.cadence)

    for step in rep.steps:
        print(f"Step on {step.side} side, t={step.time_initial_contact}")


if __name__ == "__main__":
    main()
