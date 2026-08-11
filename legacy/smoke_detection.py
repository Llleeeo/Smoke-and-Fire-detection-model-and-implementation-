"""Legacy Roboflow dataset download helper.

The current, reproducible workflow is in training/leakage_safe_yolov5. This
helper is retained only for documenting the original Roboflow source and never
stores an API key in the repository.
"""

import os

from roboflow import Roboflow


def main() -> None:
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("Set ROBOFLOW_API_KEY before running this helper.")

    roboflow = Roboflow(api_key=api_key)
    project = roboflow.workspace("leo-there").project("smoke-and-cigarette")
    project.version(1).download("yolov5")


if __name__ == "__main__":
    main()
