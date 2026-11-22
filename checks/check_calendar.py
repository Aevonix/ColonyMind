"""Check "calendar": a source with no connector. The model looks with its own tools on every run.

The script wakes the model, names the file that holds the notes of the last state, and records the run.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_base  # noqa: E402

if __name__ == "__main__":
    sys.exit(check_base.main("calendar"))
