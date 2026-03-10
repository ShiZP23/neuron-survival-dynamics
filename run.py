import os
import sys

SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from neuron_survival_dynamics.cli import main  # noqa: E402


if __name__ == "__main__":
    main()
