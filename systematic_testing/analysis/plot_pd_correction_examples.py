"""Compatibility entry point; use systematic_testing.correction_examples.plot_pd_correction_examples."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from systematic_testing.correction_examples.plot_pd_correction_examples import *
if __name__ == "__main__":
    main()
