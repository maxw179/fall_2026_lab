"""Compatibility entry point; use systematic_testing.rms_snr_sweep.analysis.analyze_pd_sweep."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from systematic_testing.rms_snr_sweep.analysis.analyze_pd_sweep import *
if __name__ == "__main__":
    main()
