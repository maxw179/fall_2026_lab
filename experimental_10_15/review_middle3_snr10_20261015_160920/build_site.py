"""Refresh this batch using the shared experimental results page."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from experimental_10_15.results_site import build_site
if __name__=='__main__':build_site(HERE)
