"""MathSeg LoveDA 1/2/5/10-shot, 200/2000-step experiment matrix."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiment.fewShot_SUIM.run_grid import main

if __name__ == "__main__":
    main(default_model="mathseg", default_dataset="loveda")
