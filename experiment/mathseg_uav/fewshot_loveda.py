"""MathSeg LoveDA fixed-holdout few-shot runs, including mapped 0-shot."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiment.loveda_fewshot.run_manual import main

if __name__ == "__main__":
    main(default_model="mathseg")
