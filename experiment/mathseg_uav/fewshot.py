"""Fine-tune and test a MathSeg checkpoint using the fixed SUIM protocol."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiment.fewShot_SUIM.run import main

if __name__ == "__main__":
    main(default_model="mathseg")
