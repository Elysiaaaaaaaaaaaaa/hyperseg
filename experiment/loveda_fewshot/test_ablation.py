"""Lightweight tests; no torch, GPU or model cache required."""
import argparse
import unittest
from pathlib import Path
from types import SimpleNamespace

from experiment.loveda_fewshot.run_ablation import commands
from experiment.loveda_fewshot.run_manual import configure_model


class AblationTests(unittest.TestCase):
    def test_exact_budget_matrix(self):
        args = argparse.Namespace(data_root=Path("LoveDA"), manifest_dir=Path("export"),
                                  init_checkpoint=Path("v2.pt"), output_root=Path("runs/new"),
                                  seed=3407, num_workers=4, backbone_path=None, skip_completed=False)
        jobs = list(commands(args))
        self.assertEqual([(mode, steps) for _, mode, steps, _ in jobs],
                         [("adapter", 200), ("semantic-head", 2000), ("semantic-head", 200)])
        self.assertEqual(len({command[command.index("--output-root") + 1] for *_, command in jobs}), 3)
        for *_, command in jobs:
            self.assertEqual(command[command.index("--shots") + 1:command.index("--seeds")], ["1", "2"])
            self.assertEqual(command[command.index("--seeds") + 1:command.index("--modes")], ["3407"])

    def test_semantic_head_excludes_boundary_and_backbone(self):
        head = [SimpleNamespace(requires_grad=False) for _ in range(2)]
        frozen = [SimpleNamespace(requires_grad=True) for _ in range(4)]
        model = SimpleNamespace(parameters=lambda: iter(head + frozen),
                                head=SimpleNamespace(parameters=lambda: iter(head)))
        configure_model(model, "semantic-head")
        self.assertTrue(all(p.requires_grad for p in head))
        self.assertTrue(all(not p.requires_grad for p in frozen))


if __name__ == "__main__":
    unittest.main()
