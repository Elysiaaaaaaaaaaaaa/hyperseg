"""Lightweight tests for the fixed four-setting experiment matrix."""
import argparse
import unittest
from pathlib import Path
from types import SimpleNamespace

from experiment.fewShot_SUIM.run_grid import SETTINGS, commands
from experiment.fewShot_SUIM.run import configure_model, legacy_encoder_key


class GridTests(unittest.TestCase):
    def test_four_update_budget_combinations(self):
        args = argparse.Namespace(
            data_root=Path("SUIM"), manifest_dir=Path("protocol"), init_checkpoint=Path("v2.pt"),
            backbone_path=None, output_root=Path("runs/suim"), settings=list(SETTINGS), shots=[1, 2],
            seeds=[3407], head_init="random", crop_size=512, batch_size=2, num_workers=4,
            lr=1e-4, amp=True, device="cuda", skip_completed=False,
        )
        jobs = list(commands(args))
        self.assertEqual(len(jobs), 8)
        self.assertEqual(
            {(setting, mode, steps) for setting, mode, steps, *_ in jobs},
            {
                ("adapter_200", "adapter", 200), ("adapter_2000", "adapter", 2000),
                ("semantic_head_200", "semantic-head", 200),
                ("semantic_head_2000", "semantic-head", 2000),
            },
        )
        for *_, command in jobs:
            self.assertIn("--amp", command)
            self.assertEqual(command[command.index("--head-init") + 1], "random")

    def test_semantic_head_is_the_only_trainable_module(self):
        head = [SimpleNamespace(requires_grad=False) for _ in range(2)]
        frozen = [SimpleNamespace(requires_grad=True) for _ in range(4)]
        model = SimpleNamespace(
            parameters=lambda: iter(head + frozen),
            head=SimpleNamespace(parameters=lambda: iter(head)),
        )
        configure_model(model, "semantic-head")
        self.assertTrue(all(parameter.requires_grad for parameter in head))
        self.assertTrue(all(not parameter.requires_grad for parameter in frozen))

    def test_transformers_five_encoder_key_translation(self):
        self.assertEqual(
            legacy_encoder_key("encoder.backbone.stages.1.blocks.2.attention.q_proj.weight"),
            "encoder.backbone.encoder.block.1.2.attention.self.query.weight",
        )


if __name__ == "__main__":
    unittest.main()
