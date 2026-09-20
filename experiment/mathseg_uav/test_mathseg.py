"""CPU regression tests using a tiny real SegFormer, without pretrained downloads."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from transformers import SegformerConfig, SegformerModel

from experiment.mathseg_uav.model import MathSegUAV, SpatialPrior, VARIANTS
from experiment.mathseg_uav.protocol import (
    boundary_map, class_weights, confusion_metrics, read_mask, read_splits, segmentation_loss,
    UpdateBatchSampler,
)


def tiny_config():
    return SegformerConfig(hidden_sizes=[8, 16, 24, 32], depths=[1, 1, 1, 1],
                           num_attention_heads=[1, 2, 3, 4], sr_ratios=[4, 2, 1, 1],
                           mlp_ratios=[2, 2, 2, 2], drop_path_rate=0.1)


class MathSegTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self, variant):
        return MathSegUAV(variant=variant, width=8, pretrained=False, encoder_config=tiny_config().to_dict())

    def test_prior_constant_and_alignment(self):
        prior = SpatialPrior()
        for value in (0.0, 0.1, 0.2, 0.5, 0.9, 1.0):
            result = prior(torch.full((1, 3, 32, 48), value), (8, 12))
            self.assertTrue(torch.equal(result, torch.zeros_like(result)), f"nonzero prior for constant {value}")
        image = torch.zeros(1, 3, 32, 48)
        image[:, :, :, 24:] = 1
        result = prior(image, (32, 48))
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue(((result >= 0) & (result <= 1)).all())
        self.assertGreater(float(result[0, 0, :, 23:25].mean()), 0.5)
        torch.testing.assert_close(prior(image.flip(-1), (32, 48)), result.flip(-1), atol=1e-6, rtol=1e-5)

    def test_all_variants_forward_backward(self):
        image = torch.rand(2, 3, 32, 48)
        target = torch.randint(0, 9, (2, 32, 48))
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                model = self.model(variant)
                output = model(image)
                self.assertEqual(output["logits"].shape, (2, 9, 32, 48))
                torch.testing.assert_close(output["scale_weights"].sum(1), torch.ones(2, 8, 12))
                weights = class_weights([0, 50, 4, 4, 4, 1, 4, 4, 1], 0.5) if variant == "M4" else None
                loss = segmentation_loss(output, target, weights)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                # Gate last layer starts at zero: after its first update the prior
                # connections should receive a gradient only in the prior variants.
                if variant in ("M2", "M3", "M4"):
                    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
                    optimizer.step()
                    optimizer.zero_grad()
                    segmentation_loss(model(image), target, weights).backward()
                    prior_gradient = model.head.gate[0].weight.grad[:, -4:]
                    self.assertEqual(bool(prior_gradient.abs().sum() > 0), variant != "M2")

    def test_paired_initialization_and_parameter_count(self):
        torch.manual_seed(42)
        m2 = self.model("M2")
        torch.manual_seed(42)
        m3 = self.model("M3")
        self.assertEqual(sum(p.numel() for p in m2.parameters()), sum(p.numel() for p in m3.parameters()))
        for name, parameter in m2.state_dict().items():
            torch.testing.assert_close(parameter, m3.state_dict()[name], rtol=0, atol=0)
        restored = MathSegUAV(**m3.model_config)
        restored.load_state_dict(m3.state_dict(), strict=True)

    def test_ignore_loss_and_metrics(self):
        logits = torch.randn(1, 9, 8, 8, requires_grad=True)
        boundary = torch.randn(1, 1, 8, 8, requires_grad=True)
        target = torch.zeros(1, 8, 8, dtype=torch.long)
        loss = segmentation_loss({"logits": logits, "boundary": boundary}, target)
        self.assertEqual(float(loss), 0)
        loss.backward()
        self.assertEqual(float(logits.grad.abs().sum()), 0)
        target[:, :, 4:] = 2
        edge, safe = boundary_map(target, target != 0)
        self.assertFalse(edge.any())  # Ignore/foreground is not a semantic edge.
        self.assertFalse(safe[:, :, 4].any())
        logits = torch.randn(1, 9, 8, 8, requires_grad=True)
        boundary = torch.randn(1, 1, 8, 8, requires_grad=True)
        segmentation_loss({"logits": logits, "boundary": boundary}, target).backward()
        self.assertEqual(float(logits.grad[:, :, :, :4].abs().sum()), 0)
        self.assertEqual(float(boundary.grad[:, :, :, :5].abs().sum()), 0)
        confusion = torch.zeros(9, 9, dtype=torch.long)
        confusion[1, 1], confusion[1, 0], confusion[2, 2] = 2, 2, 4
        self.assertAlmostEqual(confusion_metrics(confusion)["mIoU"], 0.75)
        target.zero_()
        target[:, 4, 4] = 1
        output = {"logits": logits.detach(), "boundary": boundary.detach()}
        # Weighted mean CE must not shrink if only one low-weight pixel is valid.
        torch.testing.assert_close(segmentation_loss(output, target),
                                   segmentation_loss(output, target, torch.full((9,), 0.25)))

    def test_class_weights_and_sampler_resume(self):
        weights = class_weights([100, 1000, 1, 0, 0, 0, 0, 0, 0], 0)
        torch.testing.assert_close(weights[1:], torch.ones(8))
        weights = class_weights([100, 1000, 1, 0, 0, 0, 0, 0, 0], 1)
        self.assertGreater(float(weights[2]), float(weights[1]))
        self.assertTrue(torch.isfinite(weights).all())
        full = list(UpdateBatchSampler(7, 2, 42, 0, 8))
        resumed = list(UpdateBatchSampler(7, 2, 42, 3, 8))
        self.assertEqual(full[3:], resumed)

    def test_palette_labels_cannot_silently_change(self):
        with tempfile.TemporaryDirectory(prefix="mathseg-mask-") as directory:
            path = Path(directory) / "mask.png"
            image = Image.fromarray(np.ones((8, 8), dtype=np.uint8), mode="P")
            palette = [0] * 768
            palette[3:6] = [255, 255, 255]
            image.putpalette(palette)
            image.save(path)
            with self.assertRaisesRegex(ValueError, "single-channel L"):
                read_mask(path)

    def test_cli_train_resume_eval_export(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(prefix="mathseg-test-") as directory:
            directory = Path(directory)
            images, masks = directory / "data/train/images", directory / "data/train/masks"
            images.mkdir(parents=True)
            masks.mkdir(parents=True)
            splits = directory / "splits"
            splits.mkdir()
            rng = np.random.default_rng(42)
            for index in range(4):
                Image.fromarray(rng.integers(0, 256, (32, 48, 3), dtype=np.uint8)).save(images / f"{index}.png")
                Image.fromarray(rng.integers(0, 9, (32, 48), dtype=np.uint8)).save(masks / f"{index}.png")
            for name, ids in (("train", "0\n1\n"), ("val", "2\n"), ("test", "3\n")):
                (splits / f"{name}.txt").write_text(ids)
            # Canonicalizing extensions must catch otherwise hidden overlap.
            (splits / "test.txt").write_text("0.png\n")
            with self.assertRaises(ValueError):
                read_splits(splits, images, masks)
            (splits / "test.txt").write_text("3\n")
            backbone = directory / "backbone"
            SegformerModel(tiny_config()).save_pretrained(backbone)
            common = ["--data-root", str(directory / "data"), "--split-dir", str(splits),
                      "--model-name", str(backbone), "--device", "cpu", "--num-workers", "0",
                      "--width", "8", "--size", "32", "--batch-size", "1", "--accumulation", "2",
                      "--variant", "M3", "--max-updates", "4", "--val-interval", "2", "--save-interval", "2"]
            environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", HF_HUB_OFFLINE="1")

            def run(script, arguments):
                result = subprocess.run([sys.executable, str(root / "experiment/mathseg_uav" / script), *arguments],
                                        env=environment, cwd=root, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return result

            full, resumed = directory / "full", directory / "resumed"
            run("train.py", [*common, "--work-dir", str(full)])
            run("train.py", [*common, "--work-dir", str(resumed), "--stop-after", "2"])
            run("train.py", [*common, "--work-dir", str(resumed), "--resume", str(resumed / "last.pt")])
            a = torch.load(full / "last.pt", weights_only=False)
            b = torch.load(resumed / "last.pt", weights_only=False)
            for name in a["model"]:
                torch.testing.assert_close(a["model"][name], b["model"][name], rtol=0, atol=0)
            self.assertEqual(a["update"], b["update"])
            run("train.py", [*common, "--eval-checkpoint", str(resumed / "best.pt")])
            metric = json.loads((resumed / "best_test_metrics.json").read_text())
            self.assertEqual(metric["samples"], 1)
            predictions = directory / "predictions"
            run("inference.py", ["export", "--checkpoint", str(resumed / "best.pt"), "--device", "cpu",
                                 "--input", str(images), "--output", str(predictions), "--allow-any-size"])
            self.assertEqual(sorted(p.name for p in predictions.iterdir()), [f"{i}.png" for i in range(4)])
            for path in predictions.iterdir():
                with Image.open(path) as image:
                    self.assertEqual(image.mode, "L")
                    self.assertEqual(image.size, (48, 32))
                    self.assertLessEqual(np.asarray(image).max(), 8)
            run("inference.py", ["benchmark", "--checkpoint", str(resumed / "best.pt"), "--device", "cpu",
                                 "--size", "32", "--warmup", "1", "--repeats", "2", "--profile-flops"])
            # M4 must also scan train-only counts and complete a real update.
            m4_arguments = common.copy()
            m4_arguments[m4_arguments.index("M3")] = "M4"
            run("train.py", [*m4_arguments, "--work-dir", str(directory / "m4"), "--stop-after", "1"])
            queue_arguments = common.copy()
            variant_index = queue_arguments.index("--variant")
            del queue_arguments[variant_index:variant_index + 2]
            queue_root = directory / "queue"
            queue = ["run", "--variants", "M0", "M1", "--output-root", str(queue_root),
                     "--", *queue_arguments, "--max-updates", "1"]
            run("run.py", queue)
            self.assertTrue((queue_root / "summary.csv").is_file())
            self.assertEqual(len(json.loads((queue_root / "aggregate.json").read_text())), 2)
            before = (queue_root / "M0_seed3407/train.log").stat().st_size
            queue.insert(queue.index("--"), "--resume")
            run("run.py", queue)
            self.assertEqual(before, (queue_root / "M0_seed3407/train.log").stat().st_size)


if __name__ == "__main__":
    unittest.main()
