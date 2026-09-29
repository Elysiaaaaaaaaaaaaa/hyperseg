"""Few-shot routing checks plus optional offline tiny-model transfer tests."""
import argparse
import importlib.util
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

from experiment.fewShot_SUIM.run_grid import SETTINGS, commands, summarize
from experiment.mathseg_uav.fewshot_backend import configure_mathseg, load_mathseg


class RoutingTests(unittest.TestCase):
    def test_mathseg_grid(self):
        args = argparse.Namespace(
            model="mathseg", data_root=Path("SUIM"), manifest_dir=Path("protocol"),
            init_checkpoint=Path("m3.pt"), backbone_path=None, output_root=Path("runs/mathseg"),
            settings=list(SETTINGS), shots=[1, 2, 5, 10], seeds=[3407, 3408, 3409],
            head_init="random", crop_size=512, batch_size=2, num_workers=0,
            lr=1e-4, amp=True, device="cuda", skip_completed=True,
        )
        jobs = list(commands(args))
        self.assertEqual(len(jobs), 48)
        self.assertEqual(len({job[-2] for job in jobs}), 48)
        for *_, command in jobs:
            self.assertEqual(command[command.index("--model") + 1], "mathseg")
            self.assertEqual(command[command.index("--init-checkpoint") + 1], "m3.pt")
            self.assertIn("--skip-completed", command)

    def test_loveda_grid_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(
                dataset="loveda", model="mathseg", data_root=Path("LoveDA"), manifest_dir=Path("protocol"),
                init_checkpoint=Path("m3.pt"), backbone_path=None, output_root=Path(directory),
                settings=list(SETTINGS), shots=[1, 2, 5, 10], seeds=[3407, 3408, 3409],
                head_init="semantic-map", crop_size=512, batch_size=2, num_workers=0,
                lr=1e-4, amp=True, device="cuda", skip_completed=True,
            )
            jobs = list(commands(args, "cached-hash"))
            self.assertEqual(len(jobs), 48)
            for _, mode, _, k, seed, output, command in jobs:
                self.assertTrue(command[2].endswith("fewshot_loveda.py"))
                self.assertNotIn("--checkpoint-sha256", command)
                self.assertEqual(command[command.index("--label-policy") + 1], "standard")
                parent = Path(command[command.index("--output-root") + 1])
                self.assertEqual(output, parent / f"{mode}_{k}shot_seed{seed}")
                output.mkdir(parents=True)
                (output / "summary.json").write_text(json.dumps({
                    "support_images": k, "metrics": {"mIoU": .5, "pixel_accuracy": .8,
                                                      "per_class_iou": {"forest": .4}},
                }))
            summarize(args)
            aggregates = json.loads((Path(directory) / "aggregate.json").read_text())["results"]
            self.assertEqual(len(aggregates), 16)
            self.assertTrue(all(row["runs"] == 3 and row["mean_mIoU"] == .5 for row in aggregates))

    def test_update_scopes(self):
        encoder, fusion, semantic = [SimpleNamespace(requires_grad=True) for _ in range(3)]
        model = SimpleNamespace(parameters=lambda: iter([encoder, fusion, semantic]))
        model.head = SimpleNamespace(
            parameters=lambda: iter([fusion, semantic]),
            segmentation=SimpleNamespace(parameters=lambda: iter([semantic])),
        )
        for mode, expected in (("semantic-head", [False, False, True]),
                               ("adapter", [False, True, True]), ("full", [True, True, True])):
            configure_mathseg(model, mode)
            self.assertEqual([p.requires_grad for p in model.parameters()], expected)


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"),
                     "PyTorch and Transformers are required for tiny-model checks")
class TransferTests(unittest.TestCase):
    def test_loveda_standard_labels_and_loss(self):
        import numpy as np
        from PIL import Image
        import torch
        from experiment.loveda_fewshot.train import read_mask, compute_class_weights, loveda_loss, evaluate

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.png"
            Image.fromarray(np.array([[0, 255], [1, 7]], dtype=np.uint8)).save(path)
            mask = read_mask(path)
            self.assertEqual(mask.tolist(), [[255, 255], [0, 6]])
            weights, histogram = compute_class_weights([SimpleNamespace(mask=path)])
            self.assertEqual(histogram, [1, 0, 0, 0, 0, 0, 1])
            self.assertGreater(weights[0].item(), 0)
            target = torch.tensor(mask.astype(np.int64)).unsqueeze(0)
            logits = torch.zeros(1, 7, 2, 2, requires_grad=True)
            boundary = torch.zeros(1, 1, 2, 2, requires_grad=True)
            loveda_loss({"logits": logits, "boundary": boundary}, target, weights).backward()
            self.assertEqual(logits.grad[:, :, 0, :].abs().sum().item(), 0)
            self.assertGreater(logits.grad[:, :, 1, 0].abs().sum().item(), 0)
            ignored = torch.full_like(target, 255)
            self.assertEqual(loveda_loss({"logits": logits, "boundary": boundary}, ignored, weights).item(), 0)
            predicted = torch.tensor([[[5, 4], [0, 6]]])
            values = torch.nn.functional.one_hot(predicted, 7).permute(0, 3, 1, 2).float()
            class FakeModel:
                def eval(self): pass
                def __call__(self, image): return {"logits": values}
            metrics = evaluate(FakeModel(), [{"image": torch.zeros(1, 3, 2, 2), "mask": target}], torch.device("cpu"))
            self.assertEqual(metrics["mIoU"], 1)
            self.assertEqual(sum(map(sum, metrics["confusion"])), 2)
            self.assertEqual(metrics["per_class_iou"]["background"], 1)
            Image.fromarray(np.array([[8]], dtype=np.uint8)).save(path)
            with self.assertRaises(ValueError): read_mask(path)

    def test_transfer_reload_and_backward(self):
        import torch
        from transformers import SegformerConfig
        from experiment.mathseg_uav.model import MathSegUAV, VARIANTS
        from experiment.fewShot_SUIM.run import suim_loss, training_mode, SEMANTIC_MAPPING

        torch.set_num_threads(1)
        config = SegformerConfig(hidden_sizes=[8, 16, 24, 32], depths=[1]*4,
                                 num_attention_heads=[1, 2, 3, 4], sr_ratios=[4, 2, 1, 1])
        for variant in VARIANTS:
            source = MathSegUAV(variant=variant, width=8, pretrained=False,
                                encoder_config=config.to_dict())
            old_config = dict(source.model_config)
            old_config.pop("classes")  # Historical UAV checkpoints omit classes.
            checkpoint = {"model": source.state_dict(), "model_config": old_config}
            model, _ = load_mathseg(checkpoint)
            for name, tensor in model.state_dict().items():
                if not name.startswith("head.segmentation."):
                    torch.testing.assert_close(tensor, checkpoint["model"][name], rtol=0, atol=0)
            restored = MathSegUAV(**model.model_config)
            restored.load_state_dict(model.state_dict(), strict=True)
            configure_mathseg(model, "semantic-head")
            training_mode(model)
            image = torch.rand(2, 3, 32, 48)
            output = model(image)
            self.assertEqual(output["logits"].shape, (2, 8, 32, 48))
            target = torch.zeros(2, 32, 48, dtype=torch.long)  # SUIM class 0 is valid.
            loss = suim_loss(output, target, torch.ones(8))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(model.head.segmentation.weight.grad.abs().sum().item(), 0)
            self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))
            mapped, _ = load_mathseg(checkpoint, head_init="semantic-map")
            for target_id, source_id in SEMANTIC_MAPPING.items():
                torch.testing.assert_close(mapped.head.segmentation.weight[target_id],
                                           source.head.segmentation.weight[source_id])
            loveda, _ = load_mathseg(checkpoint, head_init="semantic-map", dataset="loveda")
            torch.testing.assert_close(loveda.head.segmentation.weight, source.head.segmentation.weight[1:8])
            torch.testing.assert_close(loveda.head.segmentation.bias, source.head.segmentation.bias[1:8])
            self.assertEqual(loveda.model_config["classes"], 7)
            restored_loveda = MathSegUAV(**loveda.model_config)
            restored_loveda.load_state_dict(loveda.state_dict(), strict=True)
            self.assertEqual(restored_loveda(image)["logits"].shape[1], 7)
            configure_mathseg(loveda, "head")
            self.assertTrue(all(p.requires_grad for p in loveda.head.boundary.parameters()))
            self.assertTrue(all(not p.requires_grad for p in loveda.head.decoder.parameters()))
            broken = {**checkpoint, "model": dict(checkpoint["model"])}
            broken["model"].pop("head.boundary.weight")
            with self.assertRaises(RuntimeError):
                load_mathseg(broken)


if __name__ == "__main__":
    unittest.main()
