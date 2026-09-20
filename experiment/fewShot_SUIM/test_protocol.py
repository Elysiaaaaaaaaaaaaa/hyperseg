"""Lightweight protocol tests; no dataset, torch, or GPU required."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from experiment.fewShot_SUIM.protocol import SHOTS, load_protocol, resolve_protocol_paths


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.pool = [f"train_val/image_{i}.jpg" for i in range(20)]
        self.evaluation = [f"TEST/test_{i}.jpg" for i in range(110)]
        self.manifests = {}
        for k in SHOTS:
            per_class = {str(c): self.pool[c:c + 10][:k] for c in range(8)}
            samples = list(dict.fromkeys(key for values in per_class.values() for key in values))
            self.manifests[k] = {
                "split": "train_val", "shots_per_class": k, "target_classes": list(range(8)),
                "ignore_index": None, "per_class": per_class, "samples": samples,
                "evaluation_samples": self.evaluation,
                "reserved_support_samples": list(dict.fromkeys(
                    key for c in range(8) for key in self.pool[c:c + 10]
                )),
            }

    def write(self, manifests=None, evaluation=None):
        for k, payload in (manifests or self.manifests).items():
            (self.root / f"{k}shot.json").write_text(json.dumps(payload), encoding="utf-8")
        (self.root / "evaluation.json").write_text(
            json.dumps({"split": "TEST", "samples": evaluation or self.evaluation}), encoding="utf-8"
        )

    def test_nested_protocol(self):
        self.write()
        manifests, evaluation, fingerprint = load_protocol(self.root)
        self.assertEqual(manifests[1]["per_class"]["4"], self.manifests[10]["per_class"]["4"][:1])
        self.assertEqual(evaluation, self.evaluation)
        self.assertEqual(len(fingerprint), 64)

    def test_rejects_corruption(self):
        for mutation in ("wrong_zero", "reordered", "wrong_union", "wrong_split", "changed_test"):
            with self.subTest(mutation=mutation):
                manifests = copy.deepcopy(self.manifests)
                evaluation = self.evaluation
                if mutation == "wrong_zero":
                    manifests[1]["ignore_index"] = 0
                elif mutation == "reordered":
                    manifests[2]["per_class"]["3"].reverse()
                elif mutation == "wrong_union":
                    manifests[1]["samples"] = []
                elif mutation == "wrong_split":
                    manifests[1]["per_class"]["0"] = ["TEST/test_0.jpg"]
                elif mutation == "changed_test":
                    evaluation = ["TEST/other.jpg"]
                self.write(manifests, evaluation)
                with self.assertRaises(ValueError):
                    load_protocol(self.root)

    def test_resolves_jpg_to_top_level_bmp(self):
        for split in ("train_val", "TEST"):
            (self.root / split / "images").mkdir(parents=True)
            (self.root / split / "masks").mkdir(parents=True)
        (self.root / "train_val/images/example.jpg").touch()
        (self.root / "train_val/masks/example.bmp").touch()
        paths = resolve_protocol_paths(self.root, ["train_val/example.jpg"])
        self.assertEqual(paths["train_val/example.jpg"][1].name, "example.bmp")
        (self.root / "train_val/masks/example.bmp").unlink()
        with self.assertRaises(FileNotFoundError):
            resolve_protocol_paths(self.root, ["train_val/example.jpg"])


if __name__ == "__main__":
    unittest.main()
