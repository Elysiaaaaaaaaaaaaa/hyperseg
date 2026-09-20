"""Run: python -m unittest experiment.loveda_fewshot.test_manual_protocol"""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from experiment.loveda_fewshot.manual_protocol import SHOTS, load_protocol, resolve_samples


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.keys = [f"urban/{i}.png" for i in range(15)]
        self.evaluation = ["rural/eval.png"]
        self.manifests = {}
        for k in SHOTS:
            per_class = {"1": self.keys[:10][:k], "2": self.keys[5:15][:k]}
            self.manifests[k] = dict(split="Val", shots_per_class=k, target_classes=[1, 2],
                                     per_class=per_class,
                                     samples=list(dict.fromkeys(per_class["1"] + per_class["2"])),
                                     evaluation_samples=self.evaluation, reserved_support_samples=self.keys)

    def write(self, manifests=None, evaluation=None):
        for k, payload in (manifests or self.manifests).items():
            (self.root / f"{k}shot.json").write_text(json.dumps(payload))
        (self.root / "evaluation.json").write_text(json.dumps({"samples": evaluation or self.evaluation}))

    def test_zero_nested_and_shared_images(self):
        self.write()
        manifests, evaluation, fingerprint = load_protocol(self.root)
        self.assertEqual(manifests[0]["samples"], [])
        self.assertEqual(len(manifests[10]["samples"]), 15)
        self.assertEqual(evaluation, self.evaluation)
        self.assertEqual(len(fingerprint), 64)

    def test_reject_protocol_corruption(self):
        for mutation in ("leak", "changed_eval", "reordered", "wrong_union", "duplicate", "unsafe_path", "wrong_k"):
            with self.subTest(mutation=mutation):
                manifests = copy.deepcopy(self.manifests)
                evaluation = self.evaluation
                if mutation == "leak":
                    evaluation = [self.keys[0]]
                elif mutation == "changed_eval":
                    manifests[0]["evaluation_samples"] = ["rural/other.png"]
                elif mutation == "reordered":
                    manifests[2]["per_class"]["1"].reverse()
                elif mutation == "wrong_union":
                    manifests[1]["samples"] = self.keys[:1]
                elif mutation == "duplicate":
                    manifests[10]["samples"].append(self.keys[0])
                elif mutation == "unsafe_path":
                    manifests[0]["samples"] = ["../bad.png"]
                elif mutation == "wrong_k":
                    manifests[2]["shots_per_class"] = 5
                self.write(manifests, evaluation)
                with self.assertRaises(ValueError):
                    load_protocol(self.root)

    def test_case_insensitive_paths_and_missing_mask(self):
        for domain in ("Urban", "Rural"):
            for name in ("images_png", "masks_png"):
                folder = self.root / "vAL" / domain / name
                folder.mkdir(parents=True)
                (folder / "same.png").touch()
        paths = resolve_samples(self.root, ["urban/same.png", "rural/same.png"])
        self.assertNotEqual(paths["urban/same.png"], paths["rural/same.png"])
        paths["urban/same.png"][1].unlink()
        with self.assertRaises(FileNotFoundError):
            resolve_samples(self.root, ["urban/same.png"])


if __name__ == "__main__":
    unittest.main()
