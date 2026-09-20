"""CPU checks for selection, label handling, overlays and nested manifests."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from select_loveda_gui import (
    COLORS, SHOTS, export_manifests, overlay, ratio, read_mask, scan_dataset,
    validate_selection,
)


class SelectorChecks(unittest.TestCase):
    def test_overlay_and_ignore_denominators(self):
        mask = np.array([[0, 1], [2, 2]], dtype=np.uint8)
        rgb = Image.new("RGB", (2, 2), (100, 100, 100))
        result = np.array(overlay(rgb, mask, 2, 1, "仅目标叠加"))
        np.testing.assert_array_equal(result[0], np.full((2, 3), 100))
        np.testing.assert_array_equal(result[1, 0], COLORS[2])
        full = np.array(overlay(rgb, mask, 2, 1, "全部类别叠加"))
        np.testing.assert_array_equal(full[0, 0], [100, 100, 100])
        np.testing.assert_array_equal(full[0, 1], COLORS[1])
        np.testing.assert_array_equal(np.array(overlay(rgb, mask, 2, 0, "仅目标叠加")), np.array(rgb))
        record = dict(counts=[1, 1, 2, 0, 0, 0, 0, 0], total=4, valid=3)
        self.assertEqual(ratio(record, 2), 50)
        self.assertAlmostEqual(ratio(record, 2, True), 200 / 3)

    def test_scan_pairing_and_label_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for domain in ("urban", "rural"):
                images = root / "val" / domain / "images_png"
                masks = images.parent / "masks_png"
                images.mkdir(parents=True)
                masks.mkdir()
                Image.new("RGB", (2, 2)).save(images / "same.png")
                Image.fromarray(np.array([[255, 1], [2, 2]], dtype=np.uint8)).save(masks / "same.png")
            val, records = scan_dataset(root)
            self.assertEqual(val, root / "val")
            self.assertEqual(len(records), 2)
            self.assertNotEqual(records[0]["key"], records[1]["key"])
            self.assertEqual(records[0]["counts"], [1, 1, 2, 0, 0, 0, 0, 0])
            self.assertEqual(scan_dataset(val)[1][0]["total"], 4)
            mask_path = records[0]["mask"]
            Image.fromarray(np.array([[8]], dtype=np.uint8)).save(mask_path)
            with self.assertRaises(ValueError):
                read_mask(mask_path)
            Image.new("L", (1, 1), 2).save(mask_path)
            with self.assertRaisesRegex(ValueError, "尺寸不一致"):
                scan_dataset(root)

    def test_nested_export_shared_images_and_fixed_evaluation(self):
        records = [dict(key=f"urban/{i}.png", counts=[0, 2, 2, 0, 0, 0, 0, 0], total=4, valid=4)
                   for i in range(16)]
        chosen = {"1": [r["key"] for r in records[:10]],
                  "2": [r["key"] for r in records[5:15]]}
        with tempfile.TemporaryDirectory() as tmp:
            dest = export_manifests(Path(tmp), chosen, [1, 2], records)
            previous = set()
            for k in SHOTS:
                payload = json.loads((dest / f"{k}shot.json").read_text())
                samples = payload["samples"]
                self.assertEqual(len(samples), len(set(samples)))
                self.assertTrue(previous <= set(samples))
                previous = set(samples)
                self.assertEqual(payload["per_class"]["1"], chosen["1"][:k])
                self.assertEqual(payload["per_class"]["2"], chosen["2"][:k])
                self.assertEqual(payload["evaluation_samples"], ["urban/15.png"])
                self.assertFalse(set(samples) & set(payload["evaluation_samples"]))
                if k == 0:
                    self.assertEqual(samples, [])
            self.assertEqual(len(previous), 15)
            with self.assertRaises(ValueError):
                export_manifests(Path(tmp), {"1": chosen["1"][:9], "2": chosen["2"]}, [1, 2], records)
            with self.assertRaises(ValueError):
                validate_selection({"1": ["missing.png"]}, [1], {r["key"]: r for r in records})
            with self.assertRaises(ValueError):
                validate_selection({"1": ["urban/0.png"] * 2}, [1], {r["key"]: r for r in records})


if __name__ == "__main__":
    unittest.main()
