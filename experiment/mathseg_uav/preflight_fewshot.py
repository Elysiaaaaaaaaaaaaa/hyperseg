"""Check real MathSeg source checkpoints, data protocols and GPU adaptation steps."""
import argparse
import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from torch.utils.data import DataLoader
    from experiment.mathseg_uav.fewshot_backend import (
        load_mathseg, configure_mathseg, checkpoint_payload, restore_compact,
    )
    from experiment.fewShot_SUIM import run as suim
    from experiment.fewShot_SUIM.data import SUIMDataset
    from experiment.loveda_fewshot import manual_protocol as loveda_protocol
    from experiment.loveda_fewshot.train import Sample, LoveDADataset, compute_class_weights, loveda_loss

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--loveda-root", type=Path, required=True)
    parser.add_argument("--suim-root", type=Path, required=True)
    parser.add_argument("--loveda-manifest", type=Path, required=True)
    parser.add_argument("--suim-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = {"protocols": {}, "checks": []}
    inputs = {}
    for dataset in ("loveda", "suim"):
        if dataset == "loveda":
            manifests, evaluation, digest = loveda_protocol.load_protocol(args.loveda_manifest)
            paths = loveda_protocol.resolve_samples(args.loveda_root, manifests[10]["samples"] + evaluation)
            def samples(keys):
                return [Sample(key.split('/')[0], key.split('/')[1], *paths[key]) for key in keys]
            support = samples(manifests[1]["samples"])
            train = LoveDADataset(support, 512, True)
            test = LoveDADataset(samples(evaluation[:1]), 1024, False)
            weights, histogram = compute_class_weights(support)
            loss_fn = loveda_loss
            # Check every K has supervision for all seven semantic classes.
            for k in (1, 2, 5, 10):
                _, counts = compute_class_weights(samples(manifests[k]["samples"]))
                assert all(counts), (dataset, k, counts)
        else:
            manifests, evaluation, digest = suim.load_protocol(args.suim_manifest)
            suim.resolve_protocol_paths(args.suim_root, manifests[10]["samples"] + evaluation)
            support, queries = suim.resolve_protocol_samples(args.suim_root, manifests[1], evaluation)
            train, test = SUIMDataset(support, 512, True), SUIMDataset(queries[:1], None, False)
            weights, histogram = suim.compute_class_weights(support)
            loss_fn = suim.suim_loss
            for k in (1, 2, 5, 10):
                supports, _ = suim.resolve_protocol_samples(args.suim_root, manifests[k], evaluation)
                suim.compute_class_weights(supports)
        inputs[dataset] = (next(iter(DataLoader(train, batch_size=2))), test[0], weights.cuda(), loss_fn)
        report["protocols"][dataset] = dict(sha256=digest, evaluation_images=len(evaluation),
            support_images={str(k): len(v["samples"]) for k, v in manifests.items()}, histogram_1shot=histogram)

    for variant in ("m0", "m1", "m2", "m3"):
        path = args.source_root / f"{variant}_seed3407" / "best.pt"
        source_hash = suim.file_sha256(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        for key in ("optimizer", "scheduler", "scaler"):
            checkpoint.pop(key, None)
        for dataset, (batch, query, weights, loss_fn) in inputs.items():
            model, transfer = load_mathseg(checkpoint, dataset=dataset,
                head_init="semantic-map" if dataset == "loveda" else "random")
            assert model.variant.lower() == variant
            configure_mathseg(model, "adapter")
            model.cuda()
            suim.training_mode(model)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
            torch.cuda.reset_peak_memory_stats()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(batch["image"].cuda())
                loss = loss_fn({k: output[k].float() for k in ("logits", "boundary")}, batch["mask"].cuda(), weights)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
            assert torch.isfinite(loss) and torch.isfinite(norm)
            optimizer.step()
            model.eval()
            with torch.inference_mode():
                image = query["image"].unsqueeze(0).cuda()
                logits = model(image)["logits"]
                assert torch.isfinite(logits).all()
                assert logits.shape[1] == (7 if dataset == "loveda" else 8)
                payload = checkpoint_payload(model, 1, {}, path, source_hash, "compact")
                restored = restore_compact(payload).cuda().eval()
                torch.testing.assert_close(restored(image)["logits"], logits)
            report["checks"].append(dict(variant=variant, dataset=dataset, loss=loss.item(),
                source_sha256=source_hash, peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3,
                compact_bytes=sum(t.numel()*t.element_size() for t in payload["model"].values()),
                transfer=transfer, compact_roundtrip="passed"))
            print(json.dumps(report["checks"][-1]), flush=True)
            del model, restored, optimizer, payload, output, logits, loss, image
            gc.collect()
            torch.cuda.empty_cache()
        del checkpoint
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("PREFLIGHT_OK", flush=True)


if __name__ == "__main__":
    main()
