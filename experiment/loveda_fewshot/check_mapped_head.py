"""CPU check of mapped head initialization using the original UAV checkpoint."""
from pathlib import Path
import torch
from torch import nn
from train import load_transfer_checkpoint


def main():
    root = Path(__file__).resolve().parents[2]
    source = torch.load(root / 'models/hyperseg_b3_best.pt', map_location='cpu', weights_only=False)
    state = source.get('model', source)
    channels = state['head.weight'].shape[1]
    model = nn.Module()
    model.head = nn.Conv2d(channels, 8, 1)
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    report = load_transfer_checkpoint(model, source, 'mapped')
    for i in (1, 2, 3, 4, 5, 7):
        assert torch.equal(model.head.weight[i], state['head.weight'][i])
        assert torch.equal(model.head.bias[i], state['head.bias'][i])
    for i in (0, 6):
        assert torch.equal(model.head.weight[i], initial['head.weight'][i])
        assert torch.equal(model.head.bias[i], initial['head.bias'][i])
    assert report['head_random_channels'] == [0, 6]
    model.zero_grad()
    model.head(torch.randn(1, channels, 2, 2)).sum().backward()
    assert model.head.weight.grad is not None and model.head.bias.grad is not None
    model.load_state_dict(initial)
    load_transfer_checkpoint(model, source, 'random')
    for k, v in model.state_dict().items():
        assert torch.equal(v, initial[k]), k
    bad = {'model': {'head.weight': state['head.weight'][:8], 'head.bias': state['head.bias'][:8]}}
    try:
        load_transfer_checkpoint(model, bad, 'mapped')
    except ValueError:
        pass
    else:
        raise AssertionError('8-class source incorrectly accepted as UAV source')
    print('MAPPED_HEAD_OK: copied 1..5,7; unchanged 0,6; gradients enabled; random baseline unchanged; wrong source rejected')


if __name__ == '__main__':
    main()
