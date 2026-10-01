#!/usr/bin/env python
"""Make a finished MMSeg checkpoint resumable under a *different* LR schedule.

Why this is needed
------------------
MMEngine's ``Runner.resume`` restores the param-scheduler state verbatim, and
``_ParamScheduler.state_dict`` dumps ``self.__dict__`` while
``load_state_dict`` does ``self.__dict__.update(...)``.  A checkpoint written by
a finished 20k run therefore carries ``end=20000`` / ``last_step=19999`` /
``_global_step=19999`` for its ``PolyLR``.  Restoring that state makes the guard
inside ``_ParamScheduler.step``::

    if self.begin <= self._global_step < self.end:
        self.last_step += 1
        ... update the LR ...

false forever, so **the LR is never written again** and a naive ``--resume``
would train the whole continuation at whatever ``param_groups['lr']`` the fresh
optimizer was built with.  Dropping ``param_schedulers`` from the checkpoint
lets the scheduler freshly built from the config ("begin=0, end=<new end>") take
effect instead.

The learning-rate anchor
------------------------
``PolyParamScheduler._get_value`` is multiplicative, so with ``eta_min=0`` it
telescopes into a closed form::

    lr(n) = lr(0) * ((T - n) / T) ** power,    T = end - begin - 1

where ``n = last_step``, advanced once per training iteration.  So a *fresh*
scheduler built with ``end = total_iters - resumed_iter`` and initial value::

    anchor = base_lr * ((total_iters - 1 - resumed_iter) / (total_iters - 1)) ** power

reproduces, for the continuation steps ``n = 0 .. total_iters - resumed_iter``,
*exactly* the LR a from-scratch ``total_iters`` run would have at iterations
``resumed_iter + n``::

    anchor * ((E - 1 - n) / (E - 1))**p  ==  base_lr * ((T60 - r - n) / T60)**p

with ``E = total_iters - resumed_iter`` and ``T60 = total_iters - 1``, because
``(E-1)/T60 * (E-1-n)/(E-1) = (E-1-n)/T60``.  Run ``--self-test`` to see the
equivalence checked numerically over the whole grid (stdlib only, no torch).

Usage on the server (torch comes from the mask2former venv)::

    .venv-mask2former/bin/python experiment/mask2former_uav/prepare_resume_checkpoint.py \
        --source runs/mask2former_uav/h2_swin_l_mask2former_20k_seed3407_fp32/iter_20000.pth \
        --output runs/mask2former_uav/_resume_prep/h2_iter20000_nosched.pth \
        --plan   runs/mask2former_uav/_resume_prep/h2_to_60k_plan.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Keys of `_ParamScheduler.state_dict()` worth echoing back to the operator: they
# are the evidence for why the state has to be dropped.
SCHEDULER_STATE_KEYS = ('begin', 'end', 'last_step', '_global_step', 'power',
                        'eta_min', 'total_iters', 'base_values')


def poly_lr(base_lr: float, step: int, end: int, power: float) -> float:
    """LR of MMSeg ``PolyLR`` (``eta_min=0``, ``begin=0``) at ``last_step=step``."""
    total = end - 1
    if step >= total:
        return 0.0
    return base_lr * ((total - step) / total) ** power


def resume_plan(base_lr: float, power: float, resumed_iter: int,
                total_iters: int) -> dict:
    """Return the scheduler end and initial LR that continue a 60k schedule."""
    if not 0 < resumed_iter < total_iters:
        raise ValueError(
            f'resumed_iter must be inside (0, {total_iters}), got {resumed_iter}')
    anchor = poly_lr(base_lr, resumed_iter, total_iters, power)
    scheduler_end = total_iters - resumed_iter
    return dict(
        base_lr=anchor,
        scheduler_end=scheduler_end,
        config_base_lr=base_lr,
        power=power,
        resumed_iter=resumed_iter,
        total_iters=total_iters,
        lr_at_resume=anchor,
        lr_at_resume_as_if_from_scratch=poly_lr(base_lr, resumed_iter,
                                               total_iters, power),
        lr_at_end=poly_lr(anchor, scheduler_end - 1, scheduler_end, power),
    )


def self_test() -> None:
    """Check the equivalence numerically, over the whole continuation grid."""
    base_lr, power, resumed, total = 1e-4, 0.9, 20000, 60000
    plan = resume_plan(base_lr, power, resumed, total)
    anchor, end = plan['base_lr'], plan['scheduler_end']
    worst = 0.0
    for n in range(0, end):
        continued = poly_lr(anchor, n, end, power)
        as_if = poly_lr(base_lr, resumed + n, total, power)
        scale = as_if if as_if > 0 else 1.0
        worst = max(worst, abs(continued - as_if) / scale)
        if n in (0, 1, resumed, end - 2, end - 1):
            print(f'  step {n:6d}  continued {continued:.10e}  '
                  f'as-if-60k {as_if:.10e}  rel.diff {abs(continued - as_if) / scale:.2e}')
    print(f'  base_lr      : {base_lr:g}')
    print(f'  anchor       : {anchor:.10e}   (= LR at iter {resumed} of a 60k run)')
    print(f'  scheduler end: {end}   (fresh PolyLR counter spans 0..{end - 1})')
    print(f'  LR at end    : {poly_lr(anchor, end - 1, end, power):.10e}')
    print(f'  worst relative difference: {worst:.3e}')
    if worst >= 1e-12:
        raise SystemExit('FAIL: the continuation LR curve is not equivalent')
    print('SELF-TEST OK')


def prepare(args) -> None:
    import torch  # imported here so --self-test runs without torch

    source = args.source.resolve()
    if not source.is_file():
        raise SystemExit(f'missing source checkpoint: {source}')
    checkpoint = torch.load(source, map_location='cpu')
    if not isinstance(checkpoint, dict):
        raise SystemExit(f'{source} is not a dict checkpoint')

    for key in ('state_dict', 'meta', 'message_hub'):
        if key not in checkpoint:
            raise SystemExit(f'{source} has no {key!r}; refusing to rewrite it')

    meta = checkpoint['meta']
    resumed_iter = meta.get('iter')
    print(f'source  : {source}')
    print(f'meta    : iter={resumed_iter} epoch={meta.get("epoch")} '
          f'seed={meta.get("seed")}')
    if args.expect_iter is not None and resumed_iter != args.expect_iter:
        raise SystemExit(
            f'expected meta["iter"] == {args.expect_iter}, got {resumed_iter!r}; '
            'the LR anchor would be wrong -- stop and investigate')

    sched = checkpoint.get('param_schedulers')
    if sched is not None:
        # `Runner.save_checkpoint` writes a list for a single optimizer and a
        # dict-of-lists for multiple optimizers.
        states = ([state for group in sched.values() for state in group]
                  if isinstance(sched, dict) else list(sched))
        for index, state in enumerate(states):
            kept = {k: v for k, v in state.items() if k in SCHEDULER_STATE_KEYS}
            print(f'dropped param_schedulers[{index}]: {kept}')
    else:
        print('note: checkpoint carries no param_schedulers; nothing to drop')
    for key in ('optimizer', 'param_schedulers'):
        if key in checkpoint:
            del checkpoint[key]
    if 'optimizer' not in checkpoint:
        print('note: no optimizer state in the checkpoint (saved with '
              '--no-save-optimizer), so Adam moments restart -- the LR curve is '
              'still exact, the adaptive state is re-estimated.')

    plan = resume_plan(args.base_lr_config, args.power, resumed_iter,
                       args.total_iters)
    plan.update(source=str(source), output=str(args.output.resolve()))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    reloaded = torch.load(args.output, map_location='cpu')
    print(f'wrote   : {args.output}  keys={sorted(reloaded.keys())}')
    for key in ('param_schedulers', 'optimizer'):
        if key in reloaded:
            raise SystemExit(f'{key} survived the rewrite; aborting')
    if reloaded['meta'].get('iter') != resumed_iter:
        raise SystemExit('iteration counter changed during the rewrite; aborting')

    args.plan.parent.mkdir(parents=True, exist_ok=True)
    args.plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + '\n',
                         encoding='utf-8', newline='\n')
    print(f'plan    : {args.plan}')
    for key in ('base_lr', 'scheduler_end', 'resumed_iter', 'total_iters'):
        print(f'  {key} = {plan[key]}')
    print('PREPARE OK')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', type=Path,
                        help='checkpoint to rewrite (the finished 20k run)')
    parser.add_argument('--output', type=Path, default=Path('_resume_nosched.pth'),
                        help='rewritten copy; the source is never modified')
    parser.add_argument('--plan', type=Path, default=Path('_resume_plan.json'),
                        help='JSON sidecar consumed by the launcher')
    parser.add_argument('--total-iters', type=int, default=60000,
                        help='target budget of the continued run')
    parser.add_argument('--base-lr', type=float, default=1e-4, dest='base_lr_config',
                        help='optimizer lr of the original run (default 1e-4)')
    parser.add_argument('--power', type=float, default=0.9,
                        help='PolyLR power of the original run (default 0.9)')
    parser.add_argument('--expect-iter', type=int, default=None,
                        help='assert the checkpoint stops at this iteration')
    parser.add_argument('--self-test', action='store_true',
                        help='verify the LR equivalence (stdlib only) and exit')
    args = parser.parse_args()
    if not args.self_test and args.source is None:
        parser.error('--source is required unless --self-test is used')
    return args


if __name__ == '__main__':
    parsed = parse_args()
    if parsed.self_test:
        self_test()
    else:
        prepare(parsed)
