# -*- coding: utf-8 -*-
"""
Get the FLOPs of a model for a given configuration.
Average the FLOPs over a number of runs.
Ref: https://gist.github.com/Chillee/07b36672a0ca2d1280e42b8d10f23174#file-mfu_compute-py
"""

from __future__ import annotations

import json
import random
from typing import TYPE_CHECKING

import numpy as np
import torch
import tqdm
from fairchem.core.common.flags import flags
from fairchem.core.common.utils import build_config, new_trainer_context, setup_logging
from torch.utils.flop_counter import FlopCounterMode

if TYPE_CHECKING:
    import argparse


def seed_everywhere(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_flops_training(trainer, batch):
    flop_counter = FlopCounterMode(display=False)
    with flop_counter:
        trainer.model.train()
        out = trainer._forward(batch)

    flops_forward = flop_counter.get_total_flops()

    flop_counter = FlopCounterMode(display=False)
    with flop_counter:
        trainer.model.train()
        out = trainer._forward(batch)
        loss = trainer._compute_loss(out, batch)
        loss = trainer.scaler.scale(loss) if trainer.scaler else loss
        trainer._backward(loss)

    flops_total = flop_counter.get_total_flops()

    return flops_forward, flops_total


def main():
    setup_logging()

    seed_everywhere(42)

    parser: argparse.ArgumentParser = flags.get_parser()
    parser.add_argument(
        "--steps",
        default=100,
        type=int,
        help="Number of steps for FLOPs calculation. Default: 100",
    )
    parser.add_argument(
        "--batch-size",
        default=1,
        type=int,
        help="Batch size for FLOPs calculation (will overwrite config). Default: 1",
    )
    args: argparse.Namespace
    override_args: list[str]
    args, override_args = parser.parse_known_args()
    args.debug = True
    config = build_config(args, override_args)
    config["optim"]["batch_size"] = args.batch_size

    with new_trainer_context(config=config, distributed=False) as ctx:
        trainer = ctx.trainer
        flops_forward = 0
        flops_total = 0
        train_loader = iter(trainer.train_loader)
        for _ in tqdm.tqdm(range(args.steps)):
            batch = next(train_loader)
            flops_f, flops_t = get_flops_training(trainer, batch)
            flops_forward += flops_f
            flops_total += flops_t

        flops_forward /= args.steps
        flops_total /= args.steps

        result = {
            "flops_forward": flops_forward,
            "flops_total": flops_total,
        }
        with open(f"{args.config_yml}-flops.json", "w") as f:
            json.dump(result, f)


if __name__ == "__main__":
    main()
