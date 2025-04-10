# -*- coding: utf-8 -*-
"""
Get the Runtime of a model for a given configuration.
Average the Runtime over a number of runs.
"""

from __future__ import annotations

import json
import random
import time
from typing import TYPE_CHECKING

import numpy as np
import torch
import tqdm
from fairchem.core.common.flags import flags
from fairchem.core.common.utils import build_config, new_trainer_context, setup_logging

if TYPE_CHECKING:
    import argparse


def seed_everywhere(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def model_step(trainer, batch):
    trainer.model.train()
    with torch.amp.autocast("cuda", enabled=trainer.scaler is not None):
        out = trainer._forward(batch)
        loss = trainer._compute_loss(out, batch)
    loss = trainer.scaler.scale(loss) if trainer.scaler else loss
    trainer._backward(loss)


def main():
    setup_logging()

    seed_everywhere(42)

    parser: argparse.ArgumentParser = flags.get_parser()
    parser.add_argument(
        "--steps",
        default=8,
        type=int,
        help="Number of steps for runtime benchmark. Default: 8",
    )
    parser.add_argument(
        "--max-batch-size",
        default=4,
        type=int,
        help="Max batch size for runtime benchmark (will overwrite config). Default: 4",
    )
    args: argparse.Namespace
    override_args: list[str]
    args, override_args = parser.parse_known_args()
    args.debug = True
    config = build_config(args, override_args)

    result = {}

    for batch_size in range(1, args.max_batch_size + 1):
        print(f"Batch size: {batch_size}")
        config["optim"]["batch_size"] = batch_size

        with new_trainer_context(config=config, distributed=False) as ctx:
            trainer = ctx.trainer
            times = []
            mems = []

            # test
            print("Benchmark")
            train_loader = iter(trainer.train_loader)
            for _ in tqdm.tqdm(range(args.steps)):
                batch = next(train_loader)
                batch.to(trainer.device)
                # warmup
                for _ in range(4):
                    model_step(trainer, batch)
                # benchmark
                torch.cuda.reset_max_memory_allocated()
                torch.cuda.synchronize()
                start = time.time()
                for _ in range(8):
                    model_step(trainer, batch)
                torch.cuda.synchronize()
                end = time.time()
                ms_per_iter = (end - start) / 8 * 1000

                times.append(ms_per_iter)
                mems.append(torch.cuda.max_memory_allocated() / 1024 / 1024)
                # torch.cuda.empty_cache()

            times = np.array(times)
            mems = np.array(mems)

            result[batch_size] = {
                "mean_time": np.mean(times),
                "std_time": np.std(times),
                "mean_mem": np.mean(mems),
                "std_mem": np.std(mems),
                "max_mem": max(mems),
            }

        with open(f"{args.config_yml}-runtime.json", "w") as f:
            json.dump(result, f)


if __name__ == "__main__":
    main()
