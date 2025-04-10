# -*- coding: utf-8 -*-
"""
Get the runtime of a model for a given configuration.
Average the runtime over a number of runs.
"""

from __future__ import annotations

import json
import logging
import random
from typing import TYPE_CHECKING

import e3nn.o3 as o3
import numpy as np
import torch
import tqdm
from fairchem.core.common.flags import flags
from fairchem.core.common.utils import build_config, new_trainer_context, setup_logging
from fairchem.core.modules.evaluator import Evaluator

if TYPE_CHECKING:
    import argparse


def seed_everywhere(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_rot_mat(rot_type, device):
    if rot_type == "random":
        return o3.rand_matrix().to(device)
    elif rot_type == "x":
        return o3.matrix_x(torch.rand(1) * 2 * np.pi).to(device)[0]
    elif rot_type == "y":
        return o3.matrix_y(torch.rand(1) * 2 * np.pi).to(device)[0]
    elif rot_type == "z":
        return o3.matrix_z(torch.rand(1) * 2 * np.pi).to(device)[0]
    else:
        raise ValueError(f"Invalid rot_type: {rot_type}")


def test_equivariance(evaluator, trainer, batch, rot_type="random"):
    rot_mat = get_rot_mat(rot_type, trainer.device)

    trainer.model.eval()

    output_orig = trainer._forward(batch)
    output_orig["forces"] = torch.einsum("ij,kj->ki", rot_mat, output_orig["forces"])

    rot_batch = batch.clone()
    rot_batch.pos = torch.einsum("ij,kj->ki", rot_mat, rot_batch.pos)
    output_rot = trainer._forward(rot_batch)

    output_rot["natoms"] = batch.natoms

    return evaluator.eval(output_orig, output_rot)


def main():
    setup_logging()

    seed_everywhere(42)

    parser: argparse.ArgumentParser = flags.get_parser()
    parser.add_argument(
        "--steps",
        default=10,
        type=int,
        help="Number of batchs for equivariance check. Default: 10",
    )
    parser.add_argument(
        "--num_roataions",
        default=32,
        type=int,
        help="Number of random rotatoins for equivariance check. Default: 32",
    )
    parser.add_argument(
        "--batch-size",
        default=2,
        type=int,
        help="Batch size for equivariance chcek (will overwrite config). Default: 2",
    )
    args: argparse.Namespace
    override_args: list[str]
    args, override_args = parser.parse_known_args()
    args.debug = True
    config = build_config(args, override_args)
    config["optim"]["batch_size"] = args.batch_size
    if "batch_size" in config["model"]:
        config["model"]["batch_size"] = args.batch_size
        logging.warning(
            f"Overwriting batch_size in model config to {args.batch_size} for equivariance check"
        )

    if config["model"]["use_compile"]:
        logging.warning("Turning off compile for equivariance check")
        config["model"]["use_compile"] = False

    if config["model"]["use_pbc"]:
        logging.warning("Truning off PBC for equivariance check")
        config["model"]["use_pbc"] = False

    if config["model"]["max_neighbors"] < 50:
        logging.warning("Setting max_neighbors to 50 for equivariance check")
        config["model"]["max_neighbors"] = 50

    if "cut_off" in config["model"]:
        logging.warning("Setting cut_off to 5.0 for equivariance check")
        config["model"]["cut_off"] = 5.0

    if "max_radius" in config["model"]:
        logging.warning("Setting max_radius to 5.0 for equivariance check")
        config["model"]["max_radius"] = 5.0

    if config["checkpoint"] is None:
        logging.warning(
            "No checkpoint provided. Using random weights for equivariance check"
        )
        random_weights = True
    else:
        random_weights = False

    evaluator = Evaluator(task="s2ef")
    rot_types = ["random", "x", "y", "z"]
    metrics = {}
    for rot_type in rot_types:
        metrics[rot_type] = {}
    with new_trainer_context(config=config, distributed=False) as ctx:
        trainer = ctx.trainer
        if config["checkpoint"] is not None:
            trainer.load_checkpoint(config["checkpoint"])
        train_loader = iter(trainer.train_loader)
        for _ in tqdm.tqdm(range(args.steps)):
            batch = next(train_loader)
            for rot_type in rot_types:
                for _ in range(args.num_roataions):
                    new_metrics = test_equivariance(evaluator, trainer, batch, rot_type)
                    for key in new_metrics.keys():
                        metrics[rot_type] = evaluator.update(
                            key, new_metrics[key], metrics[rot_type]
                        )

    with open(
        f"{args.config_yml}-equivariance-test-{('random' if random_weights else 'trained')}.json",
        "w",
    ) as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    main()
