# -*- coding: utf-8 -*-
"""
Analyze the rollouts of MD17 and MD22 datasets.
Modified from: https://github.com/kyonofx/MDsim/blob/main/observable.ipynb
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from ase.io import Trajectory
from ase.neighborlist import NeighborList, natural_cutoffs
from src.utils.plot_utils import setup_plot
from src.utils.simulation_utils import ase_traj_to_gsd

"""
functions for loading simulated trajectories and computing observables.
"""


def get_thermo(filename):
    """
    read thermo logs.
    """
    with open(filename, "r") as f:
        thermo = f.read().splitlines()
        sim_time, Et, Ep, Ek, T = [], [], [], [], []
        for i in range(1, len(thermo)):
            try:
                t, Etot, Epot, Ekin, Temp = [
                    float(x) for x in thermo[i].split(" ") if x
                ]
                sim_time.append(t)
                Et.append(Etot)
                Ep.append(Epot)
                Ek.append(Ekin)
                T.append(Temp)
            except ValueError:
                sim_time, Et, Ep, Ek, T = [], [], [], [], []
    thermo = {"time": sim_time, "Et": Et, "Ep": Ep, "Ek": Ek, "T": T}
    return thermo


def get_test_metrics(md_dir):
    """
    read test metrics such as force error.
    """
    run_metrics = {}
    with open(md_dir / "test_metric.json", "r") as f:
        test_metric = json.load(f)

        if "mae_f" in test_metric:
            fmae = test_metric["mae_f"]
            run_metrics["fmae"] = fmae
        elif "f_mae" in test_metric:
            fmae = test_metric["f_mae"]
            run_metrics["fmae"] = fmae
        elif "forces_mae" in test_metric:
            fmae = test_metric["forces_mae"]["metric"]
            run_metrics["fmae"] = fmae

        if "mae_e" in test_metric:
            emae = test_metric["mae_e"]
            run_metrics["emae"] = emae
        elif "e_mae" in test_metric:
            emae = test_metric["e_mae"]
            run_metrics["emae"] = emae
        elif "energy_mae" in test_metric:
            emae = test_metric["energy_mae"]["metric"]
            run_metrics["emae"] = emae

        if "num_params" in test_metric:
            run_metrics["n_params"] = test_metric["num_params"]
        if "running_time" in test_metric:
            run_metrics["running_time"] = test_metric["running_time"]
    return run_metrics


def mae(x, y, factor):
    return np.abs(x - y).mean() * factor


def distance_pbc(x0, x1, lattices):
    delta = torch.abs(x0 - x1)
    lattices = lattices.view(-1, 1, 3)
    delta = torch.where(delta > 0.5 * lattices, delta - lattices, delta)
    return torch.sqrt((delta**2).sum(dim=-1))


def get_diffusivity_traj(pos_seq, dilation=1):
    """
    Input: B x N x T x 3
    Output: B x T
    """
    # substract CoM
    bsize, time_steps = pos_seq.shape[0], pos_seq.shape[2]
    pos_seq = pos_seq - pos_seq.mean(1, keepdims=True)
    msd = (
        (pos_seq[:, :, 1:] - pos_seq[:, :, 0].unsqueeze(2))
        .pow(2)
        .sum(dim=-1)
        .mean(dim=1)
    )
    diff = msd / (torch.arange(1, time_steps) * dilation) / 6
    return diff.view(bsize, time_steps - 1)


def get_smoothed_diff(xyz):
    seq_len = xyz.shape[0] - 1
    diff = torch.zeros(seq_len)
    for i in range(seq_len):
        diff[: seq_len - i] += get_diffusivity_traj(
            xyz[i:].transpose(0, 1).unsqueeze(0)
        ).flatten()
    diff = diff / torch.flip(torch.arange(seq_len), dims=[0])
    return diff


def get_hr(traj, bins):
    """
    compute h(r) for MD17 simulations.
    traj: T x N_atoms x 3
    """
    pdist = torch.cdist(traj, traj).flatten()
    hist, _ = np.histogram(pdist[:].flatten().numpy(), bins, density=True)
    return hist


def load_run(md_dir, xlim, bins, stability_threshold, gt_traj, hist_gt):
    """
    md_dir: directory to the finished MD simulation.
    """
    if not isinstance(md_dir, Path):
        md_dir = Path(md_dir)

    model_name = md_dir.parts[-2]
    seed = md_dir.parts[-1][-1]
    run = {
        "name": (model_name + f"_seed_{seed}"),
    }

    # get bonds
    traj = Trajectory(md_dir / "atoms.traj")
    atoms = traj[0]
    NL = NeighborList(natural_cutoffs(atoms), self_interaction=False)
    NL.update(atoms)
    bonds = NL.get_connectivity_matrix().todense().nonzero()
    bonds = torch.tensor(bonds)

    # process trajectory
    traj = [x.positions for x in traj]
    run["traj"] = torch.from_numpy(np.stack(traj))

    # This breaks for some reason
    # run['traj'] = torch.unique(run['traj'], dim=0) # remove repeated frames from restarting.

    # load thermo log
    run["thermo"] = get_thermo(md_dir / "thermo.log")
    T = np.array(run["thermo"]["T"])
    collapse_pt = len(T)
    # md_time = np.array(run["thermo"]["time"])

    # track stability
    bond_lens = distance_pbc(
        gt_traj[:, bonds[0]],
        gt_traj[:, bonds[1]],
        torch.FloatTensor([30.0, 30.0, 30.0]),
    )
    mean_bond_lens = bond_lens.mean(dim=0)

    for i in range(1, len(T)):
        bond_lens = distance_pbc(
            run["traj"][(i - 1) : i, bonds[0]],
            run["traj"][(i - 1) : i, bonds[1]],
            torch.FloatTensor([30.0, 30.0, 30.0]),
        )
        max_dev = (bond_lens[0] - mean_bond_lens).abs().max()
        if max_dev > stability_threshold:
            collapse_pt = i
            break
    run["collapse_pt"] = collapse_pt

    # compute h(r)
    hist_pred = get_hr(run["traj"][0:collapse_pt], bins)
    hr_mae = mae(hist_pred, hist_gt, xlim)
    run["hr"] = hist_pred
    run["hr_error"] = hr_mae

    return run


def main(args):
    setup_plot()

    bins = np.linspace(1e-6, args.xlim, args.n_bins + 1)  # for computing h(r)

    gt_data = np.load(args.gt_traj)
    gt_traj = torch.FloatTensor(gt_data.f.R)
    hist_gt = get_hr(gt_traj, bins)

    md_dir = Path(args.md_dir)

    ase_traj_to_gsd(Trajectory(md_dir / "atoms.traj"), md_dir / "traj.gsd")

    run = load_run(md_dir, args.xlim, bins, args.stability_threshold, gt_traj, hist_gt)
    plt.plot(bins[1:], hist_gt, label="Reference", linewidth=2, linestyle="--")
    plt.plot(bins[1:], run["hr"], label="Prediction", linewidth=2, linestyle="--")
    plt.xlabel("r")
    plt.ylabel("h(r)")
    plt.legend()
    plt.savefig(md_dir / "hr.pdf", bbox_inches="tight")

    collapse_ps = (run["collapse_pt"] - 1) / 20
    hr_mae = run["hr_error"]
    results = {
        "stability": collapse_ps,
        "hr_mae": hr_mae,
    }
    with open(md_dir / "simulation_evals.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    praser = argparse.ArgumentParser()
    praser.add_argument(
        "--md_dir", type=str, required=True, help="Path to the MD simulation directory."
    )
    praser.add_argument(
        "--gt_traj",
        type=str,
        required=True,
        help="Path to the ground truth npz trajectory.",
    )
    praser.add_argument(
        "--stability_threshold",
        type=float,
        default=0.5,
        help="Threshold for detecting collapse.",
    )
    praser.add_argument("--xlim", type=float, default=15, help="Upper limit for h(r).")
    praser.add_argument(
        "--n_bins", type=int, default=500, help="Number of bins for h(r)."
    )
    main(praser.parse_args())
