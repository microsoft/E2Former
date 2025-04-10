# -*- coding: utf-8 -*-
import pickle
from pathlib import Path

import lmdb
import numpy as np
from MDsim.preprocessing.arrays_to_graphs import AtomsToGraphs
from sklearn.model_selection import train_test_split
from tqdm import tqdm

EV_TO_KCAL_MOL = 23.06052

MD22NAME = [
    "Ac-Ala3-NHMe",
    "AT-AT",
    "AT-AT-CG-CG",
    "buckyball-catcher",
    "DHA",
    "double-walled_nanotube",
    "stachyose",
]

SGDMLSIZE = {
    "Ac-Ala3-NHMe": 6000,
    "AT-AT": 3000,
    "AT-AT-CG-CG": 2000,
    "buckyball-catcher": 600,
    "DHA": 8000,
    "double-walled_nanotube": 800,
    "stachyose": 8000,
}

USESGDMLSIZE = True

DATAPATH = "/data/ericqu/md22/orig"
DBPATH = "/data/ericqu/md22/sgdml_lmdb/"


def write_to_lmdb(data_path, db_path, dataset_name):
    print(f"process MD22 molecule: {dataset_name}.")
    a2g = AtomsToGraphs(
        max_neigh=1000,
        radius=6,
        r_energy=True,
        r_forces=True,
        r_distances=False,
        r_edges=False,
        device="cpu",
    )

    data = np.load(f"{data_path}/md22_{dataset_name}.npz")

    n_points = data["R"].shape[0]
    atomic_numbers = data["z"]
    atomic_numbers = atomic_numbers.astype(np.int64)
    positions = data["R"]

    force = data["F"] / EV_TO_KCAL_MOL
    energy = data["E"] / EV_TO_KCAL_MOL

    energy = energy.reshape(-1, 1)  # Reshape energy into 2D array ADDED TODO

    lengths = np.ones(3)[None, :] * 30.0
    angles = np.ones(3)[None, :] * 90.0

    if USESGDMLSIZE:
        n_size = SGDMLSIZE[dataset_name]
        train_val_index = np.linspace(0, n_points - 1, n_size, dtype=int)
        test = np.setdiff1d(np.arange(n_points), train_val_index)
        np.random.shuffle(test)
        test = test.tolist()
    else:
        train_val_index = np.arange(n_points)
        test = []

    train, val = train_test_split(
        train_val_index, train_size=0.95, test_size=0.05, random_state=42
    )
    ranges = [train, val, test]

    norm_stats = {
        "e_mean": energy[train].mean(),
        "e_std": energy[train].std(),
        "f_mean": force[train].mean(),
        "f_std": force[train].std(),
    }

    save_path = Path(db_path) / dataset_name
    save_path.mkdir(parents=True, exist_ok=True)
    np.save(save_path / "metadata", norm_stats)

    for spidx, split in enumerate(["train", "val", "test"]):
        print(f"processing split {split}.")
        if len(ranges[spidx]) == 0:
            continue
        save_path = Path(db_path) / dataset_name / split
        save_path.mkdir(parents=True, exist_ok=True)
        db = lmdb.open(
            str(save_path / "data.lmdb"),
            map_size=1099511627776 * 2,
            subdir=False,
            meminit=False,
            map_async=True,
        )
        for i, idx in enumerate(tqdm(ranges[spidx])):
            natoms = np.array([positions.shape[1]] * 1, dtype=np.int64)
            data = a2g.convert(
                natoms,
                positions[idx],
                atomic_numbers,
                lengths,
                angles,
                energy[idx],
                force[idx],
            )
            data.sid = 0
            data.fid = idx

            txn = db.begin(write=True)
            txn.put(f"{i}".encode("ascii"), pickle.dumps(data, protocol=-1))
            txn.commit()

        # Save count of objects in lmdb.
        txn = db.begin(write=True)
        txn.put("length".encode("ascii"), pickle.dumps(i, protocol=-1))
        txn.commit()

        db.sync()
        db.close()


if __name__ == "__main__":
    for dataset_name in MD22NAME:
        write_to_lmdb(DATAPATH, DBPATH, dataset_name)
