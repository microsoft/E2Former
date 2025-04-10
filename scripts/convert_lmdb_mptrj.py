# -*- coding: utf-8 -*-
"""
Convert MPtrj dataset to lmdb format.
"""

import argparse
import json
import os
import pickle
from pathlib import Path

import lmdb
import numpy as np
import torch
from src.utils.mptrj_utils import mptrj_dict_to_pyg_data


def write_lmdb(mptrj, idxs, dirs):
    os.makedirs(dirs, exist_ok=True)
    print(f"Writing to {str(dirs / 'data.lmdb')}")
    db = lmdb.open(
        str(dirs / "data.lmdb"),
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )
    collect_meta = {
        "force": [],
        "uncorrected_total_energy": [],
        "corrected_total_energy": [],
        "energy_per_atom": [],
        "ef_per_atom": [],
        "e_per_atom_relaxed": [],
        "ef_per_atom_relaxed": [],
        "stress": [],
        "magmom": [],
        "bandgap": [],
    }
    i = 0
    for mp_id in mptrj.keys():
        for sid in mptrj[mp_id].keys():
            if sid in idxs:
                data = mptrj_dict_to_pyg_data(mptrj[mp_id][sid], sid)
                txn = db.begin(write=True)
                txn.put(f"{i}".encode("ascii"), pickle.dumps(data, protocol=-1))
                txn.commit()
                i += 1
                if i % 1000 == 0:
                    print(f"Processed {i} frames. Total {len(idxs)} frames.")
                for key in collect_meta.keys():
                    collect_meta[key].append(mptrj[mp_id][sid][key])

    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(i, protocol=-1))
    txn.commit()

    db.sync()
    db.close()
    return collect_meta


def process_meta(results, dirs):
    all_results = {}

    for key in results.keys():
        all_results[key] = []
        for i in range(len(results[key])):
            if results[key][i] is not None:
                all_results[key].append(torch.tensor(results[key][i]))
        if all_results[key][0].numel() > 1:
            all_results[key] = torch.cat(all_results[key])
        else:
            all_results[key] = torch.tensor(all_results[key])

    meta_data = {}
    for key in all_results.keys():
        meta_data[f"{key}_mean"] = float(all_results[key].mean())
        meta_data[f"{key}_std"] = float(all_results[key].std())
        meta_data[f"{key}_min"] = float(all_results[key].min())
        meta_data[f"{key}_max"] = float(all_results[key].max())

    with open(dirs / "meta_data.json", "w") as f:
        json.dump(meta_data, f)


def main(args):
    train_idx = np.load(args.train_idx)
    val_idx = np.load(args.val_idx)

    train_dir = Path(args.output_dir) / "train"
    val_dir = Path(args.output_dir) / "val"

    mptrj = json.load(open(args.mptrj_json, "r"))

    val_meta = write_lmdb(mptrj, val_idx, val_dir)
    process_meta(val_meta, val_dir)

    train_meta = write_lmdb(mptrj, train_idx, train_dir)
    process_meta(train_meta, train_dir)

    print("Done!")


if __name__ == "__main__":
    praser = argparse.ArgumentParser()
    praser.add_argument(
        "--mptrj_json", type=str, required=True, help="Path to the MPtrj json file."
    )
    praser.add_argument(
        "--train_idx",
        type=str,
        required=True,
        help="Path to the train index. (should be a numpy file)",
    )
    praser.add_argument(
        "--val_idx",
        type=str,
        required=True,
        help="Path to the val index. (should be a numpy file)",
    )
    praser.add_argument(
        "--output_dir", type=str, required=True, help="Path to the output directory."
    )
    main(praser.parse_args())
