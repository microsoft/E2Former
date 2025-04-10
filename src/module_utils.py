# -*- coding: utf-8 -*-
import logging
import math
import warnings

import e3nn
import numpy as np
import scipy.special as sp
import torch
from e3nn import o3
from e3nn.util.jit import compile_mode
from fairchem.core.models.equiformer_v2.activation import GateActivation
from fairchem.core.models.equiformer_v2.so3 import (
    CoefficientMappingModule,
    FromS2Grid,
    SO3_LinearV2,
    ToS2Grid,
)
from fairchem.core.models.escn.so3 import SO3_Embedding
from torch import nn
from torch_cluster import radius_graph
from torch_geometric.data import Data

from .tensor_product import Simple_TensorProduct

# from fairchem.core.models.escn.so3 import SO3_Grid


class SO3_Grid(torch.nn.Module):
    """
    Helper functions for grid representation of the irreps

    Args:
        lmax (int):   Maximum degree of the spherical harmonics
        mmax (int):   Maximum order of the spherical harmonics
    """

    def __init__(
        self,
        lmax,
        mmax,
        normalization="integral",
        resolution=None,
    ):
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.lat_resolution = 2 * (self.lmax + 1)
        if lmax == mmax:
            self.long_resolution = 2 * (self.mmax + 1) + 1
        else:
            self.long_resolution = 2 * (self.mmax) + 1
        if resolution is not None:
            self.lat_resolution = resolution
            self.long_resolution = resolution

        self.mapping = CoefficientMappingModule([self.lmax], [self.lmax])

        device = "cpu"

        to_grid = ToS2Grid(
            self.lmax,
            (self.lat_resolution, self.long_resolution),
            normalization=normalization,  # normalization="integral",
            device=device,
        )
        to_grid_mat = torch.einsum("mbi, am -> bai", to_grid.shb, to_grid.sha).detach()
        # rescale based on mmax
        if lmax != mmax:
            for l in range(lmax + 1):
                if l <= mmax:
                    continue
                start_idx = l**2
                length = 2 * l + 1
                rescale_factor = math.sqrt(length / (2 * mmax + 1))
                to_grid_mat[:, :, start_idx : (start_idx + length)] = (
                    to_grid_mat[:, :, start_idx : (start_idx + length)] * rescale_factor
                )
        to_grid_mat = to_grid_mat[
            :, :, self.mapping.coefficient_idx(self.lmax, self.mmax)
        ]

        from_grid = FromS2Grid(
            (self.lat_resolution, self.long_resolution),
            self.lmax,
            normalization=normalization,  # normalization="integral",
            device=device,
        )
        from_grid_mat = torch.einsum(
            "am, mbi -> bai", from_grid.sha, from_grid.shb
        ).detach()
        # rescale based on mmax
        if lmax != mmax:
            for l in range(lmax + 1):
                if l <= mmax:
                    continue
                start_idx = l**2
                length = 2 * l + 1
                rescale_factor = math.sqrt(length / (2 * mmax + 1))
                from_grid_mat[:, :, start_idx : (start_idx + length)] = (
                    from_grid_mat[:, :, start_idx : (start_idx + length)]
                    * rescale_factor
                )
        from_grid_mat = from_grid_mat[
            :, :, self.mapping.coefficient_idx(self.lmax, self.mmax)
        ]

        # save tensors and they will be moved to GPU
        self.register_buffer("to_grid_mat", to_grid_mat)
        self.register_buffer("from_grid_mat", from_grid_mat)

    # Compute matrices to transform irreps to grid
    def get_to_grid_mat(self, device):
        return self.to_grid_mat

    # Compute matrices to transform grid to irreps
    def get_from_grid_mat(self, device):
        return self.from_grid_mat

    # Compute grid from irreps representation
    def to_grid(self, embedding, lmax, mmax):
        to_grid_mat = self.to_grid_mat[:, :, self.mapping.coefficient_idx(lmax, mmax)]
        grid = torch.einsum("bai, zic -> zbac", to_grid_mat, embedding)
        return grid

    # Compute irreps from grid representation
    def from_grid(self, grid, lmax, mmax):
        from_grid_mat = self.from_grid_mat[
            :, :, self.mapping.coefficient_idx(lmax, mmax)
        ]
        embedding = torch.einsum("bai, zbac -> zic", from_grid_mat, grid)
        return embedding


# -*- coding: utf-8 -*-
"""
    1. Normalize features of shape (N, sphere_basis, C),
    with sphere_basis = (lmax + 1) ** 2.

    2. The difference from `layer_norm.py` is that all type-L vectors have
    the same number of channels and input features are of shape (N, sphere_basis, C).
"""


@torch.jit.script
def mask_after_k_persample(n_sample: int, n_len: int, persample_k: torch.Tensor):
    assert persample_k.shape[0] == n_sample
    assert persample_k.max() <= n_len
    device = persample_k.device
    mask = torch.zeros([n_sample, n_len + 1], device=device)
    mask[torch.arange(n_sample, device=device), persample_k] = 1
    mask = mask.cumsum(dim=1)[:, :-1]
    return mask.type(torch.bool)


# follow PSM
class CellExpander:
    def __init__(
        self,
        cutoff=10.0,
        expanded_token_cutoff=512,
        pbc_expanded_num_cell_per_direction=10,
        pbc_multigraph_cutoff=10.0,
    ):
        self.cells = []
        for i in range(
            -pbc_expanded_num_cell_per_direction,
            pbc_expanded_num_cell_per_direction + 1,
        ):
            for j in range(
                -pbc_expanded_num_cell_per_direction,
                pbc_expanded_num_cell_per_direction + 1,
            ):
                for k in range(
                    -pbc_expanded_num_cell_per_direction,
                    pbc_expanded_num_cell_per_direction + 1,
                ):
                    if i == 0 and j == 0 and k == 0:
                        continue
                    self.cells.append([i, j, k])

        self.cells = torch.tensor(self.cells)

        self.cell_mask_for_pbc = self.cells != 0

        self.candidate_cells = torch.tensor(
            [
                [i, j, k]
                for i in range(
                    -pbc_expanded_num_cell_per_direction,
                    pbc_expanded_num_cell_per_direction + 1,
                )
                for j in range(
                    -pbc_expanded_num_cell_per_direction,
                    pbc_expanded_num_cell_per_direction + 1,
                )
                for k in range(
                    -pbc_expanded_num_cell_per_direction,
                    pbc_expanded_num_cell_per_direction + 1,
                )
            ]
        )

        self.cutoff = cutoff

        self.expanded_token_cutoff = expanded_token_cutoff

        self.pbc_multigraph_cutoff = pbc_multigraph_cutoff

        self.pbc_expanded_num_cell_per_direction = pbc_expanded_num_cell_per_direction

        self.conflict_cell_offsets = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    if i != 0 or j != 0 or k != 0:
                        self.conflict_cell_offsets.append([i, j, k])
        self.conflict_cell_offsets = torch.tensor(self.conflict_cell_offsets)  # 26 x 3

        conflict_to_consider = self.cells.unsqueeze(
            1
        ) - self.conflict_cell_offsets.unsqueeze(
            0
        )  # num_expand_cell x 26 x 3
        conflict_to_consider_mask = (
            ((conflict_to_consider * self.cells.unsqueeze(1)) >= 0)
            & (torch.abs(conflict_to_consider) <= self.cells.unsqueeze(1).abs())
        ).all(
            dim=-1
        )  # num_expand_cell x 26
        conflict_to_consider_mask &= (
            (conflict_to_consider <= pbc_expanded_num_cell_per_direction)
            & (conflict_to_consider >= -pbc_expanded_num_cell_per_direction)
        ).all(
            dim=-1
        )  # num_expand_cell x 26
        self.conflict_to_consider_mask = conflict_to_consider_mask

    def polynomial(self, dist: torch.Tensor, cutoff: float) -> torch.Tensor:
        """
        Polynomial cutoff function,ref: https://arxiv.org/abs/2204.13639
        Args:
            dist (tf.Tensor): distance tensor
            cutoff (float): cutoff distance
        Returns: polynomial cutoff functions
        """
        ratio = torch.div(dist, cutoff)
        result = (
            1
            - 6 * torch.pow(ratio, 5)
            + 15 * torch.pow(ratio, 4)
            - 10 * torch.pow(ratio, 3)
        )
        return torch.clamp(result, min=0.0)

    def _get_cell_tensors(self, cell, use_local_attention):
        # fitler impossible offsets according to cell size and cutoff
        def _get_max_offset_for_dim(cell, dim):
            lattice_vec_0 = cell[:, dim, :]
            lattice_vec_1_2 = cell[
                :, torch.arange(3, dtype=torch.long, device=cell.device) != dim, :
            ]
            normal_vec = torch.cross(
                lattice_vec_1_2[:, 0, :], lattice_vec_1_2[:, 1, :], dim=-1
            )
            normal_vec = normal_vec / normal_vec.norm(dim=-1, keepdim=True)
            cutoff = self.pbc_multigraph_cutoff if use_local_attention else self.cutoff

            max_offset = int(
                torch.max(
                    torch.ceil(
                        cutoff
                        / torch.abs(torch.sum(normal_vec * lattice_vec_0, dim=-1))
                    )
                )
            )
            return max_offset

        max_offsets = []
        for i in range(3):
            try:
                max_offset = _get_max_offset_for_dim(cell, i)
            except Exception as e:
                logging.warning(f"{e} with cell {cell}")
                max_offset = self.pbc_expanded_num_cell_per_direction
            max_offsets.append(max_offset)
        max_offsets = torch.tensor(max_offsets, device=cell.device)
        self.cells = self.cells.to(device=cell.device)
        self.cell_mask_for_pbc = self.cell_mask_for_pbc.to(device=cell.device)
        mask = (self.cells.abs() <= max_offsets).all(dim=-1)
        selected_cell = self.cells[mask, :]
        return selected_cell, self.cell_mask_for_pbc[mask, :], mask

    def _get_conflict_mask(self, cell, pos, atoms):
        batch_size, max_num_atoms = pos.size()[:2]
        self.conflict_cell_offsets = self.conflict_cell_offsets.to(device=pos.device)
        self.conflict_to_consider_mask = self.conflict_to_consider_mask.to(
            device=pos.device
        )
        offset = torch.bmm(
            self.conflict_cell_offsets.unsqueeze(0)
            .repeat(batch_size, 1, 1)
            .to(dtype=cell.dtype),
            cell,
        )  # batch_size x 26 x 3
        expand_pos = (pos.unsqueeze(1) + offset.unsqueeze(2)).reshape(
            batch_size, -1, 3
        )  # batch_size x max_num_atoms x 3, batch_size x 26 x 3 -> batch_size x (26 x max_num_atoms) x 3
        expand_dist = (pos.unsqueeze(2) - expand_pos.unsqueeze(1)).norm(
            p=2, dim=-1
        )  # batch_size x max_num_atoms x (26 x max_num_atoms)

        expand_atoms = atoms.repeat(
            1, self.conflict_cell_offsets.size()[0]
        )  # batch_size x (26 x max_num_atoms)
        atoms_identical_mask = atoms.unsqueeze(-1) == expand_atoms.unsqueeze(
            1
        )  # batch_size x max_num_atoms x (26 x max_num_atoms)

        conflict_mask = (
            ((expand_dist < 1e-5) & atoms_identical_mask)
            .any(dim=1)
            .reshape(batch_size, -1, max_num_atoms)
        )  # batch_size x 26 x max_num_atoms
        all_conflict_mask = (
            torch.bmm(
                self.conflict_to_consider_mask.unsqueeze(0)
                .to(dtype=cell.dtype)
                .repeat(batch_size, 1, 1),
                conflict_mask.to(dtype=cell.dtype),
            )
            .long()
            .bool()
        )  # batch_size x num_expand_cell x 26, batch_size x 26 x max_num_atoms -> batch_size x num_expand_cell x max_num_atoms
        return all_conflict_mask

    def check_conflict(self, pos, atoms, pbc_expand_batched):
        # ensure that there's no conflict in the expanded atoms
        # a conflict means that two atoms (or special tokens) share both the same position and token type
        expand_pos = pbc_expand_batched["expand_pos"]
        all_pos = torch.cat([pos, expand_pos], dim=1)
        num_expanded_atoms = all_pos.size()[1]
        all_dist = (all_pos.unsqueeze(1) - all_pos.unsqueeze(2)).norm(p=2, dim=-1)
        outcell_index = pbc_expand_batched[
            "outcell_index"
        ]  # batch_size x expanded_max_num_atoms
        all_atoms = torch.cat(
            [atoms, torch.gather(atoms, dim=-1, index=outcell_index)], dim=-1
        )
        atom_identical_mask = all_atoms.unsqueeze(1) == all_atoms.unsqueeze(-1)
        full_mask = torch.cat([atoms.eq(0), pbc_expand_batched["expand_mask"]], dim=-1)
        atom_identical_mask = atom_identical_mask.masked_fill(
            full_mask.unsqueeze(-1), False
        )
        atom_identical_mask = atom_identical_mask.masked_fill(
            full_mask.unsqueeze(1), False
        )
        conflict_mask = (all_dist < 1e-5) & atom_identical_mask
        conflict_mask[
            :,
            torch.arange(num_expanded_atoms, device=all_pos.device),
            torch.arange(num_expanded_atoms, device=all_pos.device),
        ] = False
        assert ~(
            conflict_mask.any()
        ), f"{all_dist[conflict_mask]} {all_atoms[conflict_mask.any(dim=-2)]}"

    def expand(
        self,
        pos,
        init_pos,
        pbc,
        num_atoms,
        atoms,
        cell,
        pair_token_type,
        use_local_attention=True,
        use_grad=False,
    ):
        with torch.set_grad_enabled(use_grad):
            pos = pos.float()
            cell = cell.float()
            batch_size, max_num_atoms = pos.size()[:2]
            cell_tensor, cell_mask, selected_cell_mask = self._get_cell_tensors(
                cell, use_local_attention
            )

            if not use_local_attention:
                all_conflict_mask = self._get_conflict_mask(cell, pos, atoms)
                all_conflict_mask = all_conflict_mask[:, selected_cell_mask, :].reshape(
                    batch_size, -1
                )
            # if expand_includeself:
            #     cell_tensor = torch.cat([torch.zeros((1,3),device = cell_tensor.device),cell_tensor],dim = 0)
            #     cell_mask = torch.cat([torch.ones((1,3),device = cell_mask.device).bool(),cell_mask],dim = 0)

            cell_tensor = (
                cell_tensor.unsqueeze(0).repeat(batch_size, 1, 1).to(dtype=cell.dtype)
            )
            num_expanded_cell = cell_tensor.size()[1]
            offset = torch.bmm(cell_tensor, cell)  # B x num_expand_cell x 3
            expand_pos = pos.unsqueeze(1) + offset.unsqueeze(
                2
            )  # B x num_expand_cell x T x 3
            expand_pos = expand_pos.view(
                batch_size, -1, 3
            )  # B x (num_expand_cell x T) x 3

            # eliminate duplicate atoms of expanded atoms, comparing with the original unit cell
            expand_dist = torch.norm(
                pos.unsqueeze(2) - expand_pos.unsqueeze(1), p=2, dim=-1
            )  # B x T x (num_expand_cell x T)
            expand_atoms = atoms.repeat(1, num_expanded_cell)
            expand_atom_identical = atoms.unsqueeze(-1) == expand_atoms.unsqueeze(1)
            expand_mask = (
                expand_dist
                < (self.pbc_multigraph_cutoff if use_local_attention else self.cutoff)
            ) & (
                (expand_dist > 1e-5) | ~expand_atom_identical
            )  # B x T x (num_expand_cell x T)
            expand_mask = torch.masked_fill(
                expand_mask, atoms.eq(0).unsqueeze(-1), False
            )
            expand_mask = torch.sum(expand_mask, dim=1) > 0
            if not use_local_attention:
                expand_mask = expand_mask & (~all_conflict_mask)
            expand_mask = expand_mask & (
                ~(atoms.eq(0).repeat(1, num_expanded_cell))
            )  # B x (num_expand_cell x T)

            cell_mask = (
                torch.all(pbc.unsqueeze(1) >= cell_mask.unsqueeze(0), dim=-1)
                .unsqueeze(-1)
                .repeat(1, 1, max_num_atoms)
                .reshape(expand_mask.size())
            )  # B x (num_expand_cell x T)
            expand_mask &= cell_mask
            expand_len = torch.sum(expand_mask, dim=-1)

            threshold_num_expanded_token = torch.clamp(
                self.expanded_token_cutoff - num_atoms, min=0
            )

            max_expand_len = torch.max(expand_len)

            # cutoff within expanded_token_cutoff tokens
            need_threshold = expand_len > threshold_num_expanded_token
            if need_threshold.any():
                min_expand_dist = expand_dist.masked_fill(expand_dist <= 1e-5, np.inf)
                expand_dist_mask = (
                    atoms.eq(0).unsqueeze(-1) | atoms.eq(0).unsqueeze(1)
                ).repeat(1, 1, num_expanded_cell)
                min_expand_dist = min_expand_dist.masked_fill_(expand_dist_mask, np.inf)
                min_expand_dist = min_expand_dist.masked_fill_(
                    ~cell_mask.unsqueeze(1), np.inf
                )
                min_expand_dist = torch.min(min_expand_dist, dim=1)[0]

                need_threshold_distances = min_expand_dist[
                    need_threshold
                ]  # B x (num_expand_cell x T)
                threshold_num_expanded_token = threshold_num_expanded_token[
                    need_threshold
                ]
                threshold_dist = torch.sort(
                    need_threshold_distances, dim=-1, descending=False
                )[0]

                threshold_dist = torch.gather(
                    threshold_dist, 1, threshold_num_expanded_token.unsqueeze(-1)
                )

                new_expand_mask = min_expand_dist[need_threshold] < threshold_dist
                expand_mask[need_threshold] &= new_expand_mask
                expand_len = torch.sum(expand_mask, dim=-1)
                max_expand_len = torch.max(expand_len)

            outcell_index = torch.zeros(
                [batch_size, max_expand_len], dtype=torch.long, device=pos.device
            )
            expand_pos_compressed = torch.zeros(
                [batch_size, max_expand_len, 3], dtype=pos.dtype, device=pos.device
            )
            outcell_all_index = torch.arange(
                max_num_atoms, dtype=torch.long, device=pos.device
            ).repeat(num_expanded_cell)
            for i in range(batch_size):
                outcell_index[i, : expand_len[i]] = outcell_all_index[expand_mask[i]]
                # assert torch.all(outcell_index[i, :expand_len[i]] < natoms[i])
                expand_pos_compressed[i, : expand_len[i], :] = expand_pos[
                    i, expand_mask[i], :
                ]

            expand_pair_token_type = torch.gather(
                pair_token_type,
                dim=2,
                index=outcell_index.unsqueeze(1)
                .unsqueeze(-1)
                .repeat(1, max_num_atoms, 1, pair_token_type.size()[-1]),
            )
            expand_node_type_edge = torch.cat(
                [pair_token_type, expand_pair_token_type], dim=2
            )

            if use_local_attention:
                dist = (pos.unsqueeze(2) - pos.unsqueeze(1)).norm(p=2, dim=-1)
                expand_dist_compress = (
                    pos.unsqueeze(2) - expand_pos_compressed.unsqueeze(1)
                ).norm(p=2, dim=-1)
                local_attention_weight = self.polynomial(
                    torch.cat([dist, expand_dist_compress], dim=2),
                    cutoff=self.pbc_multigraph_cutoff,
                )
                is_periodic = pbc.any(dim=-1)
                local_attention_weight = local_attention_weight.masked_fill(
                    ~is_periodic.unsqueeze(-1).unsqueeze(-1), 1.0
                )
                local_attention_weight = local_attention_weight.masked_fill(
                    atoms.eq(0).unsqueeze(-1), 1.0
                )
                expand_mask = mask_after_k_persample(
                    batch_size, max_expand_len, expand_len
                )
                full_mask = torch.cat([atoms.eq(0), expand_mask], dim=-1)
                local_attention_weight = local_attention_weight.masked_fill(
                    atoms.eq(0).unsqueeze(-1), 1.0
                )
                local_attention_weight = local_attention_weight.masked_fill(
                    full_mask.unsqueeze(1), 0.0
                )
                pbc_expand_batched = {
                    "expand_pos": expand_pos_compressed,
                    "outcell_index": outcell_index,
                    "expand_mask": expand_mask,
                    "local_attention_weight": local_attention_weight,
                    "expand_node_type_edge": expand_node_type_edge,
                }
            else:
                pbc_expand_batched = {
                    "expand_pos": expand_pos_compressed,
                    "outcell_index": outcell_index,
                    "expand_mask": mask_after_k_persample(
                        batch_size, max_expand_len, expand_len
                    ),
                    "local_attention_weight": None,
                    "expand_node_type_edge": expand_node_type_edge,
                }

            expand_pos_no_offset = torch.gather(
                pos, dim=1, index=outcell_index.unsqueeze(-1)
            )
            offset = expand_pos_compressed - expand_pos_no_offset
            init_expand_pos_no_offset = torch.gather(
                init_pos, dim=1, index=outcell_index.unsqueeze(-1)
            )
            init_expand_pos = init_expand_pos_no_offset + offset
            init_expand_pos = init_expand_pos.masked_fill(
                pbc_expand_batched["expand_mask"].unsqueeze(-1),
                0.0,
            )

            pbc_expand_batched["init_expand_pos"] = init_expand_pos

            # # self.check_conflict(pos, atoms, pbc_expand_batched)
            # print(f"local attention weight {local_attention_weight.numel()} zero:{torch.sum(local_attention_weight==0)}")
            # # print(torch.sum(local_attention_weight==0,dim = 1)==(local_attention_weight.shape[1]))
            # print("N1+N2, ",local_attention_weight.shape[2],torch.sum(
            #     torch.sum(local_attention_weight==0,dim = 1)==local_attention_weight.shape[1])/(local_attention_weight.shape[0]*1.0))

            return pbc_expand_batched

    def expand_includeself(
        self,
        pos,
        init_pos,
        pbc,
        num_atoms,
        atoms,
        cell,
        neighbors_radius,
        pair_token_type=None,
        use_topK=False,
        use_local_attention=True,
        use_grad=False,
        padding_mask=None,
    ):
        with torch.set_grad_enabled(use_grad):
            pos = torch.where(
                padding_mask.unsqueeze(dim=-1).repeat(1, 1, 3), 999.0, pos.float()
            )
            # pos = pos.float()
            cell = cell.float()
            batch_size, max_num_atoms = pos.size()[:2]
            cell_tensor, cell_mask, selected_cell_mask = self._get_cell_tensors(
                cell, use_local_attention
            )

            # if not use_local_attention:
            #     all_conflict_mask = self._get_conflict_mask(cell, pos, atoms)
            #     all_conflict_mask = all_conflict_mask[:, selected_cell_mask, :].reshape(
            #         batch_size, -1
            #     )
            cell_tensor = torch.cat(
                [torch.zeros((1, 3), device=cell_tensor.device), cell_tensor], dim=0
            )
            # self.cell_mask_for_pbc = self.cells != 0
            cell_mask = torch.cat(
                [torch.zeros((1, 3), device=cell_mask.device).bool(), cell_mask], dim=0
            )

            cell_tensor = (
                cell_tensor.unsqueeze(0).repeat(batch_size, 1, 1).to(dtype=cell.dtype)
            )
            num_expanded_cell = cell_tensor.size()[1]
            offset = torch.bmm(cell_tensor, cell)  # B x num_expand_cell x 3
            expand_pos = pos.unsqueeze(1) + offset.unsqueeze(
                2
            )  # B x num_expand_cell x T x 3
            expand_pos = expand_pos.view(
                batch_size, -1, 3
            )  # B x (num_expand_cell x T) x 3

            # eliminate duplicate atoms of expanded atoms, comparing with the original unit cell
            expand_dist = torch.norm(
                pos.unsqueeze(2) - expand_pos.unsqueeze(1), p=2, dim=-1
            )  # B x T x (num_expand_cell x T)
            # expand_atoms = atoms.repeat(1, num_expanded_cell)
            # expand_atom_identical = atoms.unsqueeze(-1) == expand_atoms.unsqueeze(1)
            if use_topK:
                # lowest dis is 0, for node itself - node itself, thus topk+1
                values, _ = torch.topk(
                    expand_dist, neighbors_radius[0] + 1, dim=-1, largest=False
                )
                expand_mask = (
                    expand_dist <= (values[:, :, neighbors_radius[0]].unsqueeze(dim=-1))
                ) & (expand_dist < neighbors_radius[1])
                # & (expand_dist > 1e-5)
                #     (
                #     (expand_dist > 1e-5) | ~expand_atom_identical
                # )  # B x T x (num_expand_cell x T)

            else:
                expand_mask = expand_dist < (
                    self.pbc_multigraph_cutoff if use_local_attention else self.cutoff
                )
                # & (
                #     (expand_dist > 1e-5)
                # )
                # | ~expand_atom_identical# B x T x (num_expand_cell x T)

            expand_mask = (
                expand_mask
                & (~padding_mask.repeat(1, num_expanded_cell).unsqueeze(1))
                & (~(atoms.eq(0).unsqueeze(-1)))
            )

            expand_mask = torch.sum(expand_mask, dim=1) > 0
            # if not use_local_attention:
            #     expand_mask = expand_mask & (~all_conflict_mask)
            expand_mask = expand_mask & (
                ~(atoms.eq(0).repeat(1, num_expanded_cell))
            )  # B x (num_expand_cell x T)

            cell_mask = (
                torch.all(pbc.unsqueeze(1) >= cell_mask.unsqueeze(0), dim=-1)
                .unsqueeze(-1)
                .repeat(1, 1, max_num_atoms)
                .reshape(expand_mask.size())
            )  # B x (num_expand_cell x T)
            expand_mask &= cell_mask
            expand_len = torch.sum(expand_mask, dim=-1)

            threshold_num_expanded_token = torch.clamp(
                self.expanded_token_cutoff - num_atoms * 0, min=0
            )

            max_expand_len = torch.max(expand_len)

            # cutoff within expanded_token_cutoff tokens
            need_threshold = expand_len > threshold_num_expanded_token
            if need_threshold.any():
                min_expand_dist = expand_dist.masked_fill(expand_dist <= 1e-5, np.inf)
                expand_dist_mask = (
                    atoms.eq(0).unsqueeze(-1) | atoms.eq(0).unsqueeze(1)
                ).repeat(1, 1, num_expanded_cell)
                min_expand_dist = min_expand_dist.masked_fill_(expand_dist_mask, np.inf)
                min_expand_dist = min_expand_dist.masked_fill_(
                    ~cell_mask.unsqueeze(1), np.inf
                )
                min_expand_dist = torch.min(min_expand_dist, dim=1)[0]

                need_threshold_distances = min_expand_dist[
                    need_threshold
                ]  # B x (num_expand_cell x T)
                threshold_num_expanded_token = threshold_num_expanded_token[
                    need_threshold
                ]
                threshold_dist = torch.sort(
                    need_threshold_distances, dim=-1, descending=False
                )[0]

                threshold_dist = torch.gather(
                    threshold_dist, 1, threshold_num_expanded_token.unsqueeze(-1).long()
                )

                new_expand_mask = min_expand_dist[need_threshold] < threshold_dist
                expand_mask[need_threshold] &= new_expand_mask
                expand_len = torch.sum(expand_mask, dim=-1)
                max_expand_len = torch.max(expand_len)

            outcell_index = torch.zeros(
                [batch_size, max_expand_len], dtype=torch.long, device=pos.device
            )
            expand_pos_compressed = torch.zeros(
                [batch_size, max_expand_len, 3], dtype=pos.dtype, device=pos.device
            )
            outcell_all_index = torch.arange(
                max_num_atoms, dtype=torch.long, device=pos.device
            ).repeat(num_expanded_cell)
            for i in range(batch_size):
                outcell_index[i, : expand_len[i]] = outcell_all_index[expand_mask[i]]
                # assert torch.all(outcell_index[i, :expand_len[i]] < natoms[i])
                expand_pos_compressed[i, : expand_len[i], :] = expand_pos[
                    i, expand_mask[i], :
                ]
            # expand_pair_token_type = torch.gather(
            #     pair_token_type,
            #     dim=2,
            #     index=outcell_index.unsqueeze(1)
            #     .unsqueeze(-1)
            #     .repeat(1, max_num_atoms, 1, pair_token_type.size()[-1]),
            # )
            # expand_node_type_edge = torch.cat(
            #     [pair_token_type, expand_pair_token_type], dim=2
            # )

            # if use_local_attention:
            #     expand_dist_compress = (
            #         pos.unsqueeze(2) - expand_pos_compressed.unsqueeze(1)
            #     ).norm(p=2, dim=-1)
            #     local_attention_weight = self.polynomial(
            #         expand_dist_compress,
            #         cutoff=self.pbc_multigraph_cutoff,
            #     )
            #     is_periodic = pbc.any(dim=-1)
            #     local_attention_weight = local_attention_weight.masked_fill(
            #         ~is_periodic.unsqueeze(-1).unsqueeze(-1), 1.0
            #     )
            #     local_attention_weight = local_attention_weight.masked_fill(
            #         atoms.eq(0).unsqueeze(-1), 1.0
            #     )
            #     expand_mask = mask_after_k_persample(
            #         batch_size, max_expand_len, expand_len
            #     )
            #     local_attention_weight = local_attention_weight.masked_fill(
            #         atoms.eq(0).unsqueeze(-1), 1.0
            #     )
            #     local_attention_weight = local_attention_weight.masked_fill(
            #         expand_mask.unsqueeze(1), 0.0
            #     )
            #     pbc_expand_batched = {
            #         "expand_pos": expand_pos_compressed,
            #         "outcell_index": outcell_index,
            #         "expand_mask": expand_mask,
            #         "local_attention_weight": local_attention_weight,
            #         "expand_node_type_edge": expand_node_type_edge,
            #     }
            # else:
            pbc_expand_batched = {
                "expand_pos": expand_pos_compressed,
                "outcell_index": outcell_index,
                "expand_mask": mask_after_k_persample(
                    batch_size, max_expand_len, expand_len
                ),
                "local_attention_weight": None,
                # "expand_node_type_edge": expand_node_type_edge,
            }
            # print(pbc_expand_batched["expand_mask"],
            #       torch.sum(local_attention_weight==0,dim = 1)!=local_attention_weight.shape[1])

            # expand_pos_no_offset = torch.gather(
            #     pos, dim=1, index=outcell_index.unsqueeze(-1)
            # )
            # offset = expand_pos_compressed - expand_pos_no_offset
            # init_expand_pos_no_offset = torch.gather(
            #     init_pos, dim=1, index=outcell_index.unsqueeze(-1)
            # )
            # init_expand_pos = init_expand_pos_no_offset + offset
            # init_expand_pos = init_expand_pos.masked_fill(
            #     pbc_expand_batched["expand_mask"].unsqueeze(-1),
            #     0.0,
            # )

            # pbc_expand_batched["init_expand_pos"] = init_expand_pos

            # # self.check_conflict(pos, atoms, pbc_expand_batched)
            # print(f"local attention weight {local_attention_weight.numel()} zero:{torch.sum(local_attention_weight==0)}")
            # # print(torch.sum(local_attention_weight==0,dim = 1)==(local_attention_weight.shape[1]))
            # print("N1+N2, ",local_attention_weight.shape[2],torch.sum(
            #     torch.sum(local_attention_weight==0,dim = 1)==local_attention_weight.shape[1])/(local_attention_weight.shape[0]*1.0))
            # pbc_expand_batched["local_attention_weight"] = None
            return pbc_expand_batched

    # B = 20
    # N = 15
    # topK = 20
    # max_radius = 5.1

    # candidate_cells = [[i, j, k]
    #             for i in range(-5, 5 + 1)
    #             for j in range(-5, 5 + 1)
    #             for k in range(-5, 5 + 1)]
    # candidate_cells = torch.tensor(candidate_cells)
    # cell = torch.randn(B,3,3)
    # node_pos = torch.randn(B,N,3)
    # padding_mask = torch.randn(B,N)>0
    # node_mask = ~padding_mask
    # node_pos[padding_mask] = 9999


#     def expand_includeself_clean(
#     pos,
#     padding_mask,
#     is_pbc,
#     cell,
#     max_radius,
#     max_neighbors,
#     candidate_cells,
#     use_grad=False
# ):
#     with torch.set_grad_enabled(use_grad):
#         # pos = torch.randn(B,N,3)
#         pos = pos.float()
#         cell = cell.float()
#         B, N = pos.size()[:2]
#         max_offsets , _ = torch.max(
#                 torch.ceil(torch.linalg.norm(cell,dim = 2)),dim = 0)
#         max_offsets = is_pbc[0]*max_offsets
#         legal_cell = candidate_cells[(candidate_cells.abs()<=max_offsets).all(dim=-1)]
#         legal_cell = legal_cell.unsqueeze(0).repeat(B, 1, 1).to(dtype=cell.dtype)
#         num_expanded_cell = legal_cell.shape[1]


#         pos[padding_mask] = 9999
#         offset = torch.bmm(legal_cell, cell)  # B x num_expand_cell x 3
#         expand_pos = (pos.unsqueeze(1) + offset.unsqueeze(2)).view(B,-1,3)

#         expand_dist = torch.norm(
#             pos.unsqueeze(2) - expand_pos.unsqueeze(1), p=2, dim=-1
#         )  # B x T x (num_expand_cell x T)

#         max_neighbors = min(max_neighbors,expand_pos.shape[1])
#         values, _ = torch.topk(expand_dist, max_neighbors,dim = -1,largest=False)

#         expand_legal_mask = (expand_dist<=(values[:,:,max_neighbors-1].unsqueeze(dim=-1))) & \
#                         (expand_dist < max_radius)       & \
#                         (expand_dist > 1e-5)

#         expand_legal_mask = torch.sum(expand_legal_mask, dim=1) > 0


#         expand_len = torch.sum(expand_legal_mask, dim=-1)
#         max_expand_len = torch.max(expand_len)
#         expand_mask = (torch.arange(0, max_expand_len,device=pos.device)[None].repeat(B,1))>=(expand_len.unsqueeze(dim = -1))


#         outcell_index = torch.zeros(
#             [B, max_expand_len], dtype=torch.long, device=pos.device
#         )
#         expand_pos_compressed = torch.zeros(
#             [B, max_expand_len, 3], dtype=pos.dtype, device=pos.device
#         )
#         outcell_all_index = torch.arange(
#             N, dtype=torch.long, device=pos.device
#         ).repeat(num_expanded_cell)
#         for i in range(B):
#             outcell_index[i, : expand_len[i]] = outcell_all_index[expand_legal_mask[i]]
#             expand_pos_compressed[i, : expand_len[i], :] = expand_pos[
#                 i, expand_legal_mask[i], :
#             ]

#         pbc_expand_batched = {
#                 "expand_pos": expand_pos_compressed,
#                 "outcell_index": outcell_index,
#                 "expand_mask": expand_mask,
#             }
#         print(pos.shape,expand_pos_compressed.shape)
#         return pbc_expand_batched


def get_normalization_layer(
    norm_type, lmax, num_channels, eps=1e-5, affine=True, normalization="component"
):
    assert norm_type in [
        "layer_norm",
        "layer_norm_sh",
        "rms_norm_sh",
        "rms_norm_sh_BL",
        "identity",
    ]
    if norm_type == "layer_norm":
        norm_class = EquivariantLayerNormArray
    elif norm_type == "layer_norm_sh" or norm_type == "layer_norm_sh_BL":
        norm_class = EquivariantLayerNormArraySphericalHarmonics
    elif norm_type == "rms_norm_sh" or norm_type == "rms_norm_sh_BL":
        #     norm_class = EquivariantRMSNormArraySphericalHarmonicsV2
        # elif norm_type == "rms_norm_sh_BL":
        norm_class = EquivariantRMSNormArraySphericalHarmonicsV2_BL
    elif norm_type == "identity":
        norm_class = nn.Identity
    else:
        raise ValueError
    return norm_class(lmax, num_channels, eps, affine, normalization)


def get_l_to_all_m_expand_index(lmax):
    expand_index = torch.zeros([(lmax + 1) ** 2]).long()
    for l in range(lmax + 1):
        start_idx = l**2
        length = 2 * l + 1
        expand_index[start_idx : (start_idx + length)] = l
    return expand_index


class EquivariantLayerNormArray(nn.Module):
    def __init__(
        self, lmax, num_channels, eps=1e-5, affine=True, normalization="component"
    ):
        super().__init__()

        self.lmax = lmax
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if affine:
            self.affine_weight = nn.Parameter(torch.ones(lmax + 1, num_channels))
            self.affine_bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)

        assert normalization in ["norm", "component"]
        self.normalization = normalization

    def __repr__(self):
        return f"{self.__class__.__name__}(lmax={self.lmax}, num_channels={self.num_channels}, eps={self.eps})"

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, node_input):
        """
        Assume input is of shape [N, sphere_basis, C]
        """

        out = []

        for l in range(self.lmax + 1):
            start_idx = l**2
            length = 2 * l + 1

            feature = node_input.narrow(1, start_idx, length)

            # For scalars, first compute and subtract the mean
            if l == 0:
                feature_mean = torch.mean(feature, dim=2, keepdim=True)
                feature = feature - feature_mean

            # Then compute the rescaling factor (norm of each feature vector)
            # Rescaling of the norms themselves based on the option "normalization"
            if self.normalization == "norm":
                feature_norm = feature.pow(2).sum(dim=1, keepdim=True)  # [N, 1, C]
            elif self.normalization == "component":
                feature_norm = feature.pow(2).mean(dim=1, keepdim=True)  # [N, 1, C]

            feature_norm = torch.mean(feature_norm, dim=2, keepdim=True)  # [N, 1, 1]
            feature_norm = (feature_norm + self.eps).pow(-0.5)

            if self.affine:
                weight = self.affine_weight.narrow(0, l, 1)  # [1, C]
                weight = weight.view(1, 1, -1)  # [1, 1, C]
                feature_norm = feature_norm * weight  # [N, 1, C]

            feature = feature * feature_norm

            if self.affine and l == 0:
                bias = self.affine_bias
                bias = bias.view(1, 1, -1)
                feature = feature + bias

            out.append(feature)

        out = torch.cat(out, dim=1)

        return out


class EquivariantLayerNormArraySphericalHarmonics(nn.Module):
    """
    1. Normalize over L = 0.
    2. Normalize across all m components from degrees L > 0.
    3. Do not normalize separately for different L (L > 0).
    """

    def __init__(
        self,
        lmax,
        num_channels,
        eps=1e-5,
        affine=True,
        normalization="component",
        std_balance_degrees=True,
    ):
        super().__init__()

        self.lmax = lmax
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.std_balance_degrees = std_balance_degrees

        # for L = 0
        self.norm_l0 = torch.nn.LayerNorm(
            self.num_channels, eps=self.eps, elementwise_affine=self.affine
        )

        # for L > 0
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.lmax, self.num_channels))
        else:
            self.register_parameter("affine_weight", None)

        assert normalization in ["norm", "component"]
        self.normalization = normalization

        if self.std_balance_degrees:
            balance_degree_weight = torch.zeros((self.lmax + 1) ** 2 - 1, 1)
            for l in range(1, self.lmax + 1):
                start_idx = l**2 - 1
                length = 2 * l + 1
                balance_degree_weight[start_idx : (start_idx + length), :] = (
                    1.0 / length
                )
            balance_degree_weight = balance_degree_weight / self.lmax
            self.register_buffer("balance_degree_weight", balance_degree_weight)
        else:
            self.balance_degree_weight = None

    def __repr__(self):
        return f"{self.__class__.__name__}(lmax={self.lmax}, num_channels={self.num_channels}, eps={self.eps}, std_balance_degrees={self.std_balance_degrees})"

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, node_input):
        """
        Assume input is of shape [N, sphere_basis, C]
        """
        out_shape = node_input.shape[:-2]
        node_input = node_input.reshape(
            out_shape.numel(), (self.lmax + 1) ** 2, self.num_channels
        )

        out = []

        # for L = 0
        feature = node_input.narrow(1, 0, 1)
        feature = self.norm_l0(feature)
        out.append(feature)

        # for L > 0
        if self.lmax > 0:
            num_m_components = (self.lmax + 1) ** 2
            feature = node_input.narrow(1, 1, num_m_components - 1)

            # Then compute the rescaling factor (norm of each feature vector)
            # Rescaling of the norms themselves based on the option "normalization"
            if self.normalization == "norm":
                feature_norm = feature.pow(2).sum(dim=1, keepdim=True)  # [N, 1, C]
            elif self.normalization == "component":
                if self.std_balance_degrees:
                    feature_norm = feature.pow(
                        2
                    )  # [N, (L_max + 1)**2 - 1, C], without L = 0
                    feature_norm = torch.einsum(
                        "nic, ia -> nac", feature_norm, self.balance_degree_weight
                    )  # [N, 1, C]
                else:
                    feature_norm = feature.pow(2).mean(dim=1, keepdim=True)  # [N, 1, C]

            feature_norm = torch.mean(feature_norm, dim=2, keepdim=True)  # [N, 1, 1]
            feature_norm = (feature_norm + self.eps).pow(-0.5)

            for l in range(1, self.lmax + 1):
                start_idx = l**2
                length = 2 * l + 1
                feature = node_input.narrow(1, start_idx, length)  # [N, (2L + 1), C]
                if self.affine:
                    weight = self.affine_weight.narrow(0, (l - 1), 1)  # [1, C]
                    weight = weight.view(1, 1, -1)  # [1, 1, C]
                    feature_scale = feature_norm * weight  # [N, 1, C]
                else:
                    feature_scale = feature_norm
                feature = feature * feature_scale
                out.append(feature)

        out = torch.cat(out, dim=1)
        return out.reshape(out_shape + ((self.lmax + 1) ** 2, self.num_channels))


class EquivariantRMSNormArraySphericalHarmonics(nn.Module):
    """
    1. Normalize across all m components from degrees L >= 0.
    """

    def __init__(
        self, lmax, num_channels, eps=1e-5, affine=True, normalization="component"
    ):
        super().__init__()

        self.lmax = lmax
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        # for L >= 0
        if self.affine:
            self.affine_weight = nn.Parameter(
                torch.ones((self.lmax + 1), self.num_channels)
            )
        else:
            self.register_parameter("affine_weight", None)

        assert normalization in ["norm", "component"]
        self.normalization = normalization

    def __repr__(self):
        return f"{self.__class__.__name__}(lmax={self.lmax}, num_channels={self.num_channels}, eps={self.eps})"

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, node_input):
        """
        Assume input is of shape [N, sphere_basis, C]
        """

        out = []

        # for L >= 0
        feature = node_input
        if self.normalization == "norm":
            feature_norm = feature.pow(2).sum(dim=1, keepdim=True)  # [N, 1, C]
        elif self.normalization == "component":
            feature_norm = feature.pow(2).mean(dim=1, keepdim=True)  # [N, 1, C]

        feature_norm = torch.mean(feature_norm, dim=2, keepdim=True)  # [N, 1, 1]
        feature_norm = (feature_norm + self.eps).pow(-0.5)

        for l in range(0, self.lmax + 1):
            start_idx = l**2
            length = 2 * l + 1
            feature = node_input.narrow(1, start_idx, length)  # [N, (2L + 1), C]
            if self.affine:
                weight = self.affine_weight.narrow(0, l, 1)  # [1, C]
                weight = weight.view(1, 1, -1)  # [1, 1, C]
                feature_scale = feature_norm * weight  # [N, 1, C]
            else:
                feature_scale = feature_norm
            feature = feature * feature_scale
            out.append(feature)

        out = torch.cat(out, dim=1)
        return out


class EquivariantRMSNormArraySphericalHarmonicsV2(nn.Module):
    """
    1. Normalize across all m components from degrees L >= 0.
    2. Expand weights and multiply with normalized feature to prevent slicing and concatenation.
    """

    def __init__(
        self,
        lmax,
        num_channels,
        eps=1e-5,
        affine=True,
        normalization="component",
        centering=True,
        std_balance_degrees=True,
    ):
        super().__init__()

        self.lmax = lmax
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.centering = centering
        self.std_balance_degrees = std_balance_degrees

        # for L >= 0
        if self.affine:
            self.affine_weight = nn.Parameter(
                torch.ones((self.lmax + 1), self.num_channels)
            )
            if self.centering:
                self.affine_bias = nn.Parameter(torch.zeros(self.num_channels))
            else:
                self.register_parameter("affine_bias", None)
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)

        assert normalization in ["norm", "component"]
        self.normalization = normalization

        expand_index = get_l_to_all_m_expand_index(self.lmax)
        self.register_buffer("expand_index", expand_index)

        if self.std_balance_degrees:
            balance_degree_weight = torch.zeros((self.lmax + 1) ** 2, 1)
            for l in range(self.lmax + 1):
                start_idx = l**2
                length = 2 * l + 1
                balance_degree_weight[start_idx : (start_idx + length), :] = (
                    1.0 / length
                )
            balance_degree_weight = balance_degree_weight / (self.lmax + 1)
            self.register_buffer("balance_degree_weight", balance_degree_weight)
        else:
            self.balance_degree_weight = None

    def __repr__(self):
        return f"{self.__class__.__name__}(lmax={self.lmax}, num_channels={self.num_channels}, eps={self.eps}, centering={self.centering}, std_balance_degrees={self.std_balance_degrees})"

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, node_input, batch=None):
        """
        Assume input is of shape [N, sphere_basis, C]
        """

        feature = node_input

        if self.centering:
            feature_l0 = feature.narrow(1, 0, 1)
            feature_l0_mean = feature_l0.mean(dim=2, keepdim=True)  # [N, 1, 1]
            feature_l0 = feature_l0 - feature_l0_mean
            feature = torch.cat(
                (feature_l0, feature.narrow(1, 1, feature.shape[1] - 1)), dim=1
            )

        # for L >= 0
        if self.normalization == "norm":
            assert not self.std_balance_degrees
            feature_norm = feature.pow(2).sum(dim=1, keepdim=True)  # [N, 1, C]
        elif self.normalization == "component":
            if self.std_balance_degrees:
                feature_norm = feature.pow(2)  # [N, (L_max + 1)**2, C]
                feature_norm = torch.einsum(
                    "nic, ia -> nac", feature_norm, self.balance_degree_weight
                )  # [N, 1, C]
            else:
                feature_norm = feature.pow(2).mean(dim=1, keepdim=True)  # [N, 1, C]

        feature_norm = torch.mean(feature_norm, dim=2, keepdim=True)  # [N, 1, 1]
        feature_norm = (feature_norm + self.eps).pow(-0.5)

        if self.affine:
            weight = self.affine_weight.view(
                1, (self.lmax + 1), self.num_channels
            )  # [1, L_max + 1, C]
            weight = torch.index_select(
                weight, dim=1, index=self.expand_index
            )  # [1, (L_max + 1)**2, C]
            feature_norm = feature_norm * weight  # [N, (L_max + 1)**2, C]

        out = feature * feature_norm

        if self.affine and self.centering:
            out[:, 0:1, :] = out.narrow(1, 0, 1) + self.affine_bias.view(
                1, 1, self.num_channels
            )

        return out


class EquivariantRMSNormArraySphericalHarmonicsV2_BL(nn.Module):
    """
    1. Normalize across all m components from degrees L >= 0.
    2. Expand weights and multiply with normalized feature to prevent slicing and concatenation.
    """

    def __init__(
        self,
        lmax,
        num_channels,
        eps=1e-5,
        affine=True,
        normalization="component",
        centering=True,
        std_balance_degrees=True,
    ):
        super().__init__()

        self.lmax = lmax
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.centering = centering
        self.std_balance_degrees = std_balance_degrees

        # for L >= 0
        if self.affine:
            self.affine_weight = nn.Parameter(
                torch.ones((self.lmax + 1), self.num_channels)
            )
            if self.centering:
                self.affine_bias = nn.Parameter(torch.zeros(self.num_channels))
            else:
                self.register_parameter("affine_bias", None)
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)

        assert normalization in ["norm", "component"]
        self.normalization = normalization

        expand_index = get_l_to_all_m_expand_index(self.lmax)
        self.register_buffer("expand_index", expand_index)

        if self.std_balance_degrees:
            balance_degree_weight = torch.zeros((self.lmax + 1) ** 2, 1)
            for l in range(self.lmax + 1):
                start_idx = l**2
                length = 2 * l + 1
                balance_degree_weight[start_idx : (start_idx + length), :] = (
                    1.0 / length
                )
            balance_degree_weight = balance_degree_weight / (self.lmax + 1)
            self.register_buffer("balance_degree_weight", balance_degree_weight)
        else:
            self.balance_degree_weight = None

    def __repr__(self):
        return f"{self.__class__.__name__}(lmax={self.lmax}, num_channels={self.num_channels}, eps={self.eps}, centering={self.centering}, std_balance_degrees={self.std_balance_degrees})"

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, node_input, batch=None):
        """
        Assume input is of shape [N, sphere_basis, C]
        """
        out_shape = node_input.shape[:-2]
        feature = node_input.reshape(
            out_shape.numel(), (self.lmax + 1) ** 2, self.num_channels
        )

        if self.centering:
            feature_l0 = feature.narrow(1, 0, 1)
            feature_l0_mean = feature_l0.mean(dim=2, keepdim=True)  # [N, 1, 1]
            feature_l0 = feature_l0 - feature_l0_mean
            feature = torch.cat(
                (feature_l0, feature.narrow(1, 1, feature.shape[1] - 1)), dim=1
            )

        # for L >= 0
        if self.normalization == "norm":
            assert not self.std_balance_degrees
            feature_norm = feature.pow(2).sum(dim=1, keepdim=True)  # [N, 1, C]
        elif self.normalization == "component":
            if self.std_balance_degrees:
                feature_norm = feature.pow(2)  # [N, (L_max + 1)**2, C]
                feature_norm = torch.einsum(
                    "nic, ia -> nac", feature_norm, self.balance_degree_weight
                )  # [N, 1, C]
            else:
                feature_norm = feature.pow(2).mean(dim=1, keepdim=True)  # [N, 1, C]

        feature_norm = torch.mean(feature_norm, dim=2, keepdim=True)  # [N, 1, 1]
        feature_norm = (feature_norm + self.eps).pow(-0.5)

        if self.affine:
            weight = self.affine_weight.view(
                1, (self.lmax + 1), self.num_channels
            )  # [1, L_max + 1, C]
            weight = torch.index_select(
                weight, dim=1, index=self.expand_index
            )  # [1, (L_max + 1)**2, C]
            feature_norm = feature_norm * weight  # [N, (L_max + 1)**2, C]

        out = feature * feature_norm

        if self.affine and self.centering:
            out[:, 0:1, :] = out.narrow(1, 0, 1) + self.affine_bias.view(
                1, 1, self.num_channels
            )

        return out.reshape(out_shape + ((self.lmax + 1) ** 2, self.num_channels))


class EquivariantDegreeLayerScale(nn.Module):
    """
    1. Similar to Layer Scale used in CaiT (Going Deeper With Image Transformers (ICCV'21)), we scale the output of both attention and FFN.
    2. For degree L > 0, we scale down the square root of 2 * L, which is to emulate halving the number of channels when using higher L.
    """

    def __init__(self, lmax, num_channels, scale_factor=2.0):
        super().__init__()

        self.lmax = lmax
        self.num_channels = num_channels
        self.scale_factor = scale_factor

        self.affine_weight = nn.Parameter(
            torch.ones(1, (self.lmax + 1), self.num_channels)
        )
        for l in range(1, self.lmax + 1):
            self.affine_weight.data[0, l, :].mul_(
                1.0 / math.sqrt(self.scale_factor * l)
            )
        expand_index = get_l_to_all_m_expand_index(self.lmax)
        self.register_buffer("expand_index", expand_index)

    def __repr__(self):
        return f"{self.__class__.__name__}(lmax={self.lmax}, num_channels={self.num_channels}, scale_factor={self.scale_factor})"

    def forward(self, node_input):
        weight = torch.index_select(
            self.affine_weight, dim=1, index=self.expand_index
        )  # [1, (L_max + 1)**2, C]
        node_input = node_input * weight  # [N, (L_max + 1)**2, C]
        return node_input


@torch.jit.script
def gaussian(x, mean, std):
    pi = torch.pi
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


from fairchem.core.common.utils import (
    compute_neighbors,
    get_max_neighbors_mask,
    get_pbc_distances,
)


class RadialFunction(nn.Module):
    """
    Contruct a radial function (linear layers + layer normalization + SiLU) given a list of channels
    """

    def __init__(self, channels_list, use_layer_norm=True):
        super().__init__()
        modules = []
        input_channels = channels_list[0]
        for i in range(len(channels_list)):
            if i == 0:
                continue

            modules.append(nn.Linear(input_channels, channels_list[i], bias=True))
            input_channels = channels_list[i]

            if i == len(channels_list) - 1:
                break
            if use_layer_norm:
                modules.append(nn.LayerNorm(channels_list[i]))
            modules.append(torch.nn.SiLU())

        self.net = nn.Sequential(*modules)

    def forward(self, inputs):
        return self.net(inputs)


# From Graphormer
class GaussianRadialBasisLayer(torch.nn.Module):
    def __init__(self, num_basis, cutoff):
        super().__init__()
        self.num_basis = num_basis
        self.cutoff = cutoff + 0.0
        self.mean = torch.nn.Parameter(torch.zeros(1, self.num_basis))
        self.std = torch.nn.Parameter(torch.zeros(1, self.num_basis))
        self.weight = torch.nn.Parameter(torch.ones(1, 1))
        self.bias = torch.nn.Parameter(torch.zeros(1, 1))

        self.std_init_max = 1.0
        self.std_init_min = 1.0 / self.num_basis
        self.mean_init_max = 1.0
        self.mean_init_min = 0
        torch.nn.init.uniform_(self.mean, self.mean_init_min, self.mean_init_max)
        torch.nn.init.uniform_(self.std, self.std_init_min, self.std_init_max)
        torch.nn.init.constant_(self.weight, 1)
        torch.nn.init.constant_(self.bias, 0)

    def forward(self, dist, node_atom=None, edge_src=None, edge_dst=None):
        x = dist / self.cutoff
        x = x.unsqueeze(-1)
        x = self.weight * x + self.bias
        x = x.expand(-1, self.num_basis)
        mean = self.mean
        std = self.std.abs() + 1e-5
        x = gaussian(x, mean, std)
        return x

    def extra_repr(self):
        return "mean_init_max={}, mean_init_min={}, std_init_max={}, std_init_min={}".format(
            self.mean_init_max, self.mean_init_min, self.std_init_max, self.std_init_min
        )


class GaussianSmearing(torch.nn.Module):
    def __init__(
        self,
        num_basis,
        cutoff: float = 5.0,
        basis_width_scalar: float = 2.0,
    ) -> None:
        super().__init__()
        offset = torch.linspace(0, cutoff, num_basis)
        self.coeff = -0.5 / (basis_width_scalar * (offset[1] - offset[0])).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist) -> torch.Tensor:
        shape = dist.shape
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        dist = torch.exp(self.coeff * torch.pow(dist, 2))
        return dist.reshape(*shape, -1)


# gaussian layer with edge type (i,j)
class GaussianLayer_Edgetype(nn.Module):
    def __init__(self, K=128, edge_types=512 * 3):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1, padding_idx=0)
        self.bias = nn.Embedding(edge_types, 1, padding_idx=0)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_types):
        mul = self.mul(edge_types).sum(dim=-2)
        bias = self.bias(edge_types).sum(dim=-2)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-2
        return gaussian(x.float(), mean, std).type_as(self.means.weight)


# class CosineCutoff(nn.Module):
#     def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0):
#         super(CosineCutoff, self).__init__()
#         self.cutoff_lower = cutoff_lower
#         self.cutoff_upper = cutoff_upper

#     def forward(self, distances):
#         if self.cutoff_lower > 0:
#             cutoffs = 0.5 * (
#                 torch.cos(
#                     math.pi
#                     * (
#                         2
#                         * (distances - self.cutoff_lower)
#                         / (self.cutoff_upper - self.cutoff_lower)
#                         + 1.0
#                     )
#                 )
#                 + 1.0
#             )
#             # remove contributions below the cutoff radius
#             cutoffs = cutoffs * (distances < self.cutoff_upper).float()
#             cutoffs = cutoffs * (distances > self.cutoff_lower).float()
#             return cutoffs
#         else:
#             cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff_upper) + 1.0)
#             # remove contributions beyond the cutoff radius
#             cutoffs = cutoffs * (distances < self.cutoff_upper).float()
#             return cutoffs


# in farchem, the max_rep = [rep_a1.max(), rep_a2.max(), rep_a3.max()] is very big for some special case in oc20 dataset
# thus we use the max_rep clip to avoid this issue
def radius_graph_pbc(
    data,
    radius,
    max_num_neighbors_threshold,
    enforce_max_neighbors_strictly: bool = False,
    rep_clip: int = 5,
    pbc=None,
):
    if pbc is None:
        pbc = [True, True, True]
    device = data.pos.device
    batch_size = len(data.natoms)

    if hasattr(data, "pbc"):
        data.pbc = torch.atleast_2d(data.pbc)
        for i in range(3):
            if not torch.any(data.pbc[:, i]).item():
                pbc[i] = False
            elif torch.all(data.pbc[:, i]).item():
                pbc[i] = True
            else:
                raise RuntimeError(
                    "Different structures in the batch have different PBC configurations. This is not currently supported."
                )

    # position of the atoms
    atom_pos = data.pos

    # Before computing the pairwise distances between atoms, first create a list of atom indices to compare for the entire batch
    num_atoms_per_image = data.natoms
    num_atoms_per_image_sqr = (num_atoms_per_image**2).long()

    # index offset between images
    index_offset = torch.cumsum(num_atoms_per_image, dim=0) - num_atoms_per_image

    index_offset_expand = torch.repeat_interleave(index_offset, num_atoms_per_image_sqr)
    num_atoms_per_image_expand = torch.repeat_interleave(
        num_atoms_per_image, num_atoms_per_image_sqr
    )

    # Compute a tensor containing sequences of numbers that range from 0 to num_atoms_per_image_sqr for each image
    # that is used to compute indices for the pairs of atoms. This is a very convoluted way to implement
    # the following (but 10x faster since it removes the for loop)
    # for batch_idx in range(batch_size):
    #    batch_count = torch.cat([batch_count, torch.arange(num_atoms_per_image_sqr[batch_idx], device=device)], dim=0)
    num_atom_pairs = torch.sum(num_atoms_per_image_sqr)
    index_sqr_offset = (
        torch.cumsum(num_atoms_per_image_sqr, dim=0) - num_atoms_per_image_sqr
    )
    index_sqr_offset = torch.repeat_interleave(
        index_sqr_offset, num_atoms_per_image_sqr
    )
    atom_count_sqr = torch.arange(num_atom_pairs, device=device) - index_sqr_offset

    # Compute the indices for the pairs of atoms (using division and mod)
    # If the systems get too large this apporach could run into numerical precision issues
    index1 = (
        torch.div(atom_count_sqr, num_atoms_per_image_expand, rounding_mode="floor")
    ) + index_offset_expand
    index2 = (atom_count_sqr % num_atoms_per_image_expand) + index_offset_expand
    # Get the positions for each atom
    pos1 = torch.index_select(atom_pos, 0, index1)
    pos2 = torch.index_select(atom_pos, 0, index2)

    # # Calculate required number of unit cells in each direction.
    # Smallest distance between planes separated by a1 is
    # 1 / ||(a2 x a3) / V||_2, since a2 x a3 is the area of the plane.
    # Note that the unit cell volume V = a1 * (a2 x a3) and that
    # (a2 x a3) / V is also the reciprocal primitive vector
    # (crystallographer's definition).

    cross_a2a3 = torch.cross(data.cell[:, 1], data.cell[:, 2], dim=-1)
    cell_vol = torch.sum(data.cell[:, 0] * cross_a2a3, dim=-1, keepdim=True)

    if pbc[0]:
        inv_min_dist_a1 = torch.norm(cross_a2a3 / cell_vol, p=2, dim=-1)
        rep_a1 = torch.ceil(radius * inv_min_dist_a1)
    else:
        rep_a1 = data.cell.new_zeros(1)

    if pbc[1]:
        cross_a3a1 = torch.cross(data.cell[:, 2], data.cell[:, 0], dim=-1)
        inv_min_dist_a2 = torch.norm(cross_a3a1 / cell_vol, p=2, dim=-1)
        rep_a2 = torch.ceil(radius * inv_min_dist_a2)
    else:
        rep_a2 = data.cell.new_zeros(1)

    if pbc[2]:
        cross_a1a2 = torch.cross(data.cell[:, 0], data.cell[:, 1], dim=-1)
        inv_min_dist_a3 = torch.norm(cross_a1a2 / cell_vol, p=2, dim=-1)
        rep_a3 = torch.ceil(radius * inv_min_dist_a3)
    else:
        rep_a3 = data.cell.new_zeros(1)

    # # Take the max over all images for uniformity. This is essentially padding.
    # # Note that this can significantly increase the number of computed distances
    # # if the required repetitions are very different between images
    # # (which they usually are). Changing this to sparse (scatter) operations
    # # might be worth the effort if this function becomes a bottleneck.
    max_rep = [
        rep_a1.max().clip(max=rep_clip),
        rep_a2.max().clip(max=rep_clip),
        rep_a3.max().clip(max=rep_clip),
    ]
    # max_rep = [rep_clip,rep_clip,rep_clip]
    # print(max_rep)
    # Tensor of unit cells
    cells_per_dim = [
        torch.arange(-rep, rep + 1, device=device, dtype=data.cell.dtype)
        for rep in max_rep
    ]
    unit_cell = torch.cartesian_prod(*cells_per_dim)
    num_cells = len(unit_cell)
    unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(len(index2), 1, 1)
    unit_cell = torch.transpose(unit_cell, 0, 1)
    unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(batch_size, -1, -1)

    # Compute the x, y, z positional offsets for each cell in each image
    data_cell = torch.transpose(data.cell, 1, 2)
    pbc_offsets = torch.bmm(data_cell, unit_cell_batch)
    pbc_offsets_per_atom = torch.repeat_interleave(
        pbc_offsets, num_atoms_per_image_sqr, dim=0
    )

    # Expand the positions and indices for the 9 cells
    pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells)
    pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
    index1 = index1.view(-1, 1).repeat(1, num_cells).view(-1)
    index2 = index2.view(-1, 1).repeat(1, num_cells).view(-1)
    # Add the PBC offsets for the second atom
    pos2 = pos2 + pbc_offsets_per_atom

    # Compute the squared distance between atoms
    atom_distance_sqr = torch.sum((pos1 - pos2) ** 2, dim=1)
    atom_distance_sqr = atom_distance_sqr.view(-1)

    # Remove pairs that are too far apart
    mask_within_radius = torch.le(atom_distance_sqr, radius * radius)
    # Remove pairs with the same atoms (distance = 0.0)
    mask_not_same = torch.gt(atom_distance_sqr, 0.0001)
    mask = torch.logical_and(mask_within_radius, mask_not_same)
    index1 = torch.masked_select(index1, mask)
    index2 = torch.masked_select(index2, mask)
    unit_cell = torch.masked_select(
        unit_cell_per_atom.view(-1, 3), mask.view(-1, 1).expand(-1, 3)
    )
    unit_cell = unit_cell.view(-1, 3)
    atom_distance_sqr = torch.masked_select(atom_distance_sqr, mask)

    mask_num_neighbors, num_neighbors_image = get_max_neighbors_mask(
        natoms=data.natoms,
        index=index1,
        atom_distance=atom_distance_sqr,
        max_num_neighbors_threshold=max_num_neighbors_threshold,
        enforce_max_strictly=enforce_max_neighbors_strictly,
    )

    if not torch.all(mask_num_neighbors):
        # Mask out the atoms to ensure each atom has at most max_num_neighbors_threshold neighbors
        index1 = torch.masked_select(index1, mask_num_neighbors)
        index2 = torch.masked_select(index2, mask_num_neighbors)
        unit_cell = torch.masked_select(
            unit_cell.view(-1, 3), mask_num_neighbors.view(-1, 1).expand(-1, 3)
        )
        unit_cell = unit_cell.view(-1, 3)

    edge_index = torch.stack((index2, index1))

    return edge_index, unit_cell, num_neighbors_image


def generate_graph(
    data,
    cutoff,
    max_neighbors=None,
    use_pbc=None,
    otf_graph=None,
    enforce_max_neighbors_strictly=True,
):
    if not otf_graph:
        try:
            edge_index = data.edge_index
            if use_pbc:
                cell_offsets = data.cell_offsets
                neighbors = data.neighbors

        except AttributeError:
            logging.warning(
                "Turning otf_graph=True as required attributes not present in data object"
            )
            otf_graph = True

    if use_pbc:
        if otf_graph:
            edge_index, cell_offsets, neighbors = radius_graph_pbc(
                data,
                cutoff,
                max_neighbors,
                enforce_max_neighbors_strictly,
            )

        out = get_pbc_distances(
            data.pos,
            edge_index,
            data.cell,
            cell_offsets,
            neighbors,
            return_offsets=True,
            return_distance_vec=True,
        )

        edge_index = out["edge_index"]
        edge_dist = out["distances"]
        cell_offset_distances = out["offsets"]
        distance_vec = out["distance_vec"]
    else:
        if otf_graph:
            edge_index = radius_graph(
                data.pos,
                r=cutoff,
                batch=data.batch,
                max_num_neighbors=max_neighbors,
            )

        j, i = edge_index
        distance_vec = data.pos[j] - data.pos[i]

        edge_dist = distance_vec.norm(dim=-1)
        cell_offsets = torch.zeros(edge_index.shape[1], 3, device=data.pos.device)
        cell_offset_distances = torch.zeros_like(cell_offsets, device=data.pos.device)
        neighbors = compute_neighbors(data, edge_index)

    return (
        edge_index,
        edge_dist,
        distance_vec,
        cell_offsets,
        cell_offset_distances,
        neighbors,
    )


def construct_o3irrps(dim, order):
    string = []
    for l in range(order + 1):
        string.append(f"{dim}x{l}e" if l % 2 == 0 else f"{dim}x{l}o")
    return "+".join(string)


def to_torchgeometric_Data(data: dict):
    torchgeometric_data = Data()
    for key in data.keys():
        torchgeometric_data[key] = data[key]
    return torchgeometric_data


def construct_o3irrps_base(dim, order):
    string = []
    for l in range(order + 1):
        string.append(f"{dim}x{l}e")
    return "+".join(string)


def polynomial(dist: torch.Tensor, cutoff: float) -> torch.Tensor:
    """
    Polynomial cutoff function,ref: https://arxiv.org/abs/2204.13639
    Args:
        dist (tf.Tensor): distance tensor
        cutoff (float): cutoff distance
    Returns: polynomial cutoff functions
    """
    ratio = torch.div(dist, cutoff)
    result = (
        1
        - 6 * torch.pow(ratio, 5)
        + 15 * torch.pow(ratio, 4)
        - 10 * torch.pow(ratio, 3)
    )
    return torch.clamp(result, min=0.0)


def SmoothSoftmax(input, edge_dis, max_dist=5.0, dim=2, eps=1e-5, batched_data=None):
    local_attn_weight = polynomial(edge_dis, max_dist)
    input = input.to(torch.float64)
    local_attn_weight = local_attn_weight.to(input.dtype)

    max_value = input.max(dim=dim, keepdim=True).values
    input = input - max_value
    e_ij = torch.exp(input) * local_attn_weight.unsqueeze(-1)
    # e_ij = input * local_attn_weight.unsqueeze(-1)

    if torch.isnan(e_ij).any() or torch.isinf(e_ij).any():
        print("e_ij has nan or inf")
        print(e_ij)
    # Compute softmax along the last dimension
    softmax = e_ij / (torch.sum(e_ij, dim=dim, keepdim=True) + eps)
    # softmax = torch.nn.functional.softmax(e_ij, dim=dim)

    softmax = softmax.to(torch.float32)

    return softmax


# def SmoothSoftmax(input, mask, max_dist=5.0, eps: float = 1e-16):
#     # Invert distances to ensure smaller distances get higher weights
#     # No need to mask out the 1000 values, they will naturally get near-zero weights
#     mask = mask.squeeze(-1)
#     input = input.masked_fill(mask, 1000)
#     inverted_input = max_dist - input

#     # Compute the maximum value for numerical stability
#     max_value = inverted_input.max(dim=-1, keepdim=True).values

#     # Shift the input by subtracting the maximum value to avoid overflow during exponentiation
#     shifted_input = inverted_input - max_value

#     # Compute e_ij (exponential of the shifted input)
#     e_ij = torch.exp(shifted_input)

#     # Check for NaN or infinite values
#     if torch.isnan(e_ij).any() or torch.isinf(e_ij).any():
#         print("e_ij has nan or inf")
#         print(e_ij)

#     # Compute Softmax
#     coeff = (mask.shape[-1] - mask.sum(-1)).unsqueeze(-1)
#     softmax = e_ij / (torch.sum(e_ij, dim=-1, keepdim=True) + eps) * coeff
#     softmax = softmax.masked_fill(mask, 1e-6)

#     return softmax


class SO3_Linear_e2former(torch.nn.Module):
    def __init__(self, in_features, out_features, lmax, bias=True):
        """
        1. Use `torch.einsum` to prevent slicing and concatenation
        2. Need to specify some behaviors in `no_weight_decay` and weight initialization.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lmax = lmax

        self.weight = torch.nn.Parameter(
            torch.randn((self.lmax + 1), out_features, in_features)
        )
        bound = 1 / math.sqrt(self.in_features)
        torch.nn.init.uniform_(self.weight, -bound, bound)
        self.bias = torch.nn.Parameter(torch.zeros(out_features))

        expand_index = torch.zeros([(lmax + 1) ** 2]).long()
        for l in range(lmax + 1):
            start_idx = l**2
            length = 2 * l + 1
            expand_index[start_idx : (start_idx + length)] = l
        self.register_buffer("expand_index", expand_index)

    def forward(self, input_embedding):
        output_shape = input_embedding.shape[:-2]
        l_sum, hidden = input_embedding.shape[-2:]
        input_embedding = input_embedding.reshape(
            [output_shape.numel()] + [l_sum, hidden]
        )
        weight = torch.index_select(
            self.weight, dim=0, index=self.expand_index
        )  # [(L_max + 1) ** 2, C_out, C_in]
        out = torch.einsum(
            "bmi, moi -> bmo", input_embedding, weight
        )  # [N, (L_max + 1) ** 2, C_out]
        bias = self.bias.view(1, 1, self.out_features)
        out[:, 0:1, :] = out.narrow(1, 0, 1) + bias

        out = out.reshape(output_shape + (l_sum, self.out_features))

        return out

    def __repr__(self):
        return f"{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features}, lmax={self.lmax})"


class Learn_PolynomialDistance(torch.nn.Module):
    def __init__(self, degree, highest_degree=3):
        """
        Constructs a polynomial model with learnable coefficients.

        P(d) = c_0 + c_1 * d + c_2 * d^2 + ... + c_n * d^n

        :param degree: The highest degree of the polynomial.
        """
        super().__init__()
        self.coefficients = 0.01 * torch.randn(highest_degree + 1)
        self.coefficients[degree] = 1

        self.coefficients = torch.nn.Parameter(self.coefficients)
        self.act = torch.nn.ReLU()

    def forward(self, distance):
        """
        Computes the polynomial value for a given distance.

        :param distance: The distance value (torch.Tensor)
        :return: The computed polynomial value.
        """
        powers = torch.stack(
            [distance**i for i in range(len(self.coefficients))], dim=-1
        )
        return self.act(torch.sum(self.coefficients * powers, dim=-1))


def drop_path_BL(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], x.shape[1]) + (1,) * (
        x.ndim - 2
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath_BL(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath_BL, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x, batch):
        batch_size = batch.max() + 1
        shape = (batch_size,) + (1,) * (
            x.ndim - 1
        )  # work with diff dim tensors, not just 2D ConvNets
        ones = torch.ones(shape, dtype=x.dtype, device=x.device)

        if len(x.shape) == 4:
            drop = drop_path_BL(ones, self.drop_prob, self.training)
        elif len(x.shape) == 3:
            drop = drop_path(ones, self.drop_prob, self.training)
        return x * drop[batch]

    def extra_repr(self):
        return "drop_prob={}".format(self.drop_prob)


class RadialProfile(nn.Module):
    def __init__(self, ch_list, use_layer_norm=True, use_offset=True):
        super().__init__()
        modules = []
        input_channels = ch_list[0]
        for i in range(len(ch_list)):
            if i == 0:
                continue
            modules.append(nn.Linear(input_channels, ch_list[i], bias=use_offset))
            input_channels = ch_list[i]

            if i == len(ch_list) - 1:
                break

            if use_layer_norm:
                modules.append(nn.LayerNorm(ch_list[i]))
            # modules.append(nn.ReLU())
            # modules.append(Activation(o3.Irreps('{}x0e'.format(ch_list[i])),
            #    acts=[torch.nn.functional.silu]))
            # modules.append(Activation(o3.Irreps('{}x0e'.format(ch_list[i])),
            #    acts=[ShiftedSoftplus()]))
            modules.append(torch.nn.SiLU())

        self.net = nn.Sequential(*modules)

    def forward(self, f_in):
        f_out = self.net(f_in)
        return f_out


class SmoothLeakyReLU(torch.nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()
        self.alpha = negative_slope

    def forward(self, x):
        ## x could be any dimension.
        return (1 - self.alpha) * x * torch.sigmoid(x) + self.alpha * x

    def extra_repr(self):
        return "negative_slope={}".format(self.alpha)


# class SmoothLeakyReLU(torch.nn.Module):
#     def __init__(self, negative_slope=0.2):
#         super().__init__()
#         self.alpha = 0.3 #negative_slope
#         self.func = nn.SiLU()
#     def forward(self, x):
#         ## x could be any dimension.
#         return self.func(x)
#         # return (1-self.alpha) * x * torch.sigmoid(x) + self.alpha * x

#     def extra_repr(self):
#         return "negative_slope={}".format(self.alpha)


class SO3_Linear2Scalar_e2former(torch.nn.Module):
    def __init__(self, in_features, out_features, lmax, bias=True):
        """
        1. Use `torch.einsum` to prevent slicing and concatenation
        2. Need to specify some behaviors in `no_weight_decay` and weight initialization.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lmax = lmax

        self.weight = torch.nn.Parameter(
            torch.randn((self.lmax + 1), out_features // 2, in_features)
        )
        bound = 1 / math.sqrt(self.in_features)
        torch.nn.init.uniform_(self.weight, -bound, bound)
        self.bias = torch.nn.Parameter(torch.zeros(out_features // 2))

        self.weight2 = torch.nn.Parameter(
            torch.randn((self.lmax + 1), out_features // 2, in_features)
        )
        bound = 1 / math.sqrt(self.in_features)
        torch.nn.init.uniform_(self.weight2, -bound, bound)
        self.bias = torch.nn.Parameter(torch.zeros(1, 1, out_features // 2))

        expand_index = torch.zeros([(lmax + 1) ** 2]).long()
        for l in range(lmax + 1):
            start_idx = l**2
            length = 2 * l + 1
            expand_index[start_idx : (start_idx + length)] = l
        self.register_buffer("expand_index", expand_index)

        self.final_linear = nn.Sequential(
            nn.Linear(out_features // 2 * (lmax + 1), out_features),
            nn.LayerNorm(out_features),
            nn.SiLU(),
            nn.Linear(out_features, out_features),
        )

    def forward(self, input_embedding):
        output_shape = input_embedding.shape[:-2]
        l_sum, hidden = input_embedding.shape[-2:]
        input_embedding = input_embedding.reshape(
            [output_shape.numel()] + [l_sum, hidden]
        )
        weight = torch.index_select(
            self.weight, dim=0, index=self.expand_index
        )  # [(L_max + 1) ** 2, C_out, C_in]
        out = torch.einsum(
            "bmi, moi -> bmo", input_embedding, weight
        )  # [N, (L_max + 1) ** 2, C_out]
        out[:, 0:1, :] = out.narrow(1, 0, 1) + self.bias

        weight2 = torch.index_select(
            self.weight2, dim=0, index=self.expand_index
        )  # [(L_max + 1) ** 2, C_out, C_in]
        out2 = torch.einsum(
            "bmi, moi -> bmo", input_embedding, weight2
        )  # [N, (L_max + 1) ** 2, C_out]
        out2[:, 0:1, :] = out2.narrow(1, 0, 1)

        tmp_out = []
        for l in range(self.lmax + 1):
            tmp_out.append(
                torch.sum(
                    out[:, l**2 : (l + 1) ** 2] * out2[:, l**2 : (l + 1) ** 2],
                    dim=1,
                )
            )

        tmp_out = self.final_linear(torch.cat(tmp_out, dim=-1))

        tmp_out = tmp_out.reshape(output_shape + (self.out_features,))

        return tmp_out


class Irreps2Scalar(torch.nn.Module):
    def __init__(
        self,
        irreps_in,
        out_dim,
        hidden_dim=None,
        bias=True,
        act="smoothleakyrelu",
        rescale=True,
    ):
        """
        1. from irreps to scalar output: [...,irreps] - > [...,out_dim]
        2. bias is used for l=0
        3. act is used for l=0
        4. rescale is default, e.g. irreps is c0*l0+c1*l1+c2*l2+c3*l3, rescale weight is 1/c0**0.5 1/c1**0.5 ...
        """
        super().__init__()
        self.irreps_in = (
            o3.Irreps(irreps_in) if isinstance(irreps_in, str) else irreps_in
        )
        if hidden_dim is not None:
            self.hidden_dim = hidden_dim
        else:
            self.hidden_dim = self.irreps_in[0][0]  # l=0 scalar_dim
        self.out_dim = out_dim
        self.act = act
        self.bias = bias
        self.rescale = rescale

        self.vec_proj_list = nn.ModuleList()
        # self.irreps_in_len = sum([mul*(ir.l*2+1) for mul, ir in self.irreps_in])
        # self.scalar_in_len = sum([mul for mul, ir in self.irreps_in])
        self.lirreps = len(self.irreps_in)
        self.output_mlp = nn.Sequential(
            SmoothLeakyReLU(0.2) if self.act == "smoothleakyrelu" else nn.Identity(),
            nn.Linear(self.hidden_dim, out_dim),  # NOTICE init
        )

        for idx in range(len(self.irreps_in)):
            l = self.irreps_in[idx][1].l
            in_feature = self.irreps_in[idx][0]
            if l == 0:
                vec_proj = nn.Linear(in_feature, self.hidden_dim)
                # bound = 1 / math.sqrt(in_feature)
                # torch.nn.init.uniform_(vec_proj.weight, -bound, bound)
                nn.init.xavier_uniform_(vec_proj.weight)
                vec_proj.bias.data.fill_(0)
            else:
                vec_proj = nn.Linear(in_feature, 2 * (self.hidden_dim), bias=False)
                # bound = 1 / math.sqrt(in_feature*(2*l+1))
                # torch.nn.init.uniform_(vec_proj.weight, -bound, bound)
                nn.init.xavier_uniform_(vec_proj.weight)
            self.vec_proj_list.append(vec_proj)

    def forward(self, input_embedding):
        """
        from e3nn import o3
        irreps_in = o3.Irreps("100x1e+40x2e+10x3e")
        irreps_out = o3.Irreps("20x1e+20x2e+20x3e")
        irrepslinear = IrrepsLinear(irreps_in, irreps_out)
        irreps2scalar = Irreps2Scalar(irreps_in, 128)
        node_embed = irreps_in.randn(200,30,5,-1)
        out_scalar = irreps2scalar(node_embed)
        out_irreps = irrepslinear(node_embed)
        """

        # if input_embedding.shape[-1]!=self.irreps_in_len:
        #     raise ValueError("input_embedding should have same length as irreps_in_len")

        shape = list(input_embedding.shape[:-1])
        num = input_embedding.shape[:-1].numel()
        input_embedding = input_embedding.reshape(num, -1)

        start_idx = 0
        scalars = 0
        for idx, (mul, ir) in enumerate(self.irreps_in):
            if idx == 0 and ir.l == 0:
                scalars = self.vec_proj_list[0](
                    input_embedding[..., : self.irreps_in[0][0]]
                )
                start_idx += mul * (2 * ir.l + 1)
                continue
            vec_proj = self.vec_proj_list[idx]
            vec = (
                input_embedding[:, start_idx : start_idx + mul * (2 * ir.l + 1)]
                .reshape(-1, mul, (2 * ir.l + 1))
                .permute(0, 2, 1)
            )  # [B, 2l+1, D]
            vec1, vec2 = torch.split(
                vec_proj(vec), self.hidden_dim, dim=-1
            )  # [B, 2l+1, D]
            vec_dot = (vec1 * vec2).sum(dim=1)  # [B, 2l+1, D]

            scalars = scalars + vec_dot  # TODO: concat
            start_idx += mul * (2 * ir.l + 1)

        output_embedding = self.output_mlp(scalars)
        output_embedding = output_embedding.reshape(shape + [self.out_dim])
        return output_embedding

    def __repr__(self):
        return f"{self.__class__.__name__}(in_features={self.irreps_in}, out_features={self.out_dim}"


# class IrrepsLinear(torch.nn.Module):
#     def __init__(
#         self,
#         irreps_in,
#         irreps_out,
#         hidden_dim=None,
#         bias=True,
#         act="smoothleakyrelu",
#         rescale=_RESCALE,
#     ):
#         """
#         1. from irreps to scalar output: [...,irreps] - > [...,out_dim]
#         2. bias is used for l=0
#         3. act is used for l=0
#         4. rescale is default, e.g. irreps is c0*l0+c1*l1+c2*l2+c3*l3, rescale weight is 1/c0**0.5 1/c1**0.5 ...
#         """
#         super().__init__()
#         self.irreps_in = o3.Irreps(irreps_in) if isinstance(irreps_in,str) else irreps_in
#         self.irreps_out = o3.Irreps(irreps_out) if isinstance(irreps_out,str) else irreps_out

#         self.irreps_in_len = sum([mul*(ir.l*2+1) for mul, ir in self.irreps_in])
#         self.irreps_out_len = sum([mul*(ir.l*2+1) for mul, ir in self.irreps_out])
#         if hidden_dim is not None:
#             self.hidden_dim = hidden_dim
#         else:
#             self.hidden_dim = self.irreps_in[0][0]  # l=0 scalar_dim
#         self.act = act
#         self.bias = bias
#         self.rescale = rescale

#         self.vec_proj_list = nn.ModuleList()
#         # self.irreps_in_len = sum([mul*(ir.l*2+1) for mul, ir in self.irreps_in])
#         # self.scalar_in_len = sum([mul for mul, ir in self.irreps_in])
#         self.output_mlp = nn.Sequential(
#             SmoothLeakyReLU(0.2) if self.act == "smoothleakyrelu" else nn.Identity(),
#             nn.Linear(self.hidden_dim, self.irreps_out[0][0]),
#         )
#         self.weight_list = nn.ParameterList()
#         for idx in range(len(self.irreps_in)):
#             l = self.irreps_in[idx][1].l
#             in_feature = self.irreps_in[idx][0]
#             if l == 0:
#                 vec_proj = nn.Linear(in_feature, self.hidden_dim)
#                 nn.init.xavier_uniform_(vec_proj.weight)
#                 vec_proj.bias.data.fill_(0)
#             else:
#                 vec_proj = nn.Linear(in_feature, 2 * self.hidden_dim, bias=False)
#                 nn.init.xavier_uniform_(vec_proj.weight)

#                 # weight for l>0
#                 out_feature = self.irreps_out[idx][0]
#                 weight = torch.nn.Parameter(
#                                 torch.randn( out_feature,in_feature)
#                             )
#                 bound = 1 / math.sqrt(in_feature) if self.rescale else 1
#                 torch.nn.init.uniform_(weight, -bound, bound)
#                 self.weight_list.append(weight)

#             self.vec_proj_list.append(vec_proj)


#     def forward(self, input_embedding):
#         """
#         from e3nn import o3
#         irreps_in = o3.Irreps("100x1e+40x2e+10x3e")
#         irreps_out = o3.Irreps("20x1e+20x2e+20x3e")
#         irrepslinear = IrrepsLinear(irreps_in, irreps_out)
#         irreps2scalar = Irreps2Scalar(irreps_in, 128)
#         node_embed = irreps_in.randn(200,30,5,-1)
#         out_scalar = irreps2scalar(node_embed)
#         out_irreps = irrepslinear(node_embed)
#         """

#         # if input_embedding.shape[-1]!=self.irreps_in_len:
#         #     raise ValueError("input_embedding should have same length as irreps_in_len")

#         shape = list(input_embedding.shape[:-1])
#         num = input_embedding.shape[:-1].numel()
#         input_embedding = input_embedding.reshape(num, -1)

#         start_idx = 0
#         scalars = self.vec_proj_list[0](input_embedding[..., : self.irreps_in[0][0]])
#         output_embedding = []
#         for idx, (mul, ir) in enumerate(self.irreps_in):
#             if idx == 0:
#                 start_idx += mul * (2 * ir.l + 1)
#                 continue
#             vec_proj = self.vec_proj_list[idx]
#             vec = (
#                 input_embedding[:, start_idx : start_idx + mul * (2 * ir.l + 1)]
#                 .reshape(-1, mul, (2 * ir.l + 1))
#             )  # [B, D, 2l+1]
#             vec1, vec2 = torch.split(
#                 vec_proj(vec.permute(0, 2, 1)), self.hidden_dim, dim=-1
#             )  # [B, 2l+1, D]
#             vec_dot = (vec1 * vec2).sum(dim=1)  # [B, 2l+1, D]

#             scalars = scalars + vec_dot # TODO: concat

#             # linear for l>0
#             weight = self.weight_list[idx-1]
#             out = torch.matmul(weight,vec).reshape(num,-1) # [B*L, -1]
#             output_embedding.append(out)

#             start_idx += mul * (2 * ir.l + 1)
#         try:
#             scalars = self.output_mlp(scalars)
#         except:
#             raise ValueError(f"scalars shape: {scalars.shape}")
#         output_embedding.insert(0, scalars)
#         output_embedding = torch.cat(output_embedding, dim=1)
#         output_embedding = output_embedding.reshape(shape + [self.irreps_out_len])
#         return output_embedding

#     def __repr__(self):
#         return f"{self.__class__.__name__}(in_features={self.irreps_in}, out_features={self.irreps_out}"


class IrrepsLinear(torch.nn.Module):
    def __init__(
        self, irreps_in, irreps_out, bias=True, act="smoothleakyrelu", rescale=True
    ):
        """
        1. from irreps_in to irreps_out output: [...,irreps_in] - > [...,irreps_out]
        2. bias is used for l=0
        3. act is used for l=0
        4. rescale is default, e.g. irreps is c0*l0+c1*l1+c2*l2+c3*l3, rescale weight is 1/c0**0.5 1/c1**0.5 ...
        """
        super().__init__()
        self.irreps_in = (
            o3.Irreps(irreps_in) if isinstance(irreps_in, str) else irreps_in
        )
        self.irreps_out = (
            o3.Irreps(irreps_out) if isinstance(irreps_out, str) else irreps_out
        )

        self.act = act
        self.bias = bias
        self.rescale = rescale

        for idx2 in range(len(self.irreps_out)):
            if self.irreps_out[idx2][1] not in self.irreps_in:
                raise ValueError(
                    f"Error: each irrep of irreps_out {self.irreps_out} should be in irreps_in {self.irreps_in}. Please check your input and output "
                )

        self.weight_list = nn.ParameterList()
        self.bias_list = nn.ParameterList()
        self.act_list = nn.ModuleList()
        self.irreps_in_len = sum([mul * (ir.l * 2 + 1) for mul, ir in self.irreps_in])
        self.irreps_out_len = sum([mul * (ir.l * 2 + 1) for mul, ir in self.irreps_out])
        self.instructions = []
        start_idx = 0
        for idx1 in range(len(self.irreps_in)):
            l = self.irreps_in[idx1][1].l
            mul = self.irreps_in[idx1][0]
            for idx2 in range(len(self.irreps_out)):
                if self.irreps_in[idx1][1].l == self.irreps_out[idx2][1].l:
                    self.instructions.append(
                        [idx1, mul, l, start_idx, start_idx + (l * 2 + 1) * mul]
                    )
                    out_feature = self.irreps_out[idx2][0]

                    weight = torch.nn.Parameter(torch.randn(out_feature, mul))
                    bound = 1 / math.sqrt(mul) if self.rescale else 1
                    torch.nn.init.uniform_(weight, -bound, bound)
                    self.weight_list.append(weight)

                    bias = torch.nn.Parameter(
                        torch.randn(1, out_feature, 1)
                        if self.bias and l == 0
                        else torch.zeros(1, out_feature, 1)
                    )
                    self.bias_list.append(bias)

                    activation = (
                        nn.Sequential(SmoothLeakyReLU())
                        if self.act == "smoothleakyrelu" and l == 0
                        else nn.Sequential()
                    )
                    self.act_list.append(activation)

            start_idx += (l * 2 + 1) * mul

    def forward(self, input_embedding):
        """
        from e3nn import o3
        irreps_in = o3.Irreps("100x1e+40x2e+10x3e")
        irreps_out = o3.Irreps("20x1e+20x2e+20x3e")
        irrepslinear = IrrepsLinear(irreps_in, irreps_out)
        irreps2scalar = Irreps2Scalar(irreps_in, 128)
        node_embed = irreps_in.randn(200,30,5,-1)
        out_scalar = irreps2scalar(node_embed)
        out_irreps = irrepslinear(node_embed)
        """

        if input_embedding.shape[-1] != self.irreps_in_len:
            raise ValueError("input_embedding should have same length as irreps_in_len")

        shape = list(input_embedding.shape[:-1])
        num = input_embedding.shape[:-1].numel()
        input_embedding = input_embedding.reshape(num, -1)

        output_embedding = []
        for idx, (_, mul, l, start, end) in enumerate(self.instructions):
            weight = self.weight_list[idx]
            bias = self.bias_list[idx]
            activation = self.act_list[idx]

            out = (
                torch.matmul(
                    weight, input_embedding[:, start:end].reshape(-1, mul, (2 * l + 1))
                )
                + bias
            )
            out = activation(out).reshape(num, -1)
            output_embedding.append(out)

        output_embedding = torch.cat(output_embedding, dim=1)
        output_embedding = output_embedding.reshape(shape + [self.irreps_out_len])
        return output_embedding


@compile_mode("script")
class Vec2AttnHeads(torch.nn.Module):
    """
    Reshape vectors of shape [..., irreps_head] to vectors of shape
    [..., num_heads, irreps_head].
    """

    def __init__(self, irreps_head, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.irreps_head = irreps_head
        self.irreps_mid_in = []
        for mul, ir in irreps_head:
            self.irreps_mid_in.append((mul * num_heads, ir))
        self.irreps_mid_in = o3.Irreps(self.irreps_mid_in)
        self.mid_in_indices = []
        start_idx = 0
        for mul, ir in self.irreps_mid_in:
            self.mid_in_indices.append((start_idx, start_idx + mul * ir.dim))
            start_idx = start_idx + mul * ir.dim

    def forward(self, x):
        shape = list(x.shape[:-1])
        num = x.shape[:-1].numel()
        x = x.reshape(num, -1)

        N, _ = x.shape
        out = []
        for ir_idx, (start_idx, end_idx) in enumerate(self.mid_in_indices):
            temp = x.narrow(1, start_idx, end_idx - start_idx)
            temp = temp.reshape(N, self.num_heads, -1)
            out.append(temp)
        out = torch.cat(out, dim=2)
        out = out.reshape(shape + [self.num_heads, -1])
        return out

    def __repr__(self):
        return "{}(irreps_head={}, num_heads={})".format(
            self.__class__.__name__, self.irreps_head, self.num_heads
        )


@compile_mode("script")
class AttnHeads2Vec(torch.nn.Module):
    """
    Convert vectors of shape [..., num_heads, irreps_head] into
    vectors of shape [..., irreps_head * num_heads].
    """

    def __init__(self, irreps_head, num_heads=-1):
        super().__init__()
        self.irreps_head = irreps_head
        self.num_heads = num_heads
        self.head_indices = []
        start_idx = 0
        for mul, ir in self.irreps_head:
            self.head_indices.append((start_idx, start_idx + mul * ir.dim))
            start_idx = start_idx + mul * ir.dim

    def forward(self, x):
        head_cnt = x.shape[-2]
        shape = list(x.shape[:-2])
        num = x.shape[:-2].numel()
        x = x.reshape(num, head_cnt, -1)
        N, _, _ = x.shape
        out = []
        for ir_idx, (start_idx, end_idx) in enumerate(self.head_indices):
            temp = x.narrow(2, start_idx, end_idx - start_idx)
            temp = temp.reshape(N, -1)
            out.append(temp)
        out = torch.cat(out, dim=1)
        out = out.reshape(shape + [-1])
        return out

    def __repr__(self):
        return "{}(irreps_head={})".format(self.__class__.__name__, self.irreps_head)


# class EquivariantDropout(nn.Module):
#     def __init__(self, irreps, drop_prob):
#         """
#         equivariant for irreps: [..., irreps]
#         """

#         super(EquivariantDropout, self).__init__()
#         self.irreps = irreps
#         self.num_irreps = irreps.num_irreps
#         self.drop_prob = drop_prob
#         self.drop = torch.nn.Dropout(drop_prob, True)
#         self.mul = o3.ElementwiseTensorProduct(
#             irreps, o3.Irreps("{}x0e".format(self.num_irreps))
#         )

#     def forward(self, x):
#         """
#         x: [..., irreps]

#         t1 = o3.Irreps("5x0e+4x1e+3x2e")
#         func = EquivariantDropout(t1, 0.5)
#         out = func(t1.randn(2,3,-1))
#         """
#         if not self.training or self.drop_prob == 0.0:
#             return x

#         shape = x.shape
#         N = x.shape[:-1].numel()
#         x = x.reshape(N, -1)
#         mask = torch.ones((N, self.num_irreps), dtype=x.dtype, device=x.device)
#         mask = self.drop(mask)

#         out = self.mul(x, mask)

#         return out.reshape(shape)


class EquivariantDropout(nn.Module):
    def __init__(self, dim, lmax, drop_prob):
        """
        equivariant for irreps: [..., irreps]
        """

        super(EquivariantDropout, self).__init__()
        self.lmax = lmax
        self.scalar_dim = dim
        self.drop_prob = drop_prob
        self.drop = torch.nn.Dropout(drop_prob, True)

    def forward(self, x):
        """
        x: [..., irreps]

        t1 = o3.Irreps("5x0e+4x1e+3x2e")
        func = EquivariantDropout(t1, 0.5)
        out = func(t1.randn(2,3,-1))
        """
        if not self.training or self.drop_prob == 0.0:
            return x
        shape = x.shape
        N = x.shape[:-2].numel()
        x = x.reshape(N, (self.lmax + 1) ** 2, -1)

        mask = torch.ones(
            (N, self.lmax + 1, self.scalar_dim), dtype=x.dtype, device=x.device
        )
        mask = self.drop(mask)
        out = []
        for l in range(self.lmax + 1):
            out.append(x[:, l**2 : (l + 1) ** 2] * mask[:, l : l + 1])
        out = torch.cat(out, dim=1)
        return out.reshape(shape)


class TensorProductRescale(torch.nn.Module):
    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        instructions,
        bias=True,
        rescale=True,
        internal_weights=None,
        shared_weights=None,
        normalization=None,
        mode="default",
    ):
        super().__init__()

        self.irreps_in1 = irreps_in1
        self.irreps_in2 = irreps_in2
        self.irreps_out = irreps_out
        self.rescale = rescale
        self.use_bias = bias

        # e3nn.__version__ == 0.4.4
        # Use `path_normalization` == 'none' to remove normalization factor
        if mode == "simple":
            self.tp = Simple_TensorProduct(
                irreps_in1=self.irreps_in1,
                irreps_in2=self.irreps_in2,
                irreps_out=self.irreps_out,
                instructions=instructions,
                rescale=rescale,
                # normalization=normalization,
                # internal_weights=internal_weights,
                # shared_weights=shared_weights,
                # path_normalization="none",
            )
        else:
            self.tp = o3.TensorProduct(
                irreps_in1=self.irreps_in1,
                irreps_in2=self.irreps_in2,
                irreps_out=self.irreps_out,
                instructions=instructions,
                normalization=normalization,
                internal_weights=internal_weights,
                shared_weights=shared_weights,
                path_normalization="none",
            )

        self.init_rescale_bias()

    def calculate_fan_in(self, ins):
        return {
            "uvw": (self.irreps_in1[ins.i_in1].mul * self.irreps_in2[ins.i_in2].mul),
            "uvu": self.irreps_in2[ins.i_in2].mul,
            "uvv": self.irreps_in1[ins.i_in1].mul,
            "uuw": self.irreps_in1[ins.i_in1].mul,
            "uuu": 1,
            "uvuv": 1,
            "uvu<v": 1,
            "u<vw": self.irreps_in1[ins.i_in1].mul
            * (self.irreps_in2[ins.i_in2].mul - 1)
            // 2,
        }[ins.connection_mode]

    def init_rescale_bias(self) -> None:
        irreps_out = self.irreps_out
        # For each zeroth order output irrep we need a bias
        # Determine the order for each output tensor and their dims
        self.irreps_out_orders = [
            int(irrep_str[-2]) for irrep_str in str(irreps_out).split("+")
        ]
        self.irreps_out_dims = [
            int(irrep_str.split("x")[0]) for irrep_str in str(irreps_out).split("+")
        ]
        self.irreps_out_slices = irreps_out.slices()

        # Store tuples of slices and corresponding biases in a list
        self.bias = None
        self.bias_slices = []
        self.bias_slice_idx = []
        self.irreps_bias = self.irreps_out.simplify()
        self.irreps_bias_orders = [
            int(irrep_str[-2]) for irrep_str in str(self.irreps_bias).split("+")
        ]
        self.irreps_bias_parity = [
            irrep_str[-1] for irrep_str in str(self.irreps_bias).split("+")
        ]
        self.irreps_bias_dims = [
            int(irrep_str.split("x")[0])
            for irrep_str in str(self.irreps_bias).split("+")
        ]
        if self.use_bias:
            self.bias = []
            for slice_idx in range(len(self.irreps_bias_orders)):
                if (
                    self.irreps_bias_orders[slice_idx] == 0
                    and self.irreps_bias_parity[slice_idx] == "e"
                ):
                    out_slice = self.irreps_bias.slices()[slice_idx]
                    out_bias = torch.nn.Parameter(
                        torch.zeros(
                            self.irreps_bias_dims[slice_idx], dtype=self.tp.weight.dtype
                        )
                    )
                    self.bias += [out_bias]
                    self.bias_slices += [out_slice]
                    self.bias_slice_idx += [slice_idx]
        self.bias = torch.nn.ParameterList(self.bias)

        self.slices_sqrt_k = {}
        with torch.no_grad():
            # Determine fan_in for each slice, it could be that each output slice is updated via several instructions
            slices_fan_in = {}  # fan_in per slice
            for instr in self.tp.instructions:
                slice_idx = instr[2]
                fan_in = self.calculate_fan_in(instr)
                slices_fan_in[slice_idx] = (
                    slices_fan_in[slice_idx] + fan_in
                    if slice_idx in slices_fan_in.keys()
                    else fan_in
                )
            for instr in self.tp.instructions:
                slice_idx = instr[2]
                if self.rescale:
                    sqrt_k = 1 / slices_fan_in[slice_idx] ** 0.5
                else:
                    sqrt_k = 1.0
                self.slices_sqrt_k[slice_idx] = (
                    self.irreps_out_slices[slice_idx],
                    sqrt_k,
                )

            # Re-initialize weights in each instruction
            if self.tp.internal_weights:
                for weight, instr in zip(self.tp.weight_views(), self.tp.instructions):
                    # The tensor product in e3nn already normalizes proportional to 1 / sqrt(fan_in), and the weights are by
                    # default initialized with unif(-1,1). However, we want to be consistent with torch.nn.Linear and
                    # initialize the weights with unif(-sqrt(k),sqrt(k)), with k = 1 / fan_in
                    slice_idx = instr[2]
                    if self.rescale:
                        sqrt_k = 1 / slices_fan_in[slice_idx] ** 0.5
                        weight.data.mul_(sqrt_k)
                    # else:
                    #    sqrt_k = 1.
                    #
                    # if self.rescale:
                    # weight.data.uniform_(-sqrt_k, sqrt_k)
                    #    weight.data.mul_(sqrt_k)
                    # self.slices_sqrt_k[slice_idx] = (self.irreps_out_slices[slice_idx], sqrt_k)

            # Initialize the biases
            # for (out_slice_idx, out_slice, out_bias) in zip(self.bias_slice_idx, self.bias_slices, self.bias):
            #    sqrt_k = 1 / slices_fan_in[out_slice_idx] ** 0.5
            #    out_bias.uniform_(-sqrt_k, sqrt_k)

    def forward_tp_rescale_bias(self, x, y, weight=None):
        out = self.tp(x, y, weight)
        # if self.rescale and self.tp.internal_weights:
        #    for (slice, slice_sqrt_k) in self.slices_sqrt_k.values():
        #        out[:, slice] /= slice_sqrt_k
        if self.use_bias:
            for _, slice, bias in zip(self.bias_slice_idx, self.bias_slices, self.bias):
                # out[:, slice] += bias
                out.narrow(-1, slice.start, slice.stop - slice.start).add_(bias)
        return out

    def forward(self, x, y, weight=None):
        out = self.forward_tp_rescale_bias(x, y, weight)
        return out


# class SeparableFCTP(torch.nn.Module):
#     def __init__(
#         self,
#         irreps_x,
#         irreps_y,
#         irreps_out,
#         fc_neurons,
#         use_activation=False,
#         norm_layer="graph",
#         internal_weights=False,
#         mode="default",
#         connection_mode='uvu',
#         rescale=True,
#         eqv2=False
#     ):
#         """
#         Use separable FCTP for spatial convolution.
#         [...,irreps_x] tp [...,irreps_y] - > [..., irreps_out]

#         fc_neurons is not needed in e2former
#         """

#         super().__init__()
#         self.irreps_node_input = o3.Irreps(irreps_x)
#         self.irreps_edge_attr = o3.Irreps(irreps_y)
#         self.irreps_node_output = o3.Irreps(irreps_out)
#         norm = get_norm_layer(norm_layer)


#         irreps_output = []
#         instructions = []

#         for i, (mul, ir_in) in enumerate(self.irreps_node_input):
#             for j, (_, ir_edge) in enumerate(self.irreps_edge_attr):
#                 for ir_out in ir_in * ir_edge:
#                     if ir_out in self.irreps_node_output: # or ir_out == o3.Irrep(0, 1):
#                         k = len(irreps_output)
#                         irreps_output.append((mul, ir_out))
#                         instructions.append((i, j, k, connection_mode, True))

#         irreps_output = o3.Irreps(irreps_output)
#         irreps_output, p, _ = sort_irreps_even_first(irreps_output)  # irreps_output.sort()
#         instructions = [
#             (i_1, i_2, p[i_out], mode, train)
#             for i_1, i_2, i_out, mode, train in instructions
#         ]
#         if mode != "default":
#             if internal_weights is False:
#                 raise ValueError("tp not support some parameter, please check your code.")

#         if eqv2==True:
#             self.dtp = TensorProductRescale(
#                 self.irreps_node_input,
#                 self.irreps_edge_attr,
#                 irreps_output,
#                 instructions,
#                 internal_weights=internal_weights,
#                 shared_weights=True,
#                 bias=False,
#                 rescale=rescale,
#                 mode=mode,
#             )


#             self.dtp_rad = None
#             self.fc_neurons = fc_neurons
#             if fc_neurons is not None:
#                 warnings.warn("NOTICEL: fc_neurons is not needed in e2former")
#                 self.dtp_rad = RadialProfile(fc_neurons + [self.dtp.tp.irreps_out.num_irreps])
#                 # for slice, slice_sqrt_k in self.dtp.slices_sqrt_k.values():
#                 #     self.dtp_rad.net[-1].weight.data[slice, :] *= slice_sqrt_k
#                 #     self.dtp_rad.offset.data[slice] *= slice_sqrt_k

#             self.norm = None

#             if use_activation:
#                 irreps_lin_output = self.irreps_node_output
#                 irreps_scalars, irreps_gates, irreps_gated = irreps2gate(
#                     self.irreps_node_output
#                 )
#                 irreps_lin_output = irreps_scalars + irreps_gates + irreps_gated
#                 irreps_lin_output = irreps_lin_output.simplify()
#                 self.lin = IrrepsLinear(
#                     self.dtp.irreps_out.simplify(), irreps_lin_output, bias=False, act=None
#                 )
#                 if norm_layer is not None:
#                     self.norm = norm(irreps_lin_output)

#             else:
#                 self.lin = IrrepsLinear(
#                     self.dtp.irreps_out.simplify(), self.irreps_node_output, bias=False, act=None
#                 )
#                 if norm_layer is not None:
#                     self.norm = norm(self.irreps_node_output)

#             self.gate = None
#             if use_activation:
#                 if irreps_gated.num_irreps == 0:
#                     gate = Activation(self.irreps_node_output, acts=[torch.nn.SiLU()])
#                 else:
#                     gate = Gate(
#                         irreps_scalars,
#                         [torch.nn.SiLU() for _, ir in irreps_scalars],  # scalar
#                         irreps_gates,
#                         [torch.sigmoid for _, ir in irreps_gates],  # gates (scalars)
#                         irreps_gated,  # gated tensors
#                     )
#                 self.gate = gate
#         else:
#             self.dtp = TensorProductRescale(
#                 self.irreps_node_input,
#                 self.irreps_edge_attr,
#                 irreps_output,
#                 instructions,
#                 internal_weights=internal_weights,
#                 shared_weights=internal_weights,
#                 bias=False,
#                 rescale=rescale,
#                 mode=mode,
#             )


#             self.dtp_rad = None
#             self.fc_neurons = fc_neurons
#             if fc_neurons is not None:
#                 warnings.warn("NOTICEL: fc_neurons is not needed in e2former")
#                 self.dtp_rad = RadialProfile(fc_neurons + [self.dtp.tp.weight_numel])
#                 for slice, slice_sqrt_k in self.dtp.slices_sqrt_k.values():
#                     self.dtp_rad.net[-1].weight.data[slice, :] *= slice_sqrt_k
#                     self.dtp_rad.offset.data[slice] *= slice_sqrt_k

#             irreps_lin_output = self.irreps_node_output
#             irreps_scalars, irreps_gates, irreps_gated = irreps2gate(
#                 self.irreps_node_output
#             )
#             if use_activation:
#                 irreps_lin_output = irreps_scalars + irreps_gates + irreps_gated
#                 irreps_lin_output = irreps_lin_output.simplify()
#             self.lin = IrrepsLinear(
#                 self.dtp.irreps_out.simplify(), irreps_lin_output, bias=False, act=None
#             )

#             self.norm = None
#             if norm_layer is not None:
#                 self.norm = norm(self.irreps_node_output)

#             self.gate = None
#             if use_activation:
#                 if irreps_gated.num_irreps == 0:
#                     gate = Activation(self.irreps_node_output, acts=[torch.nn.SiLU()])
#                 else:
#                     gate = Gate(
#                         irreps_scalars,
#                         [torch.nn.SiLU() for _, ir in irreps_scalars],  # scalar
#                         irreps_gates,
#                         [torch.sigmoid for _, ir in irreps_gates],  # gates (scalars)
#                         irreps_gated,  # gated tensors
#                     )
#                 self.gate = gate

#     def forward(self, irreps_x, irreps_y, xy_scalar_fea, batch=None,eqv2=False, **kwargs):
#         """
#         x: [..., irreps]

#         irreps_in = o3.Irreps("256x0e+64x1e+32x2e")
#         sep_tp = SeparableFCTP(irreps_in,"1x1e",irreps_in,fc_neurons=None,
#                             use_activation=False,norm_layer=None,
#                             internal_weights=True)
#         out = sep_tp(irreps_in.randn(100,10,-1),torch.randn(100,10,3),None)
#         print(out.shape)
#         """
#         if eqv2==True:
#             shape = irreps_x.shape[:-2]
#             N = irreps_x.shape[:-2].numel()
#             irreps_x = self.from_eqv2toe3nn(irreps_x)
#             irreps_y = irreps_y.reshape(N, -1)

#             out = self.dtp(irreps_x, irreps_y, None)
#             if self.dtp_rad is not None and xy_scalar_fea is not None:
#                 xy_scalar_fea = xy_scalar_fea.reshape(N, -1)
#                 weight = self.dtp_rad(xy_scalar_fea)
#                 temp = []
#                 start = 0
#                 start_scalar = 0
#                 for mul,(ir,_) in self.dtp.tp.irreps_out.simplify():
#                     temp.append((out[:,start:start+(2*ir+1)*mul].reshape(-1,mul,2*ir+1)*\
#                                                 weight[:,start_scalar:start_scalar+mul].unsqueeze(-1)).reshape(-1,(2*ir+1)*mul))
#                     start_scalar += mul
#                     start += (2*ir+1)*mul
#                 out = torch.cat(temp,dim = -1)
#             out = self.lin(out)
#             if self.norm is not None:
#                 out = self.norm(out, batch=batch)
#             if self.gate is not None:
#                 out = self.gate(out)
#             return self.from_e3nntoeqv2(out)
#         else:
#             shape = irreps_x.shape[:-1]
#             N = irreps_x.shape[:-1].numel()
#             irreps_x = irreps_x.reshape(N, -1)
#             irreps_y = irreps_y.reshape(N, -1)

#             weight = None
#             if self.dtp_rad is not None and xy_scalar_fea is not None:
#                 xy_scalar_fea = xy_scalar_fea.reshape(N, -1)
#                 weight = self.dtp_rad(xy_scalar_fea)
#             out = self.dtp(irreps_x, irreps_y, weight)
#             out = self.lin(out)
#             if self.norm is not None:
#                 out = self.norm(out, batch=batch)
#             if self.gate is not None:
#                 out = self.gate(out)
#             return out.reshape(list(shape) + [-1])


#     def from_eqv2toe3nn(self,embedding):
#         BL = embedding.shape[0]
#         lmax = self.irreps_node_input[-1][1][0]
#         start = 0
#         out = []
#         for l in range(1+lmax):
#             out.append(embedding[:,start:start+2*l+1,:].permute(0,2,1).reshape(BL,-1))
#             start += 2*l+1
#         return torch.cat(out,dim = -1)


#     def from_e3nntoeqv2(self,embedding):
#         lmax = self.irreps_node_output[-1][1][0]
#         mul = self.irreps_node_output[-1][0]

#         start = 0
#         out = []
#         for l in range(1+lmax):
#             out.append(embedding[:,start:start+mul*(2*l+1)].reshape(-1,mul,2*l+1).permute(0,2,1))
#             start += mul*(2*l+1)
#         return torch.cat(out,dim = 1)


class CosineCutoff(torch.nn.Module):
    r"""Appies a cosine cutoff to the input distances.

    .. math::
        \text{cutoffs} =
        \begin{cases}
        0.5 * (\cos(\frac{\text{distances} * \pi}{\text{cutoff}}) + 1.0),
        & \text{if } \text{distances} < \text{cutoff} \\
        0, & \text{otherwise}
        \end{cases}

    Args:
        cutoff (float): A scalar that determines the point at which the cutoff
            is applied.
    """

    def __init__(self, cutoff: float) -> None:
        super().__init__()
        self.cutoff = cutoff

    def forward(self, distances):
        r"""Applies a cosine cutoff to the input distances.

        Args:
            distances (torch.Tensor): A tensor of distances.

        Returns:
            cutoffs (torch.Tensor): A tensor where the cosine function
                has been applied to the distances,
                but any values that exceed the cutoff are set to 0.
        """
        cutoffs = 0.5 * ((distances * math.pi / self.cutoff).cos() + 1.0)
        cutoffs = cutoffs * (distances < self.cutoff).float()
        return cutoffs


def get_mul_0(irreps):
    mul_0 = 0
    for mul, ir in irreps:
        if ir.l == 0 and ir.p == 1:
            mul_0 += mul
    return mul_0


@compile_mode("trace")
class Activation(torch.nn.Module):
    """
    Directly apply activation when irreps is type-0.
    """

    def __init__(self, irreps_in, acts):
        super().__init__()
        if isinstance(irreps_in, str):
            irreps_in = o3.Irreps(irreps_in)
        assert len(irreps_in) == len(acts), (irreps_in, acts)

        # normalize the second moment
        acts = [
            e3nn.math.normalize2mom(act) if act is not None else None for act in acts
        ]

        from e3nn.util._argtools import _get_device

        irreps_out = []
        for (mul, (l_in, p_in)), act in zip(irreps_in, acts):
            if act is not None:
                if l_in != 0:
                    raise ValueError(
                        "Activation: cannot apply an activation function to a non-scalar input."
                    )

                x = torch.linspace(0, 10, 256, device=_get_device(act))

                a1, a2 = act(x), act(-x)
                if (a1 - a2).abs().max() < 1e-5:
                    p_act = 1
                elif (a1 + a2).abs().max() < 1e-5:
                    p_act = -1
                else:
                    p_act = 0

                p_out = p_act if p_in == -1 else p_in
                irreps_out.append((mul, (0, p_out)))

                if p_out == 0:
                    raise ValueError(
                        "Activation: the parity is violated! The input scalar is odd but the activation is neither even nor odd."
                    )
            else:
                irreps_out.append((mul, (l_in, p_in)))

        self.irreps_in = irreps_in
        self.irreps_out = o3.Irreps(irreps_out)
        self.acts = torch.nn.ModuleList(acts)
        assert len(self.irreps_in) == len(self.acts)

    # def __repr__(self):
    #    acts = "".join(["x" if a is not None else " " for a in self.acts])
    #    return f"{self.__class__.__name__} [{self.acts}] ({self.irreps_in} -> {self.irreps_out})"
    def extra_repr(self):
        output_str = super(Activation, self).extra_repr()
        output_str = output_str + "{} -> {}, ".format(self.irreps_in, self.irreps_out)
        return output_str

    def forward(self, features, dim=-1):
        # directly apply activation without narrow
        if len(self.acts) == 1:
            return self.acts[0](features)

        output = []
        index = 0
        for (mul, ir), act in zip(self.irreps_in, self.acts):
            if act is not None:
                output.append(act(features.narrow(dim, index, mul)))
            else:
                output.append(features.narrow(dim, index, mul * ir.dim))
            index += mul * ir.dim

        if len(output) > 1:
            return torch.cat(output, dim=dim)
        elif len(output) == 1:
            return output[0]
        else:
            return torch.zeros_like(features)


@compile_mode("script")
class Gate(torch.nn.Module):
    """
    TODO: to be optimized.  Toooooo ugly
    1. Use `narrow` to split tensor.
    2. Use `Activation` in this file.
    """

    def __init__(
        self, irreps_scalars, act_scalars, irreps_gates, act_gates, irreps_gated
    ):
        super().__init__()
        irreps_scalars = o3.Irreps(irreps_scalars)
        irreps_gates = o3.Irreps(irreps_gates)
        irreps_gated = o3.Irreps(irreps_gated)

        if len(irreps_gates) > 0 and irreps_gates.lmax > 0:
            raise ValueError(
                f"Gate scalars must be scalars, instead got irreps_gates = {irreps_gates}"
            )
        if len(irreps_scalars) > 0 and irreps_scalars.lmax > 0:
            raise ValueError(
                f"Scalars must be scalars, instead got irreps_scalars = {irreps_scalars}"
            )
        if irreps_gates.num_irreps != irreps_gated.num_irreps:
            raise ValueError(
                f"There are {irreps_gated.num_irreps} irreps in irreps_gated, but a different number ({irreps_gates.num_irreps}) of gate scalars in irreps_gates"
            )
        # assert len(irreps_scalars) == 1
        # assert len(irreps_gates) == 1

        self.irreps_scalars = irreps_scalars
        self.irreps_gates = irreps_gates
        self.irreps_gated = irreps_gated
        self._irreps_in = (irreps_scalars + irreps_gates + irreps_gated).simplify()

        self.act_scalars = Activation(irreps_scalars, act_scalars)
        irreps_scalars = self.act_scalars.irreps_out

        self.act_gates = Activation(irreps_gates, act_gates)
        irreps_gates = self.act_gates.irreps_out

        self.mul = o3.ElementwiseTensorProduct(irreps_gated, irreps_gates)
        irreps_gated = self.mul.irreps_out

        self._irreps_out = irreps_scalars + irreps_gated

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps_in} -> {self.irreps_out})"

    def forward(self, features):
        scalars_dim = self.irreps_scalars.dim
        gates_dim = self.irreps_gates.dim
        input_dim = self.irreps_in.dim

        scalars = features.narrow(-1, 0, scalars_dim)
        gates = features.narrow(-1, scalars_dim, gates_dim)
        gated = features.narrow(
            -1, (scalars_dim + gates_dim), (input_dim - scalars_dim - gates_dim)
        )

        scalars = self.act_scalars(scalars)
        if gates.shape[-1]:
            gates = self.act_gates(gates)
            gated = self.mul(gated, gates)
            features = torch.cat([scalars, gated], dim=-1)
        else:
            features = scalars
        return features

    @property
    def irreps_in(self):
        """Input representations."""
        return self._irreps_in

    @property
    def irreps_out(self):
        """Output representations."""
        return self._irreps_out


@compile_mode("script")
class Gate_s3(torch.nn.Module):
    """
    TODO: to be optimized.  Toooooo ugly
    1. Use `narrow` to split tensor.
    2. Use `Activation` in this file.
    """

    def __init__(self, sphere_channels, lmax, act_scalars="silu", act_vector="sigmoid"):
        super().__init__()

        self.sphere_channels = sphere_channels
        self.lmax = lmax
        self.gates = torch.nn.Linear(sphere_channels, sphere_channels * (lmax + 1))
        bound = 1 / math.sqrt(sphere_channels)
        torch.nn.init.uniform_(self.gates.weight, -bound, bound)

        if act_scalars == "silu":
            self.act_scalars = e3nn.math.normalize2mom(torch.nn.SiLU())
        else:
            raise ValueError("in Gate, only support silu")

        if act_vector == "sigmoid":
            self.act_vector = e3nn.math.normalize2mom(torch.nn.Sigmoid())
        else:
            raise ValueError("in Gate, only support sigmoid for vector")

    def __repr__(self):
        return f"{self.__class__.__name__} sph ({self.sphere_channels} lmax {self.lmax}"

    def forward(self, features):
        input_shape = features.shape
        features = features.reshape(input_shape[:-2].numel(), -1, input_shape[-1])

        scalars = self.gates(features[:, 0:1])
        out = [self.act_scalars(scalars[:, :, : self.sphere_channels])]

        start = 1
        for l in range(1, self.lmax + 1):
            out.append(
                self.act_vector(
                    scalars[
                        :,
                        :,
                        l * self.sphere_channels : l * self.sphere_channels
                        + self.sphere_channels,
                    ]
                )  # __ * 1 * hidden_dim
                * features[:, start : start + 2 * l + 1, :]  # __ * (2l+1) * hidden_dim
            )
            start += 2 * l + 1

        out = torch.cat(out, dim=1)
        return out.reshape(input_shape)

    @property
    def irreps_in(self):
        """Input representations."""
        return self.out


@compile_mode("script")
class FeedForwardNetwork_s3(torch.nn.Module):
    """
    Use two (FCTP + Gate)
    """

    def __init__(
        self,
        sphere_channels,
        hidden_channels,
        output_channels,
        lmax,
    ):
        super().__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels

        self.slinear_1 = SO3_Linear_e2former(
            self.sphere_channels, self.hidden_channels, lmax=lmax, bias=True
        )

        self.gate = Gate_s3(
            self.hidden_channels, lmax=lmax, act_scalars="silu", act_vector="sigmoid"
        )

        self.slinear_2 = SO3_Linear_e2former(
            self.hidden_channels, self.output_channels, lmax=lmax, bias=True
        )

    def forward(self, node_input, **kwargs):
        """
        irreps_in = o3.Irreps("128x0e+32x1e")
        func =  FeedForwardNetwork(
                irreps_in,
                irreps_in,
                proj_drop=0.1,
            )
        out = func(irreps_in.randn(10,20,-1))
        """
        node_output = self.slinear_1(node_input)
        node_output = self.gate(node_output)
        node_output = self.slinear_2(node_output)
        return node_output


class S2Activation(torch.nn.Module):
    """
    Assume we only have one resolution
    """

    def __init__(self, lmax, mmax):
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.act = torch.nn.SiLU()

    def forward(self, inputs, SO3_grid):
        to_grid_mat = SO3_grid[self.lmax][self.mmax].get_to_grid_mat(
            device=None
        )  # `device` is not used
        from_grid_mat = SO3_grid[self.lmax][self.mmax].get_from_grid_mat(device=None)
        x_grid = torch.einsum("bai, zic -> zbac", to_grid_mat, inputs)
        x_grid = self.act(x_grid)
        outputs = torch.einsum("bai, zbac -> zic", from_grid_mat, x_grid)
        return outputs


class SeparableS2Activation(torch.nn.Module):
    def __init__(self, lmax, mmax):
        super().__init__()

        self.lmax = lmax
        self.mmax = mmax

        self.scalar_act = torch.nn.SiLU()
        self.s2_act = S2Activation(self.lmax, self.mmax)

    def forward(self, input_scalars, input_tensors, SO3_grid):
        output_scalars = self.scalar_act(input_scalars)
        output_scalars = output_scalars.reshape(
            output_scalars.shape[0], 1, output_scalars.shape[-1]
        )
        output_tensors = self.s2_act(input_tensors, SO3_grid)
        outputs = torch.cat(
            (output_scalars, output_tensors.narrow(1, 1, output_tensors.shape[1] - 1)),
            dim=1,
        )
        return outputs


# follow eSCN
class FeedForwardNetwork_escn(torch.nn.Module):
    """
    FeedForwardNetwork: Perform feedforward network with S2 activation or gate activation

    Args:
        sphere_channels (int):      Number of spherical channels
        hidden_channels (int):      Number of hidden channels used during feedforward network
        output_channels (int):      Number of output channels

        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution

        SO3_grid (SO3_grid):        Class used to convert from grid the spherical harmonic representations

        activation (str):           Type of activation function
        use_gate_act (bool):        If `True`, use gate activation. Otherwise, use S2 activation
        use_grid_mlp (bool):        If `True`, use projecting to grids and performing MLPs.
        use_sep_s2_act (bool):      If `True`, use separable grid MLP when `use_grid_mlp` is True.
    """

    def __init__(
        self,
        sphere_channels,
        hidden_channels,
        output_channels,
        lmax,
        grid_resolution=18,
    ):
        super(FeedForwardNetwork_escn, self).__init__()
        self.sphere_channels = sphere_channels
        # self.hidden_channels = hidden_channels
        self.output_channels = output_channels

        self.so3_grid = torch.nn.ModuleList()
        self.lmax = lmax
        for l in range(lmax + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(lmax + 1):
                SO3_m_grid.append(
                    SO3_Grid(
                        l, m, resolution=grid_resolution  # , normalization="component"
                    )
                )
            self.so3_grid.append(SO3_m_grid)

        self.act = nn.SiLU()
        # Non-linear point-wise comvolution for the aggregated messages
        self.fc1_sphere = nn.Linear(
            2 * self.sphere_channels, self.sphere_channels, bias=False
        )

        self.fc2_sphere = nn.Linear(
            self.sphere_channels, self.sphere_channels, bias=False
        )

        self.fc3_sphere = nn.Linear(
            self.sphere_channels, self.sphere_channels, bias=False
        )

    def forward(self, node_irreps, nore_irreps_his, **kwargs):
        """_summary_
            model = FeedForwardNetwork_grid_nonlinear(
                    sphere_channels = 128,
                    hidden_channels = 128,
                    output_channels = 128,
                    lmax = 4,
                    grid_resolution = 18,
                )
            node_irreps = torch.randn(100,3,25,128)
            node_irreps_his = torch.randn(100,3,25,128)
            model(node_irreps,node_irreps_his).shape
        Args:
            node_irreps (_type_): _description_
            nore_irreps_his (_type_): _description_

        Returns:
            _type_: _description_
        """

        out_shape = node_irreps.shape[:-2]

        node_irreps = node_irreps.reshape(
            out_shape.numel(), (self.lmax + 1) ** 2, self.sphere_channels
        )
        nore_irreps_his = nore_irreps_his.reshape(
            out_shape.numel(), (self.lmax + 1) ** 2, self.sphere_channels
        )

        to_grid_mat = self.so3_grid[self.lmax][self.lmax].get_to_grid_mat(
            device=None
        )  # `device` is not used
        from_grid_mat = self.so3_grid[self.lmax][self.lmax].get_from_grid_mat(
            device=None
        )

        # Compute point-wise spherical non-linearity on aggregated messages
        # Project to grid
        x_grid = torch.einsum(
            "bai, zic -> zbac", to_grid_mat, node_irreps
        )  # input_embedding.to_grid(self.SO3_grid, lmax=max_lmax)
        x_grid_his = torch.einsum("bai, zic -> zbac", to_grid_mat, nore_irreps_his)
        x_grid = torch.cat([x_grid, x_grid_his], dim=3)

        # Perform point-wise convolution
        x_grid = self.act(self.fc1_sphere(x_grid))
        x_grid = self.act(self.fc2_sphere(x_grid))
        x_grid = self.fc3_sphere(x_grid)

        node_irreps = torch.einsum("bai, zbac -> zic", from_grid_mat, x_grid)
        return node_irreps.reshape(out_shape + (-1, self.output_channels))


class FeedForwardNetwork_s2(torch.nn.Module):
    """
    FeedForwardNetwork: Perform feedforward network with S2 activation or gate activation

    Args:
        sphere_channels (int):      Number of spherical channels
        hidden_channels (int):      Number of hidden channels used during feedforward network
        output_channels (int):      Number of output channels

        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution

        SO3_grid (SO3_grid):        Class used to convert from grid the spherical harmonic representations

        activation (str):           Type of activation function
        use_gate_act (bool):        If `True`, use gate activation. Otherwise, use S2 activation
        use_grid_mlp (bool):        If `True`, use projecting to grids and performing MLPs.
        use_sep_s2_act (bool):      If `True`, use separable grid MLP when `use_grid_mlp` is True.
    """

    def __init__(
        self,
        sphere_channels,
        hidden_channels,
        output_channels,
        lmax,
        mmax=2,
        grid_resolution=18,
        use_gate_act=False,  # [True, False] Switch between gate activation and S2 activation
        use_grid_mlp=True,  # [False, True] If `True`, use projecting to grids and performing MLPs for FFNs.
        use_sep_s2_act=True,  # Separable S2 activation. Used for ablation study.
        # activation="scaled_silu",
        # use_sep_s2_act=True,
    ):
        super(FeedForwardNetwork_s2, self).__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.sphere_channels_all = self.sphere_channels
        self.so3_grid = torch.nn.ModuleList()
        self.lmax = lmax
        self.max_lmax = self.lmax
        self.lmax_list = [lmax]
        for l in range(lmax + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(lmax + 1):
                SO3_m_grid.append(
                    SO3_Grid(
                        l, m, resolution=grid_resolution  # , normalization="component"
                    )
                )
            self.so3_grid.append(SO3_m_grid)

        self.use_gate_act = use_gate_act  # [True, False] Switch between gate activation and S2 activation
        self.use_grid_mlp = use_grid_mlp  # [False, True] If `True`, use projecting to grids and performing MLPs for FFNs.
        self.use_sep_s2_act = (
            use_sep_s2_act  # Separable S2 activation. Used for ablation study.
        )

        self.so3_linear_1 = SO3_LinearV2(
            self.sphere_channels_all, self.hidden_channels, lmax=self.lmax
        )
        if self.use_grid_mlp:
            if self.use_sep_s2_act:
                self.scalar_mlp = nn.Sequential(
                    nn.Linear(
                        self.sphere_channels_all,
                        self.hidden_channels,
                        bias=True,
                    ),
                    nn.SiLU(),
                )
            else:
                self.scalar_mlp = None
            self.grid_mlp = nn.Sequential(
                nn.Linear(self.hidden_channels, self.hidden_channels, bias=False),
                nn.SiLU(),
                nn.Linear(self.hidden_channels, self.hidden_channels, bias=False),
                nn.SiLU(),
                nn.Linear(self.hidden_channels, self.hidden_channels, bias=False),
            )
        else:
            if self.use_gate_act:
                self.gating_linear = torch.nn.Linear(
                    self.sphere_channels_all,
                    self.lmax * self.hidden_channels,
                )
                self.gate_act = GateActivation(
                    self.lmax, self.lmax, self.hidden_channels
                )
            else:
                if self.use_sep_s2_act:
                    self.gating_linear = torch.nn.Linear(
                        self.sphere_channels_all, self.hidden_channels
                    )
                    self.s2_act = SeparableS2Activation(self.lmax, self.lmax)
                else:
                    self.gating_linear = None
                    self.s2_act = S2Activation(self.lmax, self.lmax)
        self.so3_linear_2 = SO3_LinearV2(
            self.hidden_channels, self.output_channels, lmax=self.lmax
        )

    def forward(self, input_embedding):
        out_shape = input_embedding.shape[:-2]

        input_embedding = input_embedding.reshape(
            out_shape.numel(), (self.lmax + 1) ** 2, self.sphere_channels
        )
        #######################for memory saving
        x = SO3_Embedding(
            input_embedding.shape[0],
            self.lmax_list,
            self.sphere_channels,
            input_embedding.device,
            input_embedding.dtype,
        )
        x.embedding = input_embedding
        x = self._forward(x)

        return x.embedding.reshape(out_shape + (-1, self.output_channels))

    def _forward(self, input_embedding):
        gating_scalars = None
        if self.use_grid_mlp:
            if self.use_sep_s2_act:
                gating_scalars = self.scalar_mlp(
                    input_embedding.embedding.narrow(1, 0, 1)
                )
        else:
            if self.gating_linear is not None:
                gating_scalars = self.gating_linear(
                    input_embedding.embedding.narrow(1, 0, 1)
                )

        input_embedding = self.so3_linear_1(input_embedding)

        if self.use_grid_mlp:
            # Project to grid
            input_embedding_grid = input_embedding.to_grid(
                self.so3_grid, lmax=self.max_lmax
            )
            # Perform point-wise operations
            input_embedding_grid = self.grid_mlp(input_embedding_grid)
            # Project back to spherical harmonic coefficients
            input_embedding._from_grid(
                input_embedding_grid, self.so3_grid, lmax=self.max_lmax
            )

            if self.use_sep_s2_act:
                input_embedding.embedding = torch.cat(
                    (
                        gating_scalars,
                        input_embedding.embedding.narrow(
                            1, 1, input_embedding.embedding.shape[1] - 1
                        ),
                    ),
                    dim=1,
                )
        else:
            if self.use_gate_act:
                input_embedding.embedding = self.gate_act(
                    gating_scalars, input_embedding.embedding
                )
            else:
                if self.use_sep_s2_act:
                    input_embedding.embedding = self.s2_act(
                        gating_scalars,
                        input_embedding.embedding,
                        self.so3_grid,
                    )
                else:
                    input_embedding.embedding = self.s2_act(
                        input_embedding.embedding, self.so3_grid
                    )

        return self.so3_linear_2(input_embedding)


def fibonacci_sphere(samples=100):
    """
    Generate uniform grid points on a unit sphere using the Fibonacci lattice.

    Args:
        samples (int): Number of points.

    Returns:
        torch.Tensor: Shape (samples, 3), unit sphere points.
    """
    indices = torch.arange(0, samples, dtype=torch.float32) + 0.5
    phi = torch.acos(1 - 2 * indices / samples)  # Latitude
    theta = torch.pi * (1 + 5**0.5) * indices  # Longitude

    x = torch.cos(theta) * torch.sin(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(phi)

    return torch.stack([x, y, z], dim=-1)  # Shape (samples, 3)


def gaussian_function(r, gaussian_center, sigma=1, co=1):
    """
    Compute Gaussian function centered at gaussian_center.

    Args:
        r (torch.Tensor): Shape (N,sph_grid, 3), points in space.
        gaussian_center (torch.Tensor): Shape (N,topK or N,uniform point, 3), uniform point between atoms.
        sigma (float): (N,topK or N,uniform point,channel),  Standard deviation of Gaussian .
        coefficient (float): (N,topK or N,uniform point,channel),coefficient of Gaussian.

    Returns:
        torch.Tensor: Shape (N, M), Gaussian values for each point and midpoint.
    """
    N, sph_grid = r.shape[:2]
    gaussian_center = gaussian_center.unsqueeze(dim=3)
    if isinstance(sigma, torch.Tensor):
        sigma = torch.abs(sigma.unsqueeze(dim=3))
        co = co.unsqueeze(dim=3)

    dist = torch.norm(
        r.reshape(N, 1, 1, sph_grid, 3) - gaussian_center, dim=-1, keepdim=True
    )  # Compute Euclidean distances
    # the our put shape is (N,topK or N,sph_grid,uniform point,channel)
    return co * torch.exp(-(dist**2) * sigma)


# uniform_center_count means how many gaussian center between any atom pair.
# channels means in each gaussian, the function count or dimension or channel.


def cartesian_to_spherical(points):
    """
    Convert 3D Cartesian coordinates to spherical coordinates (r, theta, phi).

    Args:
        points (torch.Tensor): Shape (N, 3), 3D Cartesian coordinates.

    Returns:
        tuple: (theta, phi) in radians.
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = torch.sqrt(x**2 + y**2 + z**2)
    theta = torch.acos(z / r)  # Elevation angle
    phi = torch.atan2(y, x)  # Azimuthal angle
    return theta, phi


# Compute Gaussian function values
import torch


class Electron_Density_Descriptor(torch.nn.Module):
    def __init__(
        self,
        uniform_center_count=10,
        num_sphere_points=100,
        channel=8,
        lmax=3,
        output_channel=None,
        distribution="uniform",
    ):
        super().__init__()
        self.lmax = lmax
        self.uniform_center_count = uniform_center_count
        self.channel = channel
        self.output_channel = output_channel if output_channel is not None else channel
        self.proj = SO3_Linear_e2former(
            self.channel,
            self.output_channel,
            lmax=self.lmax,
        )
        self.gama = torch.nn.Parameter(
            torch.arange(0, uniform_center_count).reshape(1, 1, -1, 1)
            * 1.0
            / uniform_center_count,
            requires_grad=False,
        )
        # Example Usage
        self.sphere_grid = torch.nn.Parameter(
            fibonacci_sphere(num_sphere_points), requires_grad=False
        )

        print(self.gama.shape, self.sphere_grid.shape)  # Output: (100, 3)
        theta, phi = cartesian_to_spherical(self.sphere_grid)
        self.Y_lm_conj = []
        for l in range(lmax + 1):
            for m in range(-l, l + 1):
                # Compute spherical harmonics Y_{l,m} at each grid point
                Y_lm = sp.sph_harm(m, l, phi.numpy(), theta.numpy())  # Shape (N,)
                self.Y_lm_conj.append(
                    torch.tensor(Y_lm.conj(), dtype=torch.float32)
                )  # Take conjugate
        self.Y_lm_conj = torch.nn.Parameter(
            torch.stack(self.Y_lm_conj, dim=0), requires_grad=False
        )

    def forward(self, atom_positions, rji, sigma, co, neighbor_mask):
        # atom_positions = torch.randn((N, 3))  # Random atomic coordinates
        # rji = torch.randn((N,topk or N,1, 3))  # Random atomic coordinates
        # sigma = torch.randn(N,N,uniform_center_count,channel)
        # co = torch.randn(N,N,uniform_center_count,channel)
        output_shape = atom_positions.shape[:-1]
        atom_positions = atom_positions.reshape(-1, 3)
        N = atom_positions.shape[0]
        rji = rji.reshape(N, -1, 3)
        topK = rji.shape[1]

        sigma = torch.abs(sigma).reshape(
            N, topK, self.uniform_center_count, self.channel
        )
        co = co.reshape(N, topK, self.uniform_center_count, self.channel)
        gaussian_center = atom_positions.reshape(N, 1, 1, 3) + self.gama * rji.reshape(
            N, -1, 1, 3
        )
        gaussians = gaussian_function(
            atom_positions.reshape(-1, 1, 3) + self.sphere_grid.reshape(1, -1, 3),
            gaussian_center,
            sigma,
            co,
        )
        atom_center_sphgrid = torch.sum(
            gaussians * neighbor_mask.reshape(N, -1, 1, 1, 1), dim=(1, 2)
        )
        projection = (
            torch.sum(
                atom_center_sphgrid.unsqueeze(dim=1)
                * self.Y_lm_conj.reshape(1, (self.lmax + 1) ** 2, -1, 1),
                dim=2,
            )
            / self.Y_lm_conj.shape[1]
        )  # Normalize by N
        # print(prjection.shape)  # Output: ((lmax+1)^2,) → (16,)
        projection = self.proj(projection)
        return projection.reshape(
            output_shape + ((self.lmax + 1) ** 2, self.output_channel)
        )
