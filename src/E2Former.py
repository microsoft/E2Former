# -*- coding: utf-8 -*-
# Ensure E2formerESCAIP is properly imported or define
# from sfm.models.psm.psmmodel import *
from contextlib import nullcontext
from functools import partial

import torch
import torch.nn as nn
import torch_geometric
from e3nn import o3
from fairchem.core.common.registry import registry
from fairchem.core.common.transforms import RandomRotate
from fairchem.core.common.utils import conditional_grad
from fairchem.core.datasets import data_list_collater
from fairchem.core.models.base import GraphModelMixin, HeadInterface
from fairchem.core.models.scn.sampling import CalcSpherePoints
from torch_geometric.data import Data

# Ensure PSM components are properly imported
from .configs import EScAIPConfigs, init_configs
from .custom_types import GraphAttentionData
from .dit import E2DiTBlock
from .e2former import E2former, no_weight_decay
from .E2Former_configs import E2FormerConfigs
from .module_utils import (
    CellExpander,
    Electron_Density_Descriptor,
    GaussianLayer_Edgetype,
    polynomial,
)
from .modules import OutputLayer
from .utils.graph_utils import compilable_scatter, unpad_results
from .utils.nn_utils import init_linear_weights

_AVG_NUM_NODES = 77.81317


def process_batch_data(data, max_nodes=None):
    """
    Process raw batch data into padded batched format with masks.

    Args:
        data: Input data containing pos, cell, atomic_numbers, etc.
        max_nodes: Maximum number of nodes for padding. If None, uses maximum in batch.

    Returns:
        dict: Contains batched and padded data with masks
    """
    if len(data.pos.shape) == 3:
        if "masked_token_type" not in data:
            data["masked_token_type"] = data["token_id"]
        # which means the input is already B*max_nodes*3 format
        return data
    # otherwise, flatten_N * 3 format

    # Extract batch information
    batch_idx = data.batch
    num_graphs = data.ptr.size(0) - 1

    if max_nodes is None:
        max_nodes = max([data.ptr[i + 1] - data.ptr[i] for i in range(num_graphs)])

    # Initialize output tensors
    batched_pos = torch.zeros((num_graphs, max_nodes, 3), device=data.pos.device)
    batched_token_id = torch.zeros(
        (num_graphs, max_nodes), dtype=torch.long, device=data.atomic_numbers.device
    )
    masked_token_type = torch.zeros(
        (num_graphs, max_nodes), dtype=torch.long, device=data.atomic_numbers.device
    )
    padding_mask = torch.ones(
        (num_graphs, max_nodes), dtype=torch.bool, device=data.pos.device
    )
    # if "is_molecule" in data:

    # else:
    if "pbc" not in data:
        # pbc = torch.tensor([[1, 1, 1]]).repeat(num_graphs, 1).to(data.pos.device) # default for open catylst
        pbc = (
            torch.tensor([[1, 1, 0]]).repeat(num_graphs, 1).to(data.pos.device)
        )  # default for open catylst
    else:
        pbc = data.pbc

    # if "is_stable_periodic" not in data:
    #     is_stable_periodic = torch.zeros((num_graphs, 1), dtype=torch.bool, device=data.pos.device)
    # else:
    #     is_stable_periodic = data.is_stable_periodic
    # if "is_molecule" not in data:
    #     is_molecule = torch.zeros((num_graphs, 1), dtype=torch.bool, device=data.pos.device)
    # is_periodic = torch.ones((num_graphs, 1), dtype=torch.bool, device=data.pos.device)
    is_protein = torch.zeros(
        (num_graphs, max_nodes, 1), dtype=torch.bool, device=data.pos.device
    )
    num_atoms = torch.tensor(
        [data.ptr[i + 1] - data.ptr[i] for i in range(num_graphs)],
        dtype=torch.long,
        device=data.pos.device,
    )

    # Process each graph in the batch
    for i in range(num_graphs):
        start_idx = data.ptr[i]
        end_idx = data.ptr[i + 1]
        num_nodes = end_idx - start_idx

        # Fill in positions
        batched_pos[i, :num_nodes] = data.pos[start_idx:end_idx]

        # Fill in cell (assuming one cell per graph)

        # Fill in atomic numbers (token_ids)
        batched_token_id[i, :num_nodes] = data.atomic_numbers[start_idx:end_idx]
        if "masked_token_type" in data:
            masked_token_type[i, :num_nodes] = data.masked_token_type[start_idx:end_idx]
        else:
            masked_token_type[i, :num_nodes] = data.atomic_numbers[start_idx:end_idx]

        # Set mask (False indicates valid entries)
        padding_mask[i, :num_nodes] = False

    batched_data = {
        "pos": batched_pos,  # [num_graphs, max_nodes, 3]
        "cell": data.cell,  # [num_graphs, 3, 3]
        "token_id": batched_token_id,  # [num_graphs, max_nodes]
        "masked_token_type": masked_token_type,
        "padding_mask": padding_mask,  # [num_graphs, max_nodes]
        "pbc": pbc,  # [num_graphs, 3]
        "subset_name": None if "subset_name" not in data else data.subset_name,
        "forces_subset_name": None
        if "forces_subset_name" not in data
        else data.forces_subset_name,  # "is_stable_periodic": is_stable_periodic,  # [num_graphs, 1]
        # "is_molecule": is_molecule,
        # "is_periodic": is_periodic,
        "is_protein": is_protein,
        "position_ids": torch.arange(max_nodes)
        .unsqueeze(dim=0)
        .repeat(
            num_graphs, 1
        ),  # unused parameter: only for protein, id in sequence for pos cos and sin embed
        "num_atoms": num_atoms,  # [num_graphs]
        "node_batch": batch_idx,  # [num_nodes]
        "graph_padding_mask": padding_mask,  # [num_graphs, max_nodes]
    }

    batched_data = Data(**batched_data)

    return batched_data


@registry.register_model("PSM_ESCAIP_backbone")
class E2FormerBackbone(nn.Module, GraphModelMixin):
    """
    Physics Science Module backbone model integrated with EScAIP framework.

    This model combines the PSM architecture with EScAIP's configuration and processing
    pipeline, enabling it to work within the EScAIP framework while maintaining PSM's
    unique architectural features.
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()
        self.kwargs = kwargs
        # Load configs
        cfg = init_configs(E2FormerConfigs, kwargs)
        self.cfg = cfg
        self.global_cfg = cfg.global_cfg
        # self.molecular_graph_cfg = cfg.molecular_graph_cfg
        # self.gnn_cfg = cfg.gnn_cfg
        # self.reg_cfg = cfg.reg_cfg

        # Training configuration
        self.regress_forces = cfg.global_cfg.regress_forces

        # PSM specific configuration
        # TODO: Integrate PSM config with EScAIP config system
        self.psm_config = cfg.psm_config

        # Cell expansion for periodic boundary conditions
        self.cell_expander = CellExpander(
            self.kwargs["pbc_max_radius"],
            self.kwargs.get("expanded_token_cutoff", 512),  # deprecated
            self.kwargs["pbc_expanded_num_cell_per_direction"],
            self.kwargs["pbc_max_radius"],
        )

        # Token embedding layer
        self.embedding = nn.Embedding(256, cfg.encoder_embed_dim)
        self.boo_embedding = nn.Embedding(256 + 300, cfg.encoder_embed_dim)
        self.boo_embedding_linear = nn.Linear(cfg.encoder_embed_dim, 128)
        self.embedding_src = nn.Embedding(256, 128)
        self.embedding_tgt = nn.Embedding(256, 128)
        print("master config: ", cfg)
        self.dit_encoder_layers = None

        self.uniform_center_count = 5
        self.sph_grid_channel = 8
        self.linear_sigmaco = torch.nn.Sequential(
            nn.Linear(128 + 128 + 128, 128),
            nn.GELU(),
            nn.Linear(128, 2 * self.uniform_center_count * self.sph_grid_channel),
        )
        self.electron_density = Electron_Density_Descriptor(
            uniform_center_count=self.uniform_center_count,
            num_sphere_points=16,
            channel=self.sph_grid_channel,
            lmax=2,
            output_channel=cfg.encoder_embed_dim,
        )
        self.dit_gbf = GaussianLayer_Edgetype(K=128, edge_types=512)

        if cfg.encoder == "dit":
            self.dit_gbf = GaussianLayer_Edgetype(K=128, edge_types=512)
            self.gbf_proj = torch.nn.Sequential(
                nn.Linear(128, 128),
                nn.GELU(),
                nn.Linear(
                    128,
                    cfg.dit_config.num_encoder_layers
                    * cfg.dit_config.num_attention_heads,
                ),
            )
            self.edge_init = torch.nn.Sequential(
                nn.Linear(128 + 128 + 128, 128),
                nn.GELU(),
                nn.Linear(128, cfg.encoder_embed_dim),
            )
            self.dit_encoder_layers = nn.ModuleList()
            print("dit use config like follows: \n", cfg.dit_config)
            for nl in range(cfg.dit_config.num_encoder_layers):
                self.dit_encoder_layers.append(
                    E2DiTBlock(
                        embedding_dim=cfg.encoder_embed_dim, **vars(cfg.dit_config)
                    )
                )

        self.embed_proj = nn.Linear(
            cfg.encoder_embed_dim,
            o3.Irreps(cfg.backbone_config.irreps_node_embedding)[0][0],
        )
        self.fea_dim = o3.Irreps(cfg.backbone_config.irreps_node_embedding)[0][0]
        # Decoder selection and initialization
        print("e2former use config like follows: \n", cfg.backbone_config)
        self.decoder = E2former(**vars(cfg.backbone_config))

        # Enable high precision matrix multiplication if not using fp16
        if not self.global_cfg.use_fp16_backbone:
            torch.set_float32_matmul_precision("high")

        # Configure logging and compilation
        torch._logging.set_logs(recompiles=True)
        print("compiled:", self.global_cfg.use_compile)

        # Set up forward function with optional compilation
        self.forward_fn = (
            torch.compile(self.compiled_forward)
            if self.global_cfg.use_compile
            else self.compiled_forward
        )

    def BOO_feature(self, pos, expand_pos, local_attention_weight):
        B, N1 = pos.shape[:2]
        expand_pos.shape[1]
        dist = torch.norm(pos.unsqueeze(dim=2) - expand_pos.unsqueeze(dim=1), dim=-1)
        edge_vec = (pos.unsqueeze(dim=2) - expand_pos.unsqueeze(dim=1)) / (
            dist.unsqueeze(dim=-1) + 1e-5
        )
        angel = torch.sum(
            edge_vec * (local_attention_weight.unsqueeze(dim=-1) > 1e-6), dim=2
        )
        # print('before norm',angel[0],torch.sum(local_attention_weight>1e-6,dim =2)[0])
        # angel = 2*angel/(torch.sum(local_attention_weight>1e-6,dim =2).unsqueeze(dim = -1))
        # angel = angel
        angel = torch.sum(angel**2, dim=-1) - torch.sum(
            local_attention_weight > 1e-6, dim=2
        ).unsqueeze(dim=-1)
        # angel = self.boo_embedding(torch.sum(local_attention_weight>1e-6,dim =2))
        # print('after norm',angel[0],torch.max(angel),torch.min(angel))
        return angel

    def compiled_forward(
        self,
        batched_data,
        token_embedding=None,  # use dit /seq_encoder/** to prepare input for e2former
        time_step=None,
        clean_mask=None,
        aa_mask=None,
        # padding_mask=None,
        **kwargs,
    ):
        """
        Forward pass implementation that can be compiled with torch.compile.
        """
        # Enable gradient computation for forces if needed
        use_grad = (
            True  # self.global_cfg.regress_forces and not self.global_cfg.direct_force
        )
        batched_data["pos"].requires_grad_(use_grad)

        batched_data = process_batch_data(batched_data, None)
        if "padding_mask" not in batched_data:
            padding_mask = batched_data["token_id"].eq(0)
        else:
            padding_mask = batched_data["padding_mask"]

        batched_data["pos"] = torch.where(
            padding_mask.unsqueeze(dim=-1).repeat(1, 1, 3),
            999.0,
            batched_data["pos"].float(),
        )
        bsz, L = batched_data["pos"].shape[:2]
        # Create node type edge features
        # self._create_node_type_edge(batched_data)
        # print( batched_data["pos"].shape,torch.sum(padding_mask))
        # batched_data["pbc"] = batched_data["pbc"]*0
        # Use FP32 for certain operations if needed
        outcell_index = (
            torch.arange(L)
            .unsqueeze(dim=0)
            .repeat(bsz, 1)
            .to(batched_data["pos"].device)
        )
        batched_data["pos"]
        with torch.cuda.amp.autocast(enabled=self.global_cfg.use_fp16_backbone):
            # Handle periodic boundary conditions
            if (
                "pbc" in batched_data
                and batched_data["pbc"] is not None
                and torch.any(batched_data["pbc"])
            ):
                pbc_expand_batched = self.cell_expander.expand_includeself(
                    batched_data["pos"],
                    None,
                    batched_data["pbc"],
                    batched_data["num_atoms"],
                    batched_data["masked_token_type"],
                    batched_data["cell"],
                    neighbors_radius=(
                        self.kwargs["max_neighbors"],
                        self.kwargs["pbc_max_radius"],
                    ),
                    use_topK=True,  # must set true, thus when process expand neighbors, topK nearst neighbor dist could be used to filter.
                    use_local_attention=False,  # use_local_attention,
                    use_grad=use_grad,
                    padding_mask=padding_mask,
                )
                # dist: B*tgt_len*src_len
                pbc_expand_batched["expand_pos"]
                outcell_index = pbc_expand_batched["outcell_index"]
                pbc_expand_batched["expand_pos"][
                    pbc_expand_batched["expand_mask"]
                ] = 999  # set expand node pos padding to 9999
                dist = torch.norm(
                    batched_data["pos"].unsqueeze(dim=2)
                    - pbc_expand_batched["expand_pos"].unsqueeze(dim=1),
                    dim=-1,
                )
                src_len, tgt_len = (
                    pbc_expand_batched["expand_pos"].shape[1],
                    batched_data["pos"].shape[1],
                )
                node_type_edge = torch.stack(
                    [
                        batched_data["masked_token_type"]
                        .unsqueeze(dim=2)
                        .repeat(1, 1, src_len),
                        torch.gather(
                            batched_data["masked_token_type"],
                            dim=1,
                            index=pbc_expand_batched["outcell_index"],
                        )
                        .unsqueeze(dim=1)
                        .repeat(1, tgt_len, 1),
                    ],
                    dim=-1,
                )

                batched_data["node_type_edge"] = node_type_edge
                local_attention_weight = polynomial(dist, self.kwargs["pbc_max_radius"])
                local_attention_weight = torch.where(
                    padding_mask.unsqueeze(dim=-1), 1, local_attention_weight
                )
                # if torch.any(torch.sum(local_attention_weight==0,dim= -1)==(local_attention_weight.shape[2])):
                #     for i in range(len(local_attention_weight)):
                #         print(dist[i])
                #         print("idx and pos infor:",batched_data["idx"][i],batched_data["pos"][i],batched_data["cell"][i])
                #         print(torch.sum(padding_mask,dim = -1)[i],torch.sum(local_attention_weight==0,dim= -1)[i],local_attention_weight.shape[2])
                #     raise ValueError("pleas echeck you local attn weight")
                # # for some extreme case , you will suffer from no any neighbor within cut off problem, thus use this line to let it see all its neighbor with in and out cut off
                # local_attention_weight = torch.where((torch.sum(local_attention_weight==0,dim= -1)==local_attention_weight.shape[2]).unsqueeze(dim = -1),
                #                                      1/dist,local_attention_weight)

            else:
                pbc_expand_batched = None
                dist = torch.norm(
                    batched_data["pos"].unsqueeze(dim=2)
                    - batched_data["pos"].unsqueeze(dim=1),
                    dim=-1,
                )
                src_len = batched_data["pos"].shape[1]
                tgt_len = batched_data["pos"].shape[1]
                node_type_edge = torch.stack(
                    [
                        batched_data["masked_token_type"]
                        .unsqueeze(dim=2)
                        .repeat(1, 1, src_len),
                        batched_data["masked_token_type"]
                        .unsqueeze(dim=1)
                        .repeat(1, tgt_len, 1),
                    ],
                    dim=-1,
                )

                local_attention_weight = polynomial(dist, self.kwargs["max_radius"])
                local_attention_weight = torch.where(
                    padding_mask.unsqueeze(dim=-1), 1, local_attention_weight
                )
                # # for some extreme case , you will suffer from no any neighbor within cut off problem, thus use this line to let it see all its neighbor with in and out cut off
                # local_attention_weight = torch.where((torch.sum(local_attention_weight==0,dim= -1)==local_attention_weight.shape[2]).unsqueeze(dim = -1),
                #                         1/dist,local_attention_weight)
                batched_data["node_type_edge"] = node_type_edge
            # Generate embeddings
            token_id = batched_data["token_id"]
            padding_mask = token_id.eq(0)  # B x T x 1
            # boo_fea = self.BOO_feature(batched_data["pos"],expand_pos,local_attention_weight)
            # token_embedding_tgt = self.embedding_tgt(token_id)
            # token_embedding_src = self.embedding_src(torch.gather(token_id,dim = 1,index = outcell_index))
            # dist_rbf = self.dit_gbf(dist,batched_data["node_type_edge"])
            # sigma,co = torch.chunk(self.linear_sigmaco(torch.cat([
            #                 token_embedding_src.unsqueeze(dim = 1).repeat(1,tgt_len,1,1),
            #                 token_embedding_tgt.unsqueeze(dim = 2).repeat(1,1,src_len,1),
            #                 dist_rbf],dim = -1
            #                 )),dim = -1,chunks=2)

            # token_embedding = self.electron_density(
            #     batched_data["pos"],
            #     rji = -batched_data["pos"].unsqueeze(dim = 2)+pbc_expand_batched["expand_pos"].unsqueeze(dim = 1),
            #     sigma = sigma,
            #     co = co,
            #     neighbor_mask = local_attention_weight>1e-5)
            # token_embedding[:,:,0] = token_embedding[:,:,0] + self.embedding(token_id)
            if token_embedding is None:
                token_embedding = self.embedding(token_id)
                if self.dit_encoder_layers is not None:
                    dist_rbf = self.dit_gbf(dist, batched_data["node_type_edge"])
                    mixed_attn_bias = self.gbf_proj(dist_rbf)  # B*tgt*src*hidden
                    node_embedding_ef = token_embedding.clone()
                    # boo_fea = self.boo_embedding_linear(boo_fea)
                    token_embedding_tgt = self.embedding_tgt(token_id)  # +boo_fea
                    token_embedding_src = self.embedding_src(
                        torch.gather(token_id, dim=1, index=outcell_index)
                    )  # + \
                    # torch.gather(boo_fea,dim = 1,index = outcell_index.unsqueeze(dim = -1).repeat(1,1,boo_fea.shape[-1]))

                    edge_init = local_attention_weight.unsqueeze(
                        dim=-1
                    ) * self.edge_init(
                        torch.cat(
                            [
                                token_embedding_src.unsqueeze(dim=1).repeat(
                                    1, tgt_len, 1, 1
                                ),
                                token_embedding_tgt.unsqueeze(dim=2).repeat(
                                    1, 1, src_len, 1
                                ),
                                dist_rbf,
                            ],
                            dim=-1,
                        )
                    )
                    node_embedding_ef = torch.sum(edge_init, dim=2) / 23.39523
                    for j, dit_layer in enumerate(self.dit_encoder_layers):
                        node_embedding_ef = dit_layer(
                            node_embedding_ef,
                            node_embedding_ef,  # atom_embedding
                            padding_mask,
                            batched_data,
                            pbc_expand_batched=pbc_expand_batched,
                            mixed_attn_bias=mixed_attn_bias[
                                :,
                                :,
                                :,
                                self.cfg.dit_config.num_attention_heads
                                * j : self.cfg.dit_config.num_attention_heads
                                * (j + 1),
                            ]
                            .permute(0, 3, 1, 2)
                            .reshape(
                                bsz * self.cfg.dit_config.num_attention_heads,
                                tgt_len,
                                src_len,
                            ),
                            local_attention_weight=local_attention_weight,
                            # ifbackprop=True,
                        )
                        # print("layer: ",j,torch.mean(attn),torch.min(attn),torch.max(attn))
                    token_embedding = node_embedding_ef
                    # node_vec_features = node_vec_features * (node_embedding_ef.unsqueeze(dim = -2))
                token_embedding = self.embed_proj(token_embedding)

            # Forward through decoder
            (
                node_features,
                node_vec_features,
                node_irreps,
                node_irreps_his,
            ) = self.decoder(
                batched_data,
                token_embedding,
                None,
                padding_mask,
                pbc_expand_batched=pbc_expand_batched,
                return_node_irreps=True,
            )

        # flatten the node features from num batchs times num nodes to num nodes (to pyG style ), note that nodes are padded
        (
            node_features_flatten,
            node_vec_features_flatten,
            node_irreps_flatten,
            node_irreps_his_flatten,
        ) = self.flatten_node_features(
            node_features,
            node_vec_features,
            node_irreps,
            node_irreps_his,
            ~padding_mask,
        )

        return {
            "node_irrepsBxN": node_irreps,
            "node_featuresBxN": node_features,
            "node_vec_featuresBxN": node_vec_features,
            "data": batched_data,
            "node_irreps": node_irreps_flatten,
            "node_features": node_features_flatten,
            "node_vec_features": node_vec_features_flatten,
            "node_irreps_his": node_irreps_his_flatten,
        }

    def flatten_node_features(
        self,
        node_features,
        node_vec_features,
        node_irreps,
        node_irreps_his,
        padding_mask,
    ):
        flat_node_irreps = node_irreps.view(
            -1, node_irreps.size(-2), node_irreps.size(-1)
        )
        flat_node_irreps_his = node_irreps_his.view(
            -1, node_irreps_his.size(-2), node_irreps_his.size(-1)
        )
        flat_node_features = node_features.view(-1, node_features.size(-1))  # [B*N, D]
        flat_node_vec_features = node_vec_features.view(
            -1, node_vec_features.size(-2), node_vec_features.size(-1)
        )  # [B*N, D_vec]
        flat_mask = padding_mask.view(-1)  # [B*N]
        # Use the mask to filter out padded nodes
        valid_node_irreps = flat_node_irreps[flat_mask]  # [sum(valid_nodes), D]
        valid_node_irreps_his = flat_node_irreps_his[flat_mask]  # [sum(valid_nodes), D]
        valid_node_features = flat_node_features[flat_mask]  # [sum(valid_nodes), D]
        valid_node_vec_features = flat_node_vec_features[flat_mask]
        return (
            valid_node_features,
            valid_node_vec_features,
            valid_node_irreps,
            valid_node_irreps_his,
        )

    @conditional_grad(torch.enable_grad())
    def forward(
        self,
        data: torch_geometric.data.Batch,
        node_embedding=None,
        # aa_mask=None,
        # padding_mask=None,
        *args,
        **kwargs,
    ):
        """
        Main forward pass of the model.
        """
        # PSM handles preprocessing internally
        return self.forward_fn(data, token_embedding=node_embedding)

    @torch.jit.ignore
    def no_weight_decay(self):
        """
        Returns parameters that should not use weight decay.
        """
        return no_weight_decay(self)
        # return no_weight_decay

    def test_equivariant(self, original_data):
        # assume batch size is 1
        assert (
            original_data.batch.max() == 0
        ), "batch size must be 1 for test_equivariant"
        self.eval()  # this is very important
        data_2 = original_data.clone().cpu()
        transform = RandomRotate([-180, 180], [0, 1, 2])
        data_2, matrix, inv_matrix = transform(data_2)
        data_2 = data_2.to(original_data.pos.device)
        data_list = data_list_collater([original_data, data_2])
        data_list.ptr = torch.tensor(
            [
                0,
                original_data.pos.size(0),
                original_data.pos.size(0) + data_2.pos.size(0),
            ],
            device=original_data.pos.device,
        )
        results = self.compiled_forward(data_list)
        combined_node_features = results["node_features"]
        # split the node features into two parts
        node_features_1 = combined_node_features[: original_data.pos.size(0)]
        node_features_2 = combined_node_features[original_data.pos.size(0) :]

        assert node_features_1.allclose(
            node_features_2, rtol=1e-2, atol=1e-2
        ), "node features are not equivariant"

        node_vec_features_1 = results["node_vec_features"][: original_data.pos.size(0)]
        node_vec_features_2 = results["node_vec_features"][original_data.pos.size(0) :]
        # rotate the node vec features
        node_vec_features_1 = torch.einsum(
            "bsd, sj -> bjd", node_vec_features_1, matrix.to(node_vec_features_1.device)
        )
        assert node_vec_features_1.allclose(
            node_vec_features_2, rtol=1e-2, atol=1e-2
        ), "node vec features are not equivariant"


class E2FormerHeadBase(nn.Module, HeadInterface):
    def __init__(self, backbone: E2FormerBackbone):
        super().__init__()
        self.global_cfg = backbone.global_cfg
        # self.molecular_graph_cfg = backbone.molecular_graph_cfg
        # self.gnn_cfg = backbone.gnn_cfg
        # self.reg_cfg = backbone.reg_cfg

    def post_init(self, gain=1.0):
        # init weights
        self.apply(partial(init_linear_weights, gain=gain))

        self.forward_fn = (
            torch.compile(self.compiled_forward)
            if self.global_cfg.use_compile
            else self.compiled_forward
        )

    def forward(self, data, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {}

    @torch.jit.ignore
    def no_weight_decay(self):
        return no_weight_decay(self)


@registry.register_model("E2Former_energy_head")
class E2FormerEnergyHead(E2FormerHeadBase):
    def __init__(self, backbone: E2FormerBackbone):
        super().__init__(backbone)
        self.energy_layer = OutputLayer(
            global_cfg=self.global_cfg,
            gnn_cfg=self.gnn_cfg,
            reg_cfg=self.reg_cfg,
            output_type="Scalar",
        )

        self.post_init(gain=0.01)

    def compiled_forward(self, node_features, data):
        energy_output = self.energy_layer(node_features)

        # the following not compatible with torch.compile (grpah break)
        # energy_output = torch_scatter.scatter(energy_output, node_batch, dim=0, reduce="sum")
        # the shape of energy_output is [num_nodes, 1]
        # the shape of data.node_batch is [num_nodes]
        # the shape of data.graph_padding_mask is [num_graphs, num_nodes]
        # the shape of data.node_batch is [num_nodes]
        # dim size is the number of graphs
        number_of_graphs = data.node_batch.max() + 1
        energy_output = compilable_scatter(
            src=energy_output,
            index=data.node_batch,
            dim_size=number_of_graphs,
            dim=0,
            reduce="sum",
        )
        return energy_output

    def forward(self, data, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        energy_output = self.forward_fn(
            node_features=emb["node_features"],
            data=emb["data"],
        )
        return {"energy": energy_output}


@registry.register_model("E2Former_direct_force_head")
class E2FormerDirectForceHead(E2FormerHeadBase):
    def __init__(self, backbone: E2FormerBackbone):
        super().__init__(backbone)
        self.linear = nn.Linear(backbone.decoder.scalar_dim, 1, bias=False)

        self.force_magnitude_layer = OutputLayer(
            global_cfg=self.global_cfg,
            gnn_cfg=self.gnn_cfg,
            reg_cfg=self.reg_cfg,
            output_type="Scalar",
        )

        self.post_init()

    def compiled_forward(
        self, node_features, node_vec_features, data: GraphAttentionData
    ):
        # get force direction from node vector features
        force_direction = self.linear(node_vec_features).squeeze(-1)  # (num_nodes, 3)

        # get output force
        return force_direction

    def forward(self, data, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        force_output = self.forward_fn(
            node_features=emb["node_features"],
            node_vec_features=emb["node_vec_features"],
            data=emb["data"],
        )

        return {"forces": force_output}


@registry.register_model("E2Former_easy_energy_head")
class E2FormerEasyEnergyHead(E2FormerHeadBase):
    def __init__(self, backbone: E2FormerBackbone):
        super().__init__(backbone)
        self.linear = nn.Linear(backbone.decoder.scalar_dim, 1, bias=False)

        self.post_init(gain=0.01)

    def compiled_forward(self, node_features, data):
        energy_output = self.linear(node_features)

        # the following not compatible with torch.compile (grpah break)
        # energy_output = torch_scatter.scatter(energy_output, node_batch, dim=0, reduce="sum")
        # the shape of energy_output is [num_nodes, 1]
        # the shape of data.node_batch is [num_nodes]
        # the shape of data.graph_padding_mask is [num_graphs, num_nodes]
        # the shape of data.node_batch is [num_nodes]
        # dim size is the number of graphs
        number_of_graphs = data.node_batch.max() + 1
        energy_output = (
            compilable_scatter(
                src=energy_output,
                index=data.node_batch,
                dim_size=number_of_graphs,
                dim=0,
                reduce="sum",
            )
            / _AVG_NUM_NODES
        )
        return energy_output

    def forward(self, data, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        energy_output = self.forward_fn(
            node_features=emb["node_features"],
            data=emb["data"],
        )
        return {"energy": energy_output}


@registry.register_model("E2Former_easy_force_head")
class E2FormerEasyForceHead(E2FormerHeadBase):
    def __init__(self, backbone: E2FormerBackbone):
        super().__init__(backbone)
        self.linear = nn.Linear(backbone.decoder.scalar_dim, 1, bias=False)

        self.post_init()

    def compiled_forward(
        self, node_features, node_vec_features, data: GraphAttentionData
    ):
        # get force direction from node vector features
        force_direction = self.linear(node_vec_features).squeeze(-1)  # (num_nodes, 3)

        # get output force
        return force_direction

    def forward(self, data, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        force_output = self.forward_fn(
            node_features=emb["node_features"],
            node_vec_features=emb["node_vec_features"],
            data=emb["data"],
        )

        return {"forces": force_output}


@registry.register_model("E2Former_grad_energy_force_head")
class E2FormerGradEnergyHead(E2FormerHeadBase):
    def __init__(self, backbone: E2FormerBackbone):
        super().__init__(backbone)
        self.linear = nn.Linear(backbone.decoder.scalar_dim, 1, bias=False)

        self.post_init(gain=0.01)

    def compiled_forward(self, node_features, data):
        energy_output = self.linear(node_features)

        # the following not compatible with torch.compile (grpah break)
        # energy_output = torch_scatter.scatter(energy_output, node_batch, dim=0, reduce="sum")
        # the shape of energy_output is [num_nodes, 1]
        # the shape of data.node_batch is [num_nodes]
        # the shape of data.graph_padding_mask is [num_graphs, num_nodes]
        # the shape of data.node_batch is [num_nodes]
        # dim size is the number of graphs
        number_of_graphs = data.node_batch.max() + 1
        energy_output = (
            compilable_scatter(
                src=energy_output,
                index=data.node_batch,
                dim_size=number_of_graphs,
                dim=0,
                reduce="sum",
            )
            / _AVG_NUM_NODES
        )
        return energy_output

    def forward(self, data, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        energy_output = self.forward_fn(
            node_features=emb["node_features"],
            data=emb["data"],
        )

        forces_output = (
            -1
            * torch.autograd.grad(
                energy_output.sum(), data.pos, create_graph=self.training
            )[0]
        )

        return {"energy": energy_output, "forces": forces_output}


class eSCN_EnergyBlock(E2FormerHeadBase):
    """
    Energy Block: Output block computing the energy

    Args:
        num_channels (int):         Number of channels
        num_sphere_samples (int):   Number of samples used to approximate the integral on the sphere
        act (function):             Non-linear activation function
    """

    def __init__(self, backbone: E2FormerBackbone) -> None:
        super().__init__(backbone)

        self.num_channels = backbone.decoder.scalar_dim
        self.num_sphere_samples = 18  # num_sphere_samples
        lmax = backbone.decoder.lmax
        # Create a roughly evenly distributed point sampling of the sphere for the output blocks
        self.sphere_points = nn.Parameter(
            CalcSpherePoints(self.num_sphere_samples), requires_grad=False
        )

        self.sphharm_weights = nn.Parameter(
            o3.spherical_harmonics(
                torch.arange(0, lmax + 1).tolist(),
                self.sphere_points,
                False,
            ),
            requires_grad=False,
        )
        self.act = nn.SiLU()

        self.fc1 = nn.Linear(self.num_channels, self.num_channels)
        self.fc2 = nn.Linear(self.num_channels, self.num_channels)
        self.fc3 = nn.Linear(self.num_channels, 1, bias=False)

    def compiled_forward(self, x) -> torch.Tensor:
        x_pt = torch.einsum(
            "abc, pb->apc",
            x,
            self.sphharm_weights,
        ).contiguous()
        # x_pt are the values of the channels sampled at different points on the sphere
        x_pt = self.act(self.fc1(x_pt))
        x_pt = self.act(self.fc2(x_pt))
        x_pt = self.fc3(x_pt)
        x_pt = x_pt.view(-1, self.num_sphere_samples, 1)
        return torch.sum(x_pt, dim=1) / self.num_sphere_samples

    def forward(self, data, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        energy_output = self.compiled_forward(
            x=emb["node_irreps"],
        )
        number_of_graphs = emb["data"].node_batch.max() + 1
        energy_output = (
            compilable_scatter(
                src=energy_output,  # follow eSCN for numerical precision
                index=emb["data"].node_batch,
                dim_size=number_of_graphs,
                dim=0,
                reduce="sum",
            )
            / _AVG_NUM_NODES
        )
        return {"energy": energy_output}


class eSCN_ForceBlock(E2FormerHeadBase):
    """
    Force Block: Output block computing the per atom forces

    Args:
        num_channels (int):         Number of channels
        num_sphere_samples (int):   Number of samples used to approximate the integral on the sphere
        act (function):             Non-linear activation function
    """

    def __init__(self, backbone: E2FormerBackbone) -> None:
        super().__init__(backbone)

        self.num_channels = backbone.decoder.scalar_dim
        self.num_sphere_samples = 18  # num_sphere_samples
        lmax = backbone.decoder.lmax

        self.fc1 = nn.Linear(self.num_channels, self.num_channels)
        self.fc2 = nn.Linear(self.num_channels, self.num_channels)
        self.fc3 = nn.Linear(self.num_channels, 1, bias=False)

        # Create a roughly evenly distributed point sampling of the sphere for the output blocks
        self.sphere_points = nn.Parameter(
            CalcSpherePoints(self.num_sphere_samples), requires_grad=False
        )

        self.sphharm_weights = nn.Parameter(
            o3.spherical_harmonics(
                torch.arange(0, lmax + 1).tolist(),
                self.sphere_points,
                False,
            ),
            requires_grad=False,
        )
        self.act = nn.SiLU()

    def compiled_forward(self, x) -> torch.Tensor:
        x_pt = torch.einsum(
            "abc, pb->apc",
            x,
            self.sphharm_weights,
        ).contiguous()
        # x_pt are the values of the channels sampled at different points on the sphere
        x_pt = self.act(self.fc1(x_pt))
        x_pt = self.act(self.fc2(x_pt))
        x_pt = self.fc3(x_pt)
        x_pt = x_pt.view(-1, self.num_sphere_samples, 1)
        forces = x_pt * self.sphere_points.view(1, self.num_sphere_samples, 3)
        return torch.sum(forces, dim=1) / self.num_sphere_samples

    def forward(self, data, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        force_output = self.compiled_forward(
            x=emb["node_irreps"],
        )  # *0.001

        return {"forces": force_output}


class eSCN_ScalarBlock(E2FormerHeadBase):
    """
    Energy Block: Output block computing the energy

    Args:
        num_channels (int):         Number of channels
        num_sphere_samples (int):   Number of samples used to approximate the integral on the sphere
        act (function):             Non-linear activation function
    """

    def __init__(self, backbone: E2FormerBackbone, output_channels: int) -> None:
        super().__init__(backbone)

        self.num_channels = backbone.decoder.scalar_dim
        self.num_sphere_samples = 18  # num_sphere_samples
        lmax = backbone.decoder.lmax
        # Create a roughly evenly distributed point sampling of the sphere for the output blocks
        self.sphere_points = nn.Parameter(
            CalcSpherePoints(self.num_sphere_samples), requires_grad=False
        )

        self.sphharm_weights = nn.Parameter(
            o3.spherical_harmonics(
                torch.arange(0, lmax + 1).tolist(),
                self.sphere_points,
                False,
            ),
            requires_grad=False,
        )
        self.act = nn.SiLU()

        self.fc1 = nn.Linear(self.num_channels, self.num_channels)
        self.fc2 = nn.Linear(self.num_channels, self.num_channels)
        self.fc3 = nn.Linear(self.num_channels, output_channels, bias=False)
        self.output_channels = output_channels

    def compiled_forward(self, x) -> torch.Tensor:
        x_pt = torch.einsum(
            "abc, pb->apc",
            x,
            self.sphharm_weights,
        ).contiguous()
        # x_pt are the values of the channels sampled at different points on the sphere
        x_pt = self.act(self.fc1(x_pt))
        x_pt = self.act(self.fc2(x_pt))
        x_pt = self.fc3(x_pt)
        x_pt = x_pt.view(-1, self.num_sphere_samples, self.output_channels)
        return torch.sum(x_pt, dim=1) / _AVG_NUM_NODES / self.num_sphere_samples


class eSCN_ForceBlockV2(E2FormerHeadBase):
    """
    Force Block: Output block computing the per atom forces

    Args:
        backbone (E2FormerBackbone): Backbone network
        num_layers (int): Number of force/scalar layer pairs
        num_channels (int): Number of channels
        num_sphere_samples (int): Number of samples used to approximate the integral on the sphere
        act (function): Non-linear activation function
    """

    def __init__(self, backbone: E2FormerBackbone, num_layers: int = 8) -> None:
        super().__init__(backbone)
        self.num_layers = num_layers
        self.force_layers = nn.ModuleList(
            [eSCN_ForceBlock(backbone) for _ in range(num_layers)]
        )
        self.scalar_layers = nn.ModuleList(
            [eSCN_EnergyBlock(backbone) for _ in range(num_layers)]
        )

    def compiled_forward(self, x) -> torch.Tensor:
        output = 0
        for i in range(self.num_layers):
            force = self.force_layers[i].compiled_forward(x)
            scalar = self.scalar_layers[i].compiled_forward(x)
            output = output + force * scalar
        return output

    def forward(self, data, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        force_output = self.compiled_forward(
            x=emb["node_irreps"],
        )
        return {"forces": force_output}


class MoE_ForceBlockV2(E2FormerHeadBase):
    """
    Mixture of Experts Force Block: Uses multiple expert networks with a gating mechanism

    Args:
        backbone (E2FormerBackbone): Backbone network
        num_experts (int): Number of expert networks
        k (int): Number of experts to route to (top-k routing)
        noise_factor (float): Noise factor for load balancing
    """

    def __init__(
        self,
        backbone: E2FormerBackbone,
        num_experts: int = 8,
        k: int = 2,
        noise_factor: float = 1e-2,
    ) -> None:
        super().__init__(backbone)

        # Expert networks
        self.force_experts = nn.ModuleList(
            [eSCN_ForceBlock(backbone) for _ in range(num_experts)]
        )
        self.scalar_experts = nn.ModuleList(
            [eSCN_EnergyBlock(backbone) for _ in range(num_experts)]
        )

        # Gating network
        self.gate = eSCN_ScalarBlock(backbone, num_experts)

        self.num_experts = num_experts
        self.k = k
        self.noise_factor = noise_factor

    def _compute_routing_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute routing weights with load balancing noise.

        Args:
            x (torch.Tensor): Input tensor of shape [..., num_channels]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Routing weights and expert indices
        """
        # Compute gates using the gate network
        gates = self.gate.compiled_forward(x)  # [num_nodes, num_experts]

        # Add noise for load balancing during training
        if self.training:
            noise = torch.randn_like(gates) * self.noise_factor
            gates = gates + noise

        # Get top-k experts
        top_k_gates, top_k_indices = torch.topk(gates, self.k, dim=-1)
        top_k_gates = torch.softmax(top_k_gates, dim=-1)

        return top_k_gates, top_k_indices

    def compiled_forward(self, x) -> torch.Tensor:
        # x is the node features of shape numnodes \times num_spatial_features \times num_channels
        # Compute routing weights
        routing_weights, expert_indices = self._compute_routing_weights(x)

        # Initialize output tensor
        combined_output = torch.zeros(x.shape[0], 3, device=x.device, dtype=x.dtype)

        # Route to top-k experts
        for k in range(self.k):
            # Get current expert indices and weights
            current_experts = expert_indices[:, k]  # [batch_size]
            current_weights = routing_weights[:, k]  # [batch_size]

            # Compute outputs for each expert
            for expert_idx in range(self.num_experts):
                # Find which batch items use this expert
                batch_mask = current_experts == expert_idx
                if not batch_mask.any():
                    continue

                # Get relevant batch items
                expert_input = x[batch_mask]
                expert_weight = current_weights[batch_mask]

                # Compute expert outputs
                force_output = self.force_experts[expert_idx].compiled_forward(
                    expert_input
                )
                scalar_output = self.scalar_experts[expert_idx].compiled_forward(
                    expert_input
                )
                expert_output = force_output * scalar_output

                # Weight the outputs and add to result
                weighted_output = expert_output * expert_weight.view(-1, 1)
                combined_output[batch_mask] += weighted_output

        return combined_output

    def forward(self, data, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        force_output = self.compiled_forward(
            x=emb["node_irreps"],
        )

        # Compute and log expert usage statistics if training
        if self.training and hasattr(self, "logger"):
            with torch.no_grad():
                _, expert_indices = self._compute_routing_weights(emb["node_irreps"])
                expert_counts = torch.bincount(
                    expert_indices.view(-1), minlength=self.num_experts
                )
                expert_usage = expert_counts.float() / expert_counts.sum()
                self.logger.log(
                    {
                        "expert_usage_std": expert_usage.std().item(),
                        "expert_usage_max": expert_usage.max().item(),
                        "expert_usage_min": expert_usage.min().item(),
                    }
                )

        return {"forces": force_output}
