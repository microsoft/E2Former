# -*- coding: utf-8 -*-
import math
import warnings
from typing import Dict, Optional

import e3nn
import torch
from e3nn import o3
from e3nn.util.jit import compile_mode
from fairchem.core.models.equiformer_v2.equiformer_v2_deprecated import EquiformerV2
from fairchem.core.models.equiformer_v2.so3 import SO3_LinearV2
from fairchem.core.models.escn.escn import SO2Block
from fairchem.core.models.escn.so3 import SO3_Embedding, SO3_Rotation

# for bessel radial basis
from fairchem.core.models.gemnet.layers.radial_basis import RadialBasis
from torch import logical_not, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

# from fairchem.core.models.escn.so3 import SO3_Grid


# from torch.nn.attention import SDPBackend, sdpa_kernel


_AVG_DEGREE = 23.395238876342773

from .maceblocks import EquivariantProductBasisBlock, reshape_irrepstoe3nn

# # QM9
# _MAX_ATOM_TYPE = 20
# # Statistics of QM9 with cutoff radius = 5
# _AVG_NUM_NODES = 18.03065905448718
from .module_utils import (  # ,\; EquivariantInstanceNorm,EquivariantGraphNorm; EquivariantRMSNormArraySphericalHarmonicsV2,; GaussianLayer,; irreps2gate,; sort_irreps_even_first,
    Activation,
    DropPath_BL,
    Electron_Density_Descriptor,
    EquivariantDropout,
    EquivariantLayerNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonicsV2,
    EquivariantRMSNormArraySphericalHarmonicsV2_BL,
    FeedForwardNetwork_escn,
    FeedForwardNetwork_s2,
    FeedForwardNetwork_s3,
    Gate,
    GaussianLayer_Edgetype,
    GaussianRadialBasisLayer,
    GaussianSmearing,
    Irreps2Scalar,
    IrrepsLinear,
    Learn_PolynomialDistance,
    RadialFunction,
    RadialProfile,
    SmoothLeakyReLU,
    SO3_Grid,
    SO3_Linear2Scalar_e2former,
    SO3_Linear_e2former,
    TensorProductRescale,
    get_normalization_layer,
    polynomial,
)
from .so2 import _init_edge_rot_mat
from .wigner6j.tensor_product import (
    DepthWiseTensorProduct_reducesameorder,
    E2TensorProductArbitraryOrder,
    Simple_TensorProduct_oTchannel,
)

_AVG_DEGREE = 23.395238876342773  # IS2RE: 100k, max_radius = 5, max_neighbors = 100

_USE_BIAS = True


def no_weight_decay(
    self,
):
    no_wd_list = []
    named_parameters_list = [name for name, _ in self.named_parameters()]
    for module_name, module in self.named_modules():
        if (
            isinstance(module, RadialBasis)
            # or isinstance(module, RMS_Norm_SH)
            # or isinstance(module, EquivariantGraphNorm) # TODO
            or isinstance(
                module,
                (
                    torch.nn.Linear,
                    SO3_LinearV2,
                    SO3_Linear_e2former,
                    torch.nn.LayerNorm,
                    EquivariantLayerNormArraySphericalHarmonics,
                    EquivariantRMSNormArraySphericalHarmonics,
                    EquivariantRMSNormArraySphericalHarmonicsV2,
                    EquivariantRMSNormArraySphericalHarmonicsV2_BL,
                    GaussianRadialBasisLayer,
                    Simple_TensorProduct_oTchannel,
                ),
            )
        ):
            for parameter_name, _ in module.named_parameters():
                if (
                    isinstance(
                        module,
                        torch.nn.Linear,
                        SO3_LinearV2,
                        SO3_Linear_e2former,
                        Simple_TensorProduct_oTchannel,
                    )
                    and "weight" in parameter_name
                ):
                    continue
                global_parameter_name = module_name + "." + parameter_name
                assert global_parameter_name in named_parameters_list
                no_wd_list.append(global_parameter_name)

    return set(no_wd_list)


def irreps_times(irreps, factor):
    out = [(int(mul * factor), ir) for mul, ir in irreps if mul > 0]
    return e3nn.o3.Irreps(out)


class Body2_interaction(torch.nn.Module):
    def __init__(
        self,
        irreps_x,
    ):
        """
        Use separable FCTP for spatial convolution.
        [...,irreps_x] tp [...,irreps_y] - > [..., irreps_out]

        fc_neurons is not needed in e2former
        """

        super().__init__()

        self.irreps_node_input = (
            o3.Irreps(irreps_x) if isinstance(irreps_x, str) else irreps_x
        )
        self.input_dim = self.irreps_node_input[0][0]
        self.output_dim = self.irreps_node_input[0][0]
        self.lmax = len(self.irreps_node_input) - 1
        self.irreps_small_fc_left = SO3_Linear_e2former(
            self.irreps_node_input[0][0],
            self.irreps_node_input[0][0],
            lmax=self.lmax,
        )

        self.irreps_small_fc_right = SO3_Linear_e2former(
            self.irreps_node_input[0][0],
            self.irreps_node_input[0][0],
            lmax=self.lmax,
        )
        self.body2_tp = Simple_TensorProduct_oTchannel(
            irreps_in1=self.irreps_node_input,
            irreps_in2=self.irreps_node_input,
            irreps_out=self.irreps_node_input,
            instructions=[
                (2, 2, 0, "uuu", False),
                (1, 2, 1, "uuu", False),
                (1, 1, 2, "uuu", False),
                (2, 2, 3, "uuu", False),
                (2, 2, 4, "uuu", False),
            ][:3],
        )

        self.linear_final = SO3_Linear_e2former(
            self.irreps_node_input[0][0],
            self.irreps_node_input[0][0],
            lmax=self.lmax,
        )

    def forward(self, irreps_x, *args, **kwargs):
        """
        x: [..., irreps]

        irreps_in = o3.Irreps("256x0e+64x1e+32x2e")
        sep_tp = SeparableFCTP(irreps_in,"1x1e",irreps_in,fc_neurons=None,
                            use_activation=False,norm_layer=None,
                            internal_weights=True)
        out = sep_tp(irreps_in.randn(100,10,-1),torch.randn(100,10,3),None)
        print(out.shape)
        """
        shape = irreps_x.shape[:-2]
        N = irreps_x.shape[:-2].numel()
        irreps_x = irreps_x.reshape((N, (self.lmax + 1) ** 2, self.input_dim))

        out = self.body2_tp(
            self.irreps_small_fc_left(irreps_x),
            self.irreps_small_fc_right(irreps_x),
            None,
        )
        # print(out.shape,self.dtp.irreps_out)
        out = self.linear_final(out)

        return out.reshape(list(shape) + [(self.lmax + 1) ** 2, self.output_dim])


class Body3_interaction_MACE(torch.nn.Module):
    def __init__(
        self,
        irreps_x,
        fc_neurons=None,
        use_activation=False,
        norm_layer="graph",
        internal_weights=False,
    ):
        """
        Use separable FCTP for spatial convolution.
        [...,irreps_x] tp [...,irreps_y] - > [..., irreps_out]

        fc_neurons is not needed in e2former
        """

        super().__init__()
        self.irreps_node_input = (
            o3.Irreps(irreps_x) if isinstance(irreps_x, str) else irreps_x
        )
        self.irreps_small = self.irreps_node_input
        self.irreps_small_fc = SO3_Linear_e2former(
            self.irreps_node_input[0][0],
            self.irreps_small[0][0],
            lmax=len(self.irreps_node_input) - 1,
        )

        self.reshape_func = reshape_irrepstoe3nn(self.irreps_small)

        self.num_elements = 300
        # dtp input shape is *xdim*(sumL)
        self.dtp = EquivariantProductBasisBlock(
            node_feats_irreps=self.irreps_small,
            target_irreps=self.irreps_small,
            correlation=3,
            num_elements=self.num_elements,
            use_sc=False,
        )
        # dtp out shape is *x(128x0e_128x1e_128x2e) same like e3nn

        self.lin = SO3_Linear_e2former(
            self.irreps_small[0][0],
            self.irreps_node_input[0][0],
            lmax=len(self.irreps_node_input) - 1,
        )

    def forward(self, irreps_x, atomic_numbers, **kwargs):
        """
        x: [..., irreps]
        import torch
        from torch import logical_not, nn
        from sfm.models.psm.equivariant.e2former import Body3_interaction_MACE

        B,N = 4,128

        pos= torch.randn(B,N,3)
        irreps_x = torch.randn(B,N,9,128)
        atomic_number = torch.randint(0,100,(B,N))

        model =  Body3_interaction_MACE(
                '128x0e+128x1e+128x2e',
                fc_neurons=None,
                use_activation=False,
                norm_layer=None,
                internal_weights=False,
            )
        model(irreps_x,atomic_number)
        """
        # atomic_numbers = atomic_numbers[:,:irreps_x.shape[1]]
        shape = irreps_x.shape[:-2]
        N = irreps_x.shape[:-2].numel()
        irreps_x = irreps_x.reshape((N,) + irreps_x.shape[-2:])
        irreps_x_small = self.irreps_small_fc(irreps_x)
        irreps_x_small = irreps_x_small.permute(
            0, 2, 1
        )  # self.reshape_func(irreps_x_small)
        # print(irreps_x_small.shape,torch.max(atomic_numbers),torch.min(atomic_numbers))
        irreps_x_small = self.dtp(
            irreps_x_small,
            sc=None,
            node_attrs=torch.nn.functional.one_hot(
                atomic_numbers.reshape(-1).long(), num_classes=self.num_elements
            ).float(),
        )

        irreps_x_small = self.reshape_func.back2orderTmul(irreps_x_small)
        # print(out.shape,self.dtp.irreps_out)
        irreps_x_small = self.lin(irreps_x_small)

        # warnings.warn("please check your correctness of MACE")
        return irreps_x_small.reshape(shape + (-1, self.irreps_node_input[0][0]))


class E2AttentionArbOrder_sparse(torch.nn.Module):
    """
    Use IrrepsLinear with external weights W(|r_i|)

    """

    def __init__(
        self,
        irreps_node_input="256x0e+64x1e+32x2e",
        attn_weight_input_dim: int = 32,  # e.g. rbf(|r_ij|) or relative pos in sequence
        num_attn_heads: int = 8,
        attn_scalar_head: int = 32,
        irreps_head="32x0e+8x1e+4x2e",
        alpha_drop=0.1,
        rescale_degree=False,
        nonlinear_message=False,
        proj_drop=0.1,
        tp_type="v1",
        attn_type="first-order",  ## second-order
        add_rope=True,
        layer_id=0,
        irreps_origin="1x0e+1x1e+1x2e",
        neighbor_weight=None,
        atom_type_cnt=256,
        norm_layer="identity",
        **kwargs,
    ):
        super().__init__()
        self.atom_type_cnt = atom_type_cnt
        self.neighbor_weight = neighbor_weight
        self.irreps_node_input = (
            e3nn.o3.Irreps(irreps_node_input)
            if isinstance(irreps_node_input, str)
            else irreps_node_input
        )
        self.scalar_dim = self.irreps_node_input[0][0]  # scalar_dim x 0e
        self.num_attn_heads = num_attn_heads
        self.attn_scalar_head = attn_scalar_head
        self.attn_weight_input_dim = attn_weight_input_dim
        irreps_head = (
            e3nn.o3.Irreps(irreps_head) if isinstance(irreps_head, str) else irreps_head
        )

        self.irreps_head = irreps_head
        # irreps_node_output,  attention will not change the input shape/embeding length
        self.irreps_node_output = self.irreps_node_input
        self.lmax = self.irreps_node_input[-1][1][0]
        # new params
        self.attn_type = attn_type
        self.tp_type = tp_type.split("+")[0]
        self.use_smooth_softmax = "use_smooth_softmax" in tp_type

        self.node_embed_dim = 128

        self.source_embedding = nn.Embedding(self.atom_type_cnt, self.node_embed_dim)
        self.target_embedding = nn.Embedding(self.atom_type_cnt, self.node_embed_dim)
        nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
        nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)

        self.alpha_act = SmoothLeakyReLU(0.2)
        # *3 means, rij, src_embedding, tgt_embedding
        self.edge_channel_list = [
            attn_weight_input_dim + self.node_embed_dim * 2,
            min(128, attn_weight_input_dim // 2),
            min(128, attn_weight_input_dim // 2),
        ]
        self.alpha_dropout = torch.nn.Dropout(alpha_drop)

        if self.tp_type.startswith("dot_alpha"):
            self.dot_linear = SO3_Linear_e2former(
                self.irreps_node_input[0][0],
                attn_weight_input_dim,
                lmax=self.lmax,
            )
            self.alpha_norm = torch.nn.LayerNorm(self.attn_scalar_head)
            self.alpha_dot = torch.nn.Parameter(
                torch.randn(self.num_attn_heads, self.attn_scalar_head)
            )
            std = 1.0 / math.sqrt(self.attn_scalar_head)
            torch.nn.init.uniform_(self.alpha_dot, -std, std)
            self.alpha_dropout = torch.nn.Dropout(alpha_drop)

            self.fc_m0 = nn.Linear(
                2 * self.attn_weight_input_dim * (self.lmax + 1),
                self.num_attn_heads * self.attn_scalar_head,
            )
            self.rad_func_m0 = RadialFunction(
                self.edge_channel_list
                + [2 * self.attn_weight_input_dim * (self.lmax + 1)]
            )
        elif self.tp_type.startswith("dot_alpha_small"):
            self.dot_linear = SO3_Linear_e2former(
                self.irreps_node_input[0][0],
                attn_weight_input_dim // 8,
                lmax=self.lmax,
            )
            self.alpha_norm = torch.nn.LayerNorm(self.attn_scalar_head)
            self.alpha_dot = torch.nn.Parameter(
                torch.randn(self.num_attn_heads, self.attn_scalar_head)
            )
            std = 1.0 / math.sqrt(self.attn_scalar_head)
            torch.nn.init.uniform_(self.alpha_dot, -std, std)
            self.alpha_dropout = torch.nn.Dropout(alpha_drop)

            self.fc_m0 = nn.Linear(
                2 * self.attn_weight_input_dim // 8 * (self.lmax + 1),
                self.num_attn_heads * self.attn_scalar_head,
            )
            self.rad_func_m0 = RadialFunction(
                self.edge_channel_list
                + [2 * self.attn_weight_input_dim // 8 * (self.lmax + 1)]
            )

        elif self.tp_type == "QK_alpha":
            self.query_linear = SO3_Linear2Scalar_e2former(
                self.irreps_node_input[0][0],
                num_attn_heads * self.attn_scalar_head,
                lmax=self.lmax,
            )
            self.key_linear = SO3_Linear2Scalar_e2former(
                self.irreps_node_input[0][0],
                num_attn_heads * self.attn_scalar_head,
                lmax=self.lmax,
            )
            self.alpha_dot = torch.nn.Parameter(
                torch.randn(self.num_attn_heads, self.attn_scalar_head)
            )
            std = 1.0 / math.sqrt(self.attn_scalar_head)
            torch.nn.init.uniform_(self.alpha_dot, -std, std)

            self.query_linear = SO3_Linear2Scalar_e2former(
                self.irreps_node_input[0][0],
                num_attn_heads * self.attn_scalar_head,
                lmax=self.lmax,
            )
            self.key_linear = SO3_Linear2Scalar_e2former(
                self.irreps_node_input[0][0],
                num_attn_heads * self.attn_scalar_head,
                lmax=self.lmax,
            )

            self.fc_easy = RadialFunction(
                self.edge_channel_list + [self.num_attn_heads]
            )

        else:
            raise ValueError("please check your tp_type")

        # self.gbf = GaussianLayer(self.attn_weight_input_dim)  # default output_dim = 128
        self.pos_embedding_proj = nn.Linear(
            self.attn_weight_input_dim, self.scalar_dim * 2
        )
        self.node_scalar_proj = nn.Linear(self.scalar_dim, self.scalar_dim * 2)

        self.poly1 = Learn_PolynomialDistance(degree=1)
        self.poly2 = Learn_PolynomialDistance(degree=2)

        if self.attn_type == "zero-order":
            self.rad_func_intputhead = RadialFunction(
                self.edge_channel_list + [self.irreps_node_input[0][0]]
            )

            self.proj_zero = SO3_Linear_e2former(
                self.irreps_node_input[0][0],
                self.irreps_node_output[0][0],
                lmax=self.lmax,
            )

        elif self.attn_type == "first-order":
            self.rad_func_intputhead = RadialFunction(
                self.edge_channel_list + [self.irreps_node_input[0][0]]
            )

            self.first_order_tp = E2TensorProductArbitraryOrder(
                self.irreps_node_input,
                (self.irreps_head * num_attn_heads).sort().irreps.simplify(),
                order=1,
                head=self.irreps_node_input[0][0],
                learnable_weight=True,
                connection_mode="uvw",
            )

            self.proj_first = SO3_Linear_e2former(
                num_attn_heads * self.irreps_head[0][0],
                self.irreps_node_output[0][0],
                lmax=self.lmax,
            )

        elif self.attn_type == "second-order":
            self.rad_func_intputhead = RadialFunction(
                self.edge_channel_list + [self.irreps_node_input[0][0] // 2]
            )
            self.proj_value = SO3_Linear_e2former(
                self.irreps_node_input[0][0],
                self.irreps_node_input[0][0] // 2,
                lmax=self.lmax,
            )
            # self.second_order_tp = E2TensorProductArbitraryOrder_woequal(self.irreps_node_input,
            self.second_order_tp = E2TensorProductArbitraryOrder(
                irreps_times(self.irreps_node_input, 0.5),
                (self.irreps_head * num_attn_heads).sort().irreps.simplify(),
                order=2,
                head=self.irreps_node_input[0][0] // 2,
                learnable_weight=True,
                connection_mode="uvw",
            )
            self.proj_sec = SO3_Linear_e2former(
                num_attn_heads * self.irreps_head[0][0],
                self.irreps_node_output[0][0],
                lmax=self.lmax,
            )

        elif self.attn_type == "all-order":
            self.rad_func_intputhead_zero = RadialFunction(
                self.edge_channel_list + [self.irreps_node_input[0][0]]
            )
            self.proj_zero = SO3_Linear_e2former(
                self.irreps_node_input[0][0],
                self.irreps_node_output[0][0],
                lmax=self.lmax,
            )

            self.rad_func_intputhead_fir = RadialFunction(
                self.edge_channel_list + [self.irreps_node_input[0][0] // 2]
            )
            self.proj_value_fir = SO3_Linear_e2former(
                self.irreps_node_input[0][0],
                self.irreps_node_input[0][0] // 2,
                lmax=self.lmax,
            )
            self.first_order_tp = E2TensorProductArbitraryOrder(
                irreps_times(self.irreps_node_input, 0.5),
                (self.irreps_head * num_attn_heads).sort().irreps.simplify(),
                order=1,
                head=self.irreps_node_input[0][0] // 2,
                learnable_weight=True,
                connection_mode="uvw",
            )

            self.proj_first = SO3_Linear_e2former(
                num_attn_heads * self.irreps_head[0][0],
                self.irreps_node_output[0][0],
                lmax=self.lmax,
            )

            self.rad_func_intputhead_sec = RadialFunction(
                self.edge_channel_list + [self.irreps_node_input[0][0] // 4]
            )
            self.proj_value_sec = SO3_Linear_e2former(
                self.irreps_node_input[0][0],
                self.irreps_node_input[0][0] // 4,
                lmax=self.lmax,
            )
            # self.second_order_tp = E2TensorProductArbitraryOrder_woequal(self.irreps_node_input,
            self.second_order_tp = E2TensorProductArbitraryOrder(
                irreps_times(self.irreps_node_input, 0.25),
                (self.irreps_head * num_attn_heads).sort().irreps.simplify(),
                order=2,
                head=self.irreps_node_input[0][0] // 4,
                learnable_weight=True,
                connection_mode="uvw",
            )
            self.proj_sec = SO3_Linear_e2former(
                num_attn_heads * self.irreps_head[0][0],
                self.irreps_node_output[0][0],
                lmax=self.lmax,
            )
        self.norm_1 = get_normalization_layer(
            norm_layer,
            lmax=self.lmax,
            num_channels=self.irreps_node_output[0][0],
        )
        self.norm_2 = get_normalization_layer(
            norm_layer,
            lmax=self.lmax,
            num_channels=self.irreps_node_output[0][0],
        )

        # self.edge_updater = RadialFunction([attn_weight_input_dim,
        #                                     min(128,attn_weight_input_dim//2),
        #                                     attn_weight_input_dim])

    @staticmethod
    def vector_rejection(vec, d_ij):
        r"""Computes the component of :obj:`vec` orthogonal to :obj:`d_ij`.

        Args:
            vec (torch.Tensor): The input vector.
            d_ij (torch.Tensor): The reference vector.
        """
        vec_proj = (vec * d_ij).sum(dim=-2, keepdim=True)
        return vec - vec_proj * d_ij

    def forward(
        self,
        node_pos,
        node_irreps_input,
        edge_dis,
        edge_vec,
        attn_weight,  # e.g. rbf(|r_ij|) or relative pos in sequence
        atomic_numbers,
        poly_dist=None,
        attn_mask=None,  # non-adj is True
        batch=None,
        batched_data=None,
        **kwargs,
    ):
        """
        from sfm.models.psm.equivariant.e2former import *

        irreps_in = o3.Irreps("256x0e+256x1e+256x2e")
        B,L = 4,20
        node_pos = torch.randn(B,L,3)
        node_dis = torch.sqrt(torch.sum((node_pos.view(B,L,1,3)-node_pos.view(B,1,L,3))**2,dim = -1))
        attn_scalar_head = 32
        func = E2AttentionSecondOrder(
            irreps_node_input = irreps_in,
            attn_weight_input_dim = 32, # e.g. rbf(|r_ij|) or relative pos in sequence
            num_attn_heads = 8,
            attn_scalar_head = attn_scalar_head,
            irreps_head = "32x0e+32x1e+32x2e",
            alpha_drop=0.1,
            tp_type='easy_alpha'
        )
        out = func(node_pos,
            torch.randn(B,L,9,256),
            node_dis,
            torch.randn(B,L,L,3),
            torch.randn(B,L,L,attn_scalar_head),
            atomic_numbers = torch.randint(0,19,(B,L)),
            attn_mask = torch.randn(B,L,L,1)>0)
        print(out.shape)
        """
        # attn_weight f(|r_ij|): B*L*L*heads
        # edge_dis: B*L*L
        # node_input: B*L*irreps (irreps e.g. 128x0e+64x1e+32x2e)
        # ir2scalar: B*L*irreps -> N*L*hidden (hidden e.g. head*32)
        # attn_weight: B*L*L*rbf_dim
        f_N1, _, hidden = node_irreps_input.shape
        # f_N2 =
        topK = attn_weight.shape[1]

        f_sparse_idx_node = batched_data["f_sparse_idx_node"]

        attn_weight = attn_weight.masked_fill(attn_mask, 0)
        edge_feature = attn_weight.sum(dim=1)  # B*L*-1
        # print(node_irreps_input.shape,torch.ones_like(node_irreps_input[:,:,:1,:1]).shape,self.tp_weight(node_scalars).shape)
        # value = self.value_tp(node_irreps_input,torch.ones_like(node_irreps_input[:,:1,:1]),self.tp_weight(node_scalars))
        value = node_irreps_input  # *node_scalars[:,None]

        src_node = self.source_embedding(atomic_numbers)
        tgt_node = self.target_embedding(atomic_numbers)

        # sparse_indices = batched_data["batchANDneighbor_indices"]
        # topK = sparse_indices[0].shape[2]
        x_edge = torch.cat(
            [
                attn_weight,
                tgt_node.reshape(f_N1, 1, -1).repeat(1, topK, 1),
                src_node[f_sparse_idx_node],
            ],
            dim=-1,
        )

        x_0_extra = []
        if self.tp_type == "dot_alpha" or self.tp_type == "dot_alpha_small":
            node_irreps_input_dot = self.dot_linear(node_irreps_input)
            for l in range(self.lmax + 1):
                rij_l = e3nn.o3.spherical_harmonics(
                    l, edge_vec, normalize=True
                ).unsqueeze(
                    dim=-1
                )  # B*N*N*2l+1*1

                node_l = node_irreps_input_dot[
                    :, l**2 : (l + 1) ** 2
                ]  # B*N*2l+1*hidden
                # print(rij_l.shape,node_l.shape,node_irreps_input.shape)
                x_0_extra.append(torch.sum(rij_l * node_l.unsqueeze(dim=1), dim=-2))
                x_0_extra.append(torch.sum(rij_l * node_l[f_sparse_idx_node], dim=-2))

        if self.tp_type == "QK_alpha":
            ## QK alpha
            query = self.query_linear(node_irreps_input).reshape(
                f_N1, self.num_attn_heads, -1
            )
            key = self.key_linear(node_irreps_input)

            key = key.reshape(f_N1, self.num_attn_heads, -1)

            key = key[f_sparse_idx_node]

            alpha = self.alpha_act(
                self.fc_easy(x_edge)
                * torch.sum(query.unsqueeze(dim=1) * key, dim=3)
                / math.sqrt(query.shape[-1])
            )

        elif self.tp_type.startswith("dot_alpha"):
            edge_m0 = self.rad_func_m0(x_edge)

            x_0_alpha = self.fc_m0(torch.cat(x_0_extra, dim=-1) * edge_m0)
            x_0_alpha = x_0_alpha.reshape(
                f_N1, -1, self.num_attn_heads, self.attn_scalar_head
            )
            x_0_alpha = self.alpha_norm(x_0_alpha)
            x_0_alpha = self.alpha_act(x_0_alpha)
            alpha = torch.einsum("qeik, ik -> qei", x_0_alpha, self.alpha_dot)

        # key = key[sparse_indices[0],sparse_indices[1]]
        # alpha = self.alpha_act(
        #     self.fc_easy(x_edge) * torch.einsum("bihd,bijhd->bijh",query,key)/math.sqrt(query.shape[-1]))

        if self.use_smooth_softmax:
            alpha = alpha.to(torch.float64)
            poly_dist = poly_dist.to(alpha.dtype)
            alpha = alpha - alpha.max(dim=1, keepdim=True).values
            alpha = torch.exp(alpha) * poly_dist.unsqueeze(-1)
            alpha = alpha.masked_fill(attn_mask, 0)
            alpha = (alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-3)).to(
                torch.float32
            )
        else:
            alpha = alpha.masked_fill(attn_mask, -1e6)
            #######################biggest bug here!
            # alpha = torch.nn.functional.softmax(alpha, 2)
            alpha = torch.nn.functional.softmax(alpha, 1)
            alpha = alpha.masked_fill(attn_mask, 0)

        # alpha = alpha*x_0_extra_wosm
        if self.alpha_dropout is not None:
            alpha = self.alpha_dropout(alpha)

        alpha_org = alpha

        edge_dis = edge_dis
        if self.attn_type != "all-order":
            inputhead = self.rad_func_intputhead(x_edge)
            alpha = alpha.reshape(f_N1, -1, self.num_attn_heads, 1) * inputhead.reshape(
                alpha.shape[:2] + (self.num_attn_heads, -1)
            )
            alpha = alpha.reshape(alpha.shape[:2] + (-1,))
            # batched_data.update(
            #     {'f_sparse_idx_node':f_sparse_idx_node,
            #     'f_sparse_idx_expnode':f_sparse_idx_expnode,
            #     'f_exp_node_pos':f_exp_node_pos,
            #     'f_outcell_index':f_outcell_index
            #     }
            if self.attn_type == "zero-order":
                node_output = self.proj_zero(
                    torch.sum(
                        alpha.unsqueeze(dim=2)
                        * value[batched_data["f_sparse_idx_node"]],
                        dim=1,
                    )
                )

            if self.attn_type == "first-order":
                node_output = self.proj_first(
                    self.first_order_tp(
                        node_pos,
                        batched_data["f_exp_node_pos"],
                        None,
                        value[batched_data["f_outcell_index"]],
                        alpha / (edge_dis.unsqueeze(dim=-1) + 1e-8),
                        batched_data["f_sparse_idx_expnode"],
                        batched_data=batched_data,
                    )
                )

            if self.attn_type == "second-order":
                value = self.proj_value(value)
                node_output = self.proj_sec(
                    self.second_order_tp(
                        node_pos,
                        batched_data["f_exp_node_pos"],
                        None,
                        value[batched_data["f_outcell_index"]],
                        alpha / (edge_dis.unsqueeze(dim=-1) ** 2 + 1e-8),
                        batched_data["f_sparse_idx_expnode"],
                        batched_data=batched_data,
                    )
                )

        if self.attn_type == "all-order":
            node_gate = torch.nn.functional.sigmoid(
                self.pos_embedding_proj(edge_feature)
                + self.node_scalar_proj(node_irreps_input[:, 0, :])
            )

            inputhead_zero = self.rad_func_intputhead_zero(x_edge)
            alpha_zero = alpha_org.reshape(
                f_N1, -1, self.num_attn_heads, 1
            ) * inputhead_zero.reshape(alpha_org.shape[:2] + (self.num_attn_heads, -1))
            alpha_zero = alpha_zero.reshape(alpha_zero.shape[:2] + (-1,))
            node_output_zero = self.proj_zero(
                torch.sum(
                    alpha_zero.unsqueeze(dim=2)
                    * value[batched_data["f_sparse_idx_node"]],
                    dim=1,
                )
            )

            # value = self.norm_1(value+node_output_zero)
            inputhead_fir = self.rad_func_intputhead_fir(x_edge)
            alpha_fir = alpha_org.reshape(
                f_N1, -1, self.num_attn_heads, 1
            ) * inputhead_fir.reshape(alpha_org.shape[:2] + (self.num_attn_heads, -1))
            alpha_fir = alpha_fir.reshape(alpha_fir.shape[:2] + (-1,))
            node_output_fir = self.proj_first(
                self.first_order_tp(
                    node_pos,
                    batched_data["f_exp_node_pos"],
                    None,
                    self.proj_value_fir(value)[batched_data["f_outcell_index"]],
                    alpha_fir / (edge_dis.unsqueeze(dim=-1) + 1e-8),
                    batched_data["f_sparse_idx_expnode"],
                    batched_data=batched_data,
                )
            )

            # value = self.norm_2(value+node_output_fir)

            inputhead_sec = self.rad_func_intputhead_sec(x_edge)
            alpha_sec = alpha_org.reshape(
                f_N1, -1, self.num_attn_heads, 1
            ) * inputhead_sec.reshape(alpha_org.shape[:2] + (self.num_attn_heads, -1))
            alpha_sec = alpha_sec.reshape(alpha_sec.shape[:2] + (-1,))
            node_output_sec = self.proj_sec(
                self.second_order_tp(
                    node_pos,
                    batched_data["f_exp_node_pos"],
                    None,
                    self.proj_value_sec(value)[batched_data["f_outcell_index"]],
                    alpha_sec / (edge_dis.unsqueeze(dim=-1) ** 2 + 1e-8),
                    batched_data["f_sparse_idx_expnode"],
                    batched_data=batched_data,
                )
            )
            node_output = (
                node_output_zero * node_gate[:, None, : self.scalar_dim]
                + node_output_fir * node_gate[:, None, self.scalar_dim :]
                + node_output_sec * (1 - node_gate[:, None, self.scalar_dim :])
            )
            # node_output = node_output_own*(1-node_gate[:,None,:self.scalar_dim])+
        # updated_attn_weight = attn_weight + node_irreps_input[:,:1].reshape(f_N1, 1, -1).repeat(1,topK,1) + node_irreps_input[:,0][f_sparse_idx_node]
        # updated_attn_weight = attn_weight + self.edge_updater(updated_attn_weight)
        return node_output, attn_weight


# class FeedForwardNetwork(torch.nn.Module):
#     """
#     Use two (FCTP + Gate)
#     """

#     def __init__(
#         self,
#         irreps_node_input,
#         irreps_node_output,
#     ):
#         super().__init__()
#         self.irreps_node_input = (
#             o3.Irreps(irreps_node_input)
#             if isinstance(irreps_node_input, str)
#             else irreps_node_input
#         )
#         self.irreps_node_output = (
#             o3.Irreps(irreps_node_output)
#             if isinstance(irreps_node_output, str)
#             else irreps_node_output
#         )

#         irreps_scalars, irreps_gates, irreps_gated = irreps2gate(self.irreps_node_input)
#         self.irreps_mlp_mid = (
#             (self.irreps_node_input + irreps_gates).sort().irreps.simplify()
#         )

#         # warnings.warn(f"FeedForwardNetwork:GATE is tooooooo ugly, please refine this later")

#         self.slinear_1 = IrrepsLinear(
#             self.irreps_node_input, self.irreps_mlp_mid, bias=True, act=None
#         )
#         # TODO: to be optimized.  Toooooo ugly
#         if irreps_gated.num_irreps == 0:
#             self.gate = Activation(self.irreps_mlp_mid, acts=[torch.nn.functional.silu])
#         else:
#             self.gate = Gate(
#                 irreps_scalars,
#                 [torch.nn.functional.silu for _, ir in irreps_scalars],  # scalar
#                 irreps_gates,
#                 [torch.sigmoid for _, ir in irreps_gates],  # gates (scalars)
#                 irreps_gated,  # gated tensors
#             )

#         self.slinear_2 = IrrepsLinear(
#             self.irreps_node_input, self.irreps_node_output, bias=True, act=None
#         )
#         # self.proj_drop = None
#         # if proj_drop != 0.0:
#         #     self.proj_drop = EquivariantDropout(
#         #         self.irreps_node_output, drop_prob=proj_drop
#         #     )

#     def forward(self, node_input, **kwargs):
#         """
#         irreps_in = o3.Irreps("128x0e+32x1e")
#         func =  FeedForwardNetwork(
#                 irreps_in,
#                 irreps_in,
#                 proj_drop=0.1,
#             )
#         out = func(irreps_in.randn(10,20,-1))
#         """
#         node_output = self.slinear_1(node_input)
#         node_output = self.gate(node_output)
#         node_output = self.slinear_2(node_output)
#         if self.proj_drop is not None:
#             node_output = self.proj_drop(node_output)
#         return node_output


@compile_mode("script")
class TransBlock(torch.nn.Module):
    """
    1. Layer Norm 1 -> E2Attention -> Layer Norm 2 -> FeedForwardNetwork
    2. Use pre-norm architecture
    """

    def __init__(
        self,
        irreps_node_input,
        irreps_node_output,
        attn_weight_input_dim,  # e.g. rbf(|r_ij|) or relative pos in sequence
        num_attn_heads,
        attn_scalar_head,
        irreps_head,
        rescale_degree=False,
        nonlinear_message=False,
        alpha_drop=0.1,
        proj_drop=0,
        drop_path_rate=0.1,
        norm_layer="rms_norm_sh",  # used for norm 1 and norm2
        layer_id=0,
        attn_type=0,
        tp_type="v2",
        ffn_type="default",
        add_rope=True,
        sparse_attn=False,
        max_radius=15,
    ):
        super().__init__()
        self.irreps_node_input = (
            o3.Irreps(irreps_node_input)
            if isinstance(irreps_node_input, str)
            else irreps_node_input
        )
        self.irreps_node_output = (
            o3.Irreps(irreps_node_output)
            if isinstance(irreps_node_output, str)
            else irreps_node_output
        )

        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message
        # self.norm_1 = get_norm_layer(norm_layer)(self.irreps_node_input) # this is e2former norm
        self.lmax = irreps_node_input[-1][1][0]
        self.norm_1 = get_normalization_layer(
            norm_layer, lmax=self.lmax, num_channels=irreps_node_input[0][0]
        )

        self.layer_id = layer_id
        func = None

        if "+" in attn_type:
            attn_type = attn_type.split("+")
            if layer_id >= int(attn_type[0][-1]) + int(attn_type[1][-1]):
                raise ValueError("sorry you attn type is bigger than layer id")
            if layer_id < int(attn_type[0][-1]):
                attn_type = attn_type[0][:-1]
            else:
                attn_type = attn_type[1][:-1]

        self.attn_type = attn_type

        if isinstance(attn_type, str) and attn_type.endswith("order"):
            func = E2AttentionArbOrder_sparse

        elif isinstance(attn_type, str) and attn_type.startswith("escn"):
            func = MessageBlock_escn
        elif isinstance(attn_type, str) and attn_type.startswith("eqv2"):
            func = MessageBlock_eqv2
        else:
            raise ValueError(
                f" sorry, the attn type is not support, please check {attn_type}"
            )
        self.attn_weight_input_dim = attn_weight_input_dim
        self.ga = func(
            irreps_node_input,
            attn_weight_input_dim,  # e.g. rbf(|r_ij|) or relative pos in sequence
            num_attn_heads,
            attn_scalar_head,
            irreps_head,
            rescale_degree=rescale_degree,
            nonlinear_message=nonlinear_message,
            alpha_drop=alpha_drop,
            proj_drop=proj_drop,
            layer_id=layer_id,
            attn_type=attn_type,
            tp_type=tp_type,
            add_rope=add_rope,
            sparse_attn=sparse_attn,
            max_radius=max_radius,
            norm_layer=norm_layer,
        )

        self.drop_path = None  # nn.Identity()
        if drop_path_rate > 0.0:
            self.drop_path = DropPath_BL(drop_path_rate)

        self.proj_drop_func = nn.Identity()
        if proj_drop > 0.0:
            self.proj_drop_func = EquivariantDropout(
                self.irreps_node_input[0][0], self.lmax, proj_drop
            )

        self.so2_ffn = None
        self.SO3_grid = None
        ffn_type = ffn_type.split("+")
        # if "gate_op" in ffn_type:
        #     self.ffn = FeedForwardNetwork_GateOP(
        #         irreps_node_input=self.irreps_node_input,  # self.concat_norm_output.irreps_out,
        #         irreps_node_output=self.irreps_node_input,
        #         proj_drop=proj_drop,
        #     )
        #     raise ValueError("not support yet")
        # el
        self.ffn_s2 = None
        if ("eqv2ffn" in ffn_type) or ("default" in ffn_type) or ("s2" in ffn_type):
            # self.norm_2 = get_norm_layer(norm_layer)(self.irreps_node_input) # this is e2former norm
            self.norm_s2 = get_normalization_layer(
                norm_layer, lmax=self.lmax, num_channels=irreps_node_input[0][0]
            )

            self.ffn_s2 = FeedForwardNetwork_s2(
                self.irreps_node_input[0][0],
                self.irreps_node_input[0][0],
                self.irreps_node_input[0][0],
                lmax=self.lmax,
                grid_resolution=18,
                use_grid_mlp=False,  # notice in eqv2, default is True
            )
        else:
            self.ffn_s2 = None
            self.norm_s2 = None

        if "s3" in ffn_type:
            self.norm_s3 = get_normalization_layer(
                norm_layer, lmax=self.lmax, num_channels=irreps_node_input[0][0]
            )
            self.ffn_s3 = FeedForwardNetwork_s3(
                self.irreps_node_input[0][0],
                self.irreps_node_input[0][0],
                self.irreps_node_input[0][0],
                lmax=self.lmax,
            )
        else:
            self.ffn_s3 = None
            self.norm_s3 = None

        if self.ffn_s3 is not None and self.ffn_s2 is not None:
            self.gate_s2s3 = nn.Sequential(
                nn.Linear(irreps_node_input[0][0], irreps_node_input[0][0]),
                nn.Sigmoid(),
            )

        # if "so2" in ffn_type:
        #     self.norm_3 = get_norm_layer(norm_layer)(self.irreps_node_input)
        #     self.rot_func = SO3_Rotation(
        #         lmax=max([i[1].l for i in self.irreps_node_input]),
        #         irreps=self.irreps_node_input,
        #     )
        #     self.so2_ffn = SO2_Convolution(irreps_in=self.irreps_node_input)
        # elif "newso2" in ffn_type:
        #     self.norm_3 = get_norm_layer(norm_layer)(self.irreps_node_input)
        #     self.rot_func = SO3_Rotation(
        #         lmax=max([i[1].l for i in self.irreps_node_input]),
        #         irreps=self.irreps_node_input,
        #     )
        #     self.so2_ffn = SO2_Convolution_sameorder(irreps_in=self.irreps_node_input)

        self.manybody_ffn = None
        if "2body" in ffn_type:
            self.gate_manybody = nn.Sequential(
                nn.Linear(irreps_node_input[0][0], irreps_node_input[0][0]),
                nn.Sigmoid(),
            )
            self.norm_manybody = get_normalization_layer(
                norm_layer, lmax=self.lmax, num_channels=irreps_node_input[0][0]
            )
            self.manybody_ffn = Body2_interaction(self.irreps_node_input)

        if "3body" in ffn_type:
            self.norm_manybody = get_normalization_layer(
                norm_layer, lmax=self.lmax, num_channels=irreps_node_input[0][0]
            )
            self.manybody_ffn = Body3_interaction_MACE(
                self.irreps_node_input, internal_weights=True
            )
        self.ffn_grid_escn = None
        if "grid_nonlinear" in ffn_type:
            self.ffn_grid_escn = FeedForwardNetwork_escn(
                self.irreps_node_input[0][0],
                self.irreps_node_input[0][0],
                self.irreps_node_input[0][0],
                lmax=self.lmax,
            )

        # self.norm_3 = get_norm_layer(norm_layer)(self.irreps_node_input)
        # self.ffn_vec2scalar = FeedForwardVec2Scalar(
        #     irreps_node_input=self.irreps_node_input,  # self.concat_norm_output.irreps_out,
        #     irreps_node_output=self.irreps_node_output,
        # )

        self.add_rope = add_rope
        self.sparse_attn = sparse_attn

        self.edge_attn = None
        if "edge_attn" in ffn_type:
            self.attn_scalar = nn.Parameter(torch.ones(1), requires_grad=True)
            self.edge_attn = nn.MultiheadAttention(
                embed_dim=attn_weight_input_dim,
                num_heads=32,
                dropout=0.1,
                bias=True,
                batch_first=True,
            )
            self.edge_to_node = nn.Sequential(
                nn.Linear(attn_weight_input_dim, self.irreps_node_input[0][0]),
                nn.LayerNorm(self.irreps_node_input[0][0]),
                nn.SiLU(),
                nn.Linear(self.irreps_node_input[0][0], self.irreps_node_input[0][0]),
            )

    def forward(
        self,
        node_pos,
        node_irreps,
        edge_dis,
        edge_vec,
        attn_weight,  # e.g. rbf(|r_ij|) or relative pos in sequence
        atomic_numbers,
        attn_mask,
        poly_dist=None,
        batch=None,
        batched_data=None,
        **kwargs,
    ):
        """

        from sfm.models.psm.equivariant.e2former import TransBlock
        irreps_in = e3nn.o3.Irreps("256x0e+256x1e+256x2e")
        B,L = 4,100
        dis_embedding_dim = 32
        node_pos = torch.randn(B,L,3)
        edge_dis = torch.sqrt(torch.sum((node_pos.view(B,L,1,3)-node_pos.view(B,1,L,3))**2,dim = -1))
        dis_embedding = torch.randn(B,L,L,dis_embedding_dim)
        attn_mask = torch.randn(B,L,L,1)>0
        atomic_numbers = torch.randint(0,10,(B,L))
        func = TransBlock(
                irreps_in,
                irreps_in,
                attn_weight_input_dim=dis_embedding_dim, # e.g. rbf(|r_ij|) or relative pos in sequence
                num_attn_heads=8,
                attn_scalar_head = 48,
                irreps_head="32x0e+32x1e+32x2e",
                rescale_degree=False,
                nonlinear_message=False,
                alpha_drop=0.1,
                proj_drop=0,
                drop_path_rate=0.1,
                attn_type = 'second-order',
                ffn_type="eqv2ffn",
                norm_layer="rms_norm_sh_BL", # used for norm 1 and norm2
            )

        out = func.forward(
                node_pos,
                torch.randn(B,L,9,256),
                edge_dis,
                dis_embedding, # e.g. rbf(|r_ij|) or relative pos in sequence
                atomic_numbers,
                attn_mask,

                batch=None)
        """

        ## residual connection
        node_irreps_res = node_irreps
        node_irreps = self.norm_1(node_irreps)
        # B,L1,L2 = attn_weight.shape[:3]

        # if self.attn_type.startswith("scalable"):
        #     attn_weight = attn_weight.reshape(B,L1,L2,-1,self.attn_weight_input_dim)
        #     attn_weight = self.norm_edge_1(attn_weight)

        node_irreps, attn_weight = self.ga(
            node_pos=node_pos,
            node_irreps_input=node_irreps,
            edge_dis=edge_dis,
            poly_dist=poly_dist,
            edge_vec=edge_vec,
            attn_weight=attn_weight,
            atomic_numbers=atomic_numbers,
            attn_mask=attn_mask,
            batched_data=batched_data,
            add_rope=self.add_rope,
            sparse_attn=self.sparse_attn,
        )

        if self.ffn_grid_escn is not None:
            node_irreps = self.ffn_grid_escn(node_irreps, node_irreps_res)
            return node_irreps, attn_weight
        if self.drop_path is not None:
            node_irreps = self.drop_path(node_irreps, batch)
        node_irreps = node_irreps + node_irreps_res

        if self.ffn_s2 is not None and self.ffn_s3 is None:
            ## residual connection
            node_irreps_res = node_irreps
            node_irreps = self.norm_s2(node_irreps)
            node_irreps = self.ffn_s2(node_irreps)
            if self.drop_path is not None:
                node_irreps = self.drop_path(node_irreps, batch)
            node_irreps = self.proj_drop_func(node_irreps)
            node_irreps = node_irreps_res + node_irreps

        if self.ffn_s3 is not None and self.ffn_s2 is None:
            ## residual connection
            node_irreps_res = node_irreps
            node_irreps = self.norm_s3(node_irreps)
            node_irreps = self.ffn_s3(node_irreps)
            if self.drop_path is not None:
                node_irreps = self.drop_path(node_irreps, batch)
            node_irreps = self.proj_drop_func(node_irreps)

            node_irreps = node_irreps_res + node_irreps

        if self.ffn_s2 is not None and self.ffn_s3 is not None:
            node_irreps_res = node_irreps
            node_irreps_s2 = self.norm_s2(node_irreps)
            node_irreps_s2 = self.ffn_s2(node_irreps_s2)
            if self.drop_path is not None:
                node_irreps_s2 = self.drop_path(node_irreps_s2, batch)

            node_irreps_s3 = self.norm_s3(node_irreps)
            node_irreps_s3 = self.ffn_s3(node_irreps_s3)
            if self.drop_path is not None:
                node_irreps_s3 = self.drop_path(node_irreps_s3, batch)

            gates = self.gate_s2s3(node_irreps[:, 0:1])

            node_irreps = node_irreps_res + self.proj_drop_func(
                node_irreps_s2 * gates + node_irreps_s3 * (1 - gates)
            )

        if self.so2_ffn is not None:
            node_irreps_res = node_irreps
            self.rot_func.set_wigner(
                self.rot_func.init_edge_rot_mat(node_pos.reshape(-1, 3))
            )

            node_irreps = self.norm_3(node_irreps, batch=batch)
            node_irreps = self.rot_func.rotate(node_irreps)
            node_irreps = self.so2_ffn(node_irreps)
            node_irreps = self.rot_func.rotate_inv(node_irreps)

            node_irreps = node_irreps_res + node_irreps

        if self.manybody_ffn is not None:
            gates = self.gate_manybody(node_irreps[:, 0:1])
            node_irreps_res = node_irreps
            node_irreps = self.norm_manybody(node_irreps, batch=batch)
            node_irreps = self.manybody_ffn(node_irreps, atomic_numbers)
            node_irreps = gates * node_irreps_res + (1 - gates) * node_irreps

        if self.edge_attn is not None:
            angle_embed = edge_vec / torch.norm(edge_vec, dim=-1, keepdim=True)
            angle_embed = torch.sum(
                angle_embed.unsqueeze(dim=1) * angle_embed.unsqueeze(dim=2), dim=-1
            )
            angle_embed = self.attn_scalar * angle_embed.unsqueeze(dim=1).expand(
                -1, self.edge_attn.num_heads, -1, -1
            ).reshape(-1, angle_embed.shape[-1], angle_embed.shape[-1])
            attn_hidden = self.edge_attn(
                query=attn_weight,
                key=attn_weight,
                value=attn_weight,
                attn_mask=batched_data["edge_inter_mask"] + angle_embed,
                need_weights=False,
            )[0]
            attn_hidden = attn_hidden.masked_fill(attn_mask, 0)
            attn_hidden = self.edge_to_node(attn_hidden)
            node_irreps[:, 0, :] = node_irreps[:, 0, :] + torch.mean(attn_hidden, dim=1)
            attn_weight = attn_weight + attn_hidden
        # node_irreps_res = node_irreps
        # node_irreps = self.norm_3(node_irreps, batch=batch)
        # node_irreps = self.ffn_vec2scalar(node_irreps)
        # node_irreps = node_irreps_res + node_irreps
        return node_irreps, attn_weight


class EdgeDegreeEmbeddingNetwork_higherorder(torch.nn.Module):
    def __init__(
        self,
        irreps_node_embedding,
        avg_aggregate_num=10,
        number_of_basis=32,
        cutoff=15,
        time_embed=False,
        use_layer_norm=True,
        use_atom_edge=False,
        name="default",
        **kwargs,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.irreps_node_embedding = (
            o3.Irreps(irreps_node_embedding)
            if isinstance(irreps_node_embedding, str)
            else irreps_node_embedding
        )
        if self.irreps_node_embedding[0][1].l != 0:
            raise ValueError("node embedding must have sph order 0 embedding.")
        self.number_of_basis = number_of_basis
        # self.gbf = GaussianLayer(number_of_basis)  # default output_dim = 128
        self.gbf_projs = nn.ModuleList()

        self.scalar_dim = self.irreps_node_embedding[0][0]
        if time_embed:
            self.time_embed_proj = nn.Sequential(
                nn.Linear(self.scalar_dim, self.scalar_dim, bias=True),
                nn.SiLU(),
                nn.Linear(self.scalar_dim, number_of_basis, bias=True),
            )
        self.max_num_elements = 300
        self.use_atom_edge = use_atom_edge
        if use_atom_edge:
            self.source_embedding = nn.Embedding(self.max_num_elements, number_of_basis)
            self.target_embedding = nn.Embedding(self.max_num_elements, number_of_basis)
        else:
            self.source_embedding = None
            self.target_embedding = None
        self.weight_list = nn.ParameterList()
        self.lmax = len(self.irreps_node_embedding) - 1
        for idx in range(len(self.irreps_node_embedding)):
            self.gbf_projs.append(
                RadialProfile(
                    [
                        number_of_basis * 3 if use_atom_edge else number_of_basis,
                        min(number_of_basis, 128),
                        min(number_of_basis, 128),
                        self.irreps_node_embedding[idx][0],
                    ],
                    use_layer_norm=use_layer_norm,
                )
            )

            # out_feature = self.irreps_node_embedding[idx][0]
            # weight = torch.nn.Parameter(torch.randn(out_feature, number_of_basis))
            # bound = 1 / math.sqrt(number_of_basis)
            # torch.nn.init.uniform_(weight, -bound, bound)
            # self.weight_list.append(weight)

        self.name = name

        self.source_embedding_elec = nn.Embedding(
            self.max_num_elements, number_of_basis
        )
        self.target_embedding_elec = nn.Embedding(
            self.max_num_elements, number_of_basis
        )
        self.uniform_center_count = 5
        self.sph_grid_channel = 32
        self.linear_sigmaco = torch.nn.Sequential(
            nn.Linear(number_of_basis * 3 if use_atom_edge else number_of_basis, 128),
            nn.GELU(),
            nn.Linear(128, 2 * self.uniform_center_count * self.sph_grid_channel),
        )
        self.electron_density = Electron_Density_Descriptor(
            uniform_center_count=self.uniform_center_count,
            num_sphere_points=16,
            channel=self.sph_grid_channel,
            lmax=self.lmax,
            output_channel=self.irreps_node_embedding[idx][0],
        )

        self.proj = SO3_Linear_e2former(
            self.irreps_node_embedding[idx][0] * 2,
            self.irreps_node_embedding[idx][0],
            lmax=self.lmax,
        )
        self.avg_aggregate_num = avg_aggregate_num

    def forward(
        self,
        node_input,
        node_pos,
        edge_dis,
        atomic_numbers,
        edge_vec,
        batched_data,
        attn_mask,
        edge_scalars,
        time_embed=None,
        **kwargs,
    ):
        """
        model =  EdgeDegreeEmbeddingNetwork_higherorder(
                "256x0e+256x1e+256x2e",
                avg_aggregate_num=10,
                number_of_basis=32,
                cutoff=5,
                time_embed=False,
                use_atom_edge=True)

        f_N = 3+9+20
        f_N2 = 70
        topK = 5
        num_basis = 32
        hidden = 256
        node_input = None
        exp_node_pos = torch.randn(f_N2,3)

        node_pos = torch.randn(f_N,3)
        edge_dis = torch.randn(f_N,topK)
        atomic_numbers = torch.randint(0,10,(f_N,))
        edge_vec = torch.randn(f_N,topK,3)

        attn_mask = torch.randn(f_N,topK,1)>0
        edge_scalars = torch.randn(f_N,topK,num_basis)
        f_sparse_idx_node = torch.randint(0,f_N,(f_N,topK))
        f_sparse_idx_expnode = torch.randint(0,f_N2,(f_N2,topK))
        batched_data = {'f_sparse_idx_node':f_sparse_idx_node,'f_sparse_idx_expnode':f_sparse_idx_expnode}

        out = model(node_input,
                node_pos,
                edge_dis,
                atomic_numbers,
                edge_vec,
                batched_data,
                attn_mask,
                edge_scalars,)

        """

        f_sparse_idx_node = batched_data["f_sparse_idx_node"]
        topK = edge_vec.shape[1]
        tgt_atm = (
            self.target_embedding(atomic_numbers).unsqueeze(dim=1).repeat(1, topK, 1)
        )
        src_atm = self.source_embedding(atomic_numbers)[f_sparse_idx_node]

        edge_dis_embed = torch.cat(
            [edge_scalars, tgt_atm, src_atm],
            dim=-1,
        )
        node_features = []
        for idx in range(len(self.irreps_node_embedding)):
            lx = o3.spherical_harmonics(
                l=self.irreps_node_embedding[idx][1].l,
                x=edge_vec,
                normalize=True,  # TODO： norm ablation 3
                normalization="norm",
            )  # * adj.reshape(B,L,L,1) #B*L*L*(2l+1)
            edge_fea = self.gbf_projs[idx](edge_dis_embed)
            edge_fea = torch.where(attn_mask, 0, edge_fea)
            # lx_embed = torch.sum(lx.unsqueeze(dim = 3)*edge_fea.unsqueeze(dim = 2),dim = 1)  # lx:B*L*L*(2l+1)  edge_fea:B*L*L*number of basis
            lx_embed = torch.einsum(
                "mnd,mnh->mdh", lx, edge_fea
            )  # lx:B*L*L*(2l+1)  edge_fea:B*L*L*number of basis
            node_features.append(lx_embed)

        node_features = torch.cat(node_features, dim=1) / self.avg_aggregate_num

        if self.name == "elec":
            tgt_atm = (
                self.target_embedding_elec(atomic_numbers)
                .unsqueeze(dim=1)
                .repeat(1, topK, 1)
            )
            src_atm = self.source_embedding_elec(atomic_numbers)[f_sparse_idx_node]
            edge_dis_embed2 = torch.cat(
                [edge_scalars, tgt_atm, src_atm],
                dim=-1,
            )
            sigma, co = torch.chunk(
                self.linear_sigmaco(edge_dis_embed2), dim=-1, chunks=2
            )

            token_embedding = self.electron_density(
                node_pos, rji=-edge_vec, sigma=sigma, co=co, neighbor_mask=~attn_mask
            )
            node_features = self.proj(
                torch.cat([node_features, token_embedding], dim=-1)
            )
            node_features = self.proj(node_features)
        return node_features  # node_features


class EdgeDegreeEmbeddingNetwork_higherorder_v3(torch.nn.Module):
    def __init__(
        self,
        irreps_node_embedding,
        avg_aggregate_num=10,
        number_of_basis=32,
        cutoff=15,
        time_embed=False,
        use_atom_edge=False,
        **kwargs,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.irreps_node_embedding = (
            o3.Irreps(irreps_node_embedding)
            if isinstance(irreps_node_embedding, str)
            else irreps_node_embedding
        )
        if self.irreps_node_embedding[0][1].l != 0:
            raise ValueError("node embedding must have sph order 0 embedding.")
        self.gbf = GaussianLayer_Edgetype(number_of_basis)  # default output_dim = 128
        self.gbf_projs = nn.ModuleList()

        self.scalar_dim = self.irreps_node_embedding[0][0]
        if time_embed:
            self.time_embed_proj = nn.Sequential(
                nn.Linear(self.scalar_dim, self.scalar_dim, bias=True),
                nn.SiLU(),
                nn.Linear(self.scalar_dim, number_of_basis, bias=True),
            )
        self.max_num_elements = 300
        self.use_atom_edge = use_atom_edge
        if use_atom_edge:
            self.source_embedding = nn.Embedding(self.max_num_elements, number_of_basis)
            self.target_embedding = nn.Embedding(self.max_num_elements, number_of_basis)
        else:
            self.source_embedding = None
            self.target_embedding = None
        self.weight_list = nn.ParameterList()
        for idx in range(len(self.irreps_node_embedding)):
            self.gbf_projs.append(
                RadialProfile(
                    [
                        number_of_basis * 3 if use_atom_edge else number_of_basis,
                        number_of_basis,
                        self.irreps_node_embedding[idx][0],
                    ],
                    use_layer_norm=True,
                )
            )

            # out_feature = self.irreps_node_embedding[idx][0]
            # weight = torch.nn.Parameter(torch.randn(out_feature, number_of_basis))
            # bound = 1 / math.sqrt(number_of_basis)
            # torch.nn.init.uniform_(weight, -bound, bound)
            # self.weight_list.append(weight)

        # self.proj = IrrepsLinear(self.irreps_node_embedding, self.irreps_node_embedding)
        self.avg_aggregate_num = avg_aggregate_num

    def forward(
        self,
        node_input,
        node_pos,
        edge_dis,
        atomic_numbers,
        edge_vec,
        batched_data,
        attn_mask,
        time_embed=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        node_pos : postiion
            tensor of shape ``(B, L, 3)``

        edge_scalars : rbf of distance
            tensor of shape ``(B, L, L, number_of_basis)``

        edge_dis : distances of all node pairs
            tensor of shape ``(B, L, L)``
        """

        B, L = node_pos.shape[:2]

        # edge_vec = node_pos.unsqueeze(2) - node_pos.unsqueeze(1)  # B, L, L, 3
        node_type_edge = batched_data["node_type_edge"]
        edge_dis_embed = self.gbf(edge_dis, node_type_edge.long())
        if time_embed is not None:
            edge_dis_embed += self.time_embed_proj(time_embed).unsqueeze(-2)

        if self.source_embedding is not None:
            src_atm = self.source_embedding(atomic_numbers)  # B*L*hidden
            tgt_atm = self.target_embedding(atomic_numbers)  # B*L*hidden

            edge_dis_embed = torch.cat(
                [
                    edge_dis_embed,
                    tgt_atm.reshape(B, L, 1, -1).repeat(1, 1, L, 1),
                    src_atm.reshape(B, 1, L, -1).repeat(1, L, 1, 1),
                ],
                dim=-1,
            )

        edge_vec = edge_vec / edge_dis.unsqueeze(
            dim=-1
        )  # norm ablation 4: command this line
        node_features = []
        for idx in range(len(self.irreps_node_embedding)):
            # if self.irreps_node_embedding[idx][1].l ==0:
            #     node_features.append(torch.zeros(
            #                         (B,L,self.irreps_node_embedding[idx][0]),
            #                          dtype=edge_dis.dtype,
            #                          device = edge_dis.device))
            #     continue

            lx = o3.spherical_harmonics(
                l=self.irreps_node_embedding[idx][1].l,
                x=edge_vec,
                normalize=False,  # TODO： norm ablation 3
                normalization="norm",
            )  # * adj.reshape(B,L,L,1) #B*L*L*(2l+1)
            edge_fea = self.gbf_projs[idx](edge_dis_embed)
            edge_fea = torch.where(attn_mask, 0, edge_fea)

            # lx_embed = torch.einsum("bmnd,bnh->bmhd",lx,node_embed) #lx:B*L*L*(2l+1)  node_embed:B*L*hidden
            lx_embed = torch.einsum(
                "bmnd,bmnh->bmdh", lx, edge_fea
            )  # lx:B*L*L*(2l+1)  edge_fea:B*L*L*number of basis

            # lx_embed = torch.matmul(self.weight_list[idx], lx_embed).reshape(
            #     B, L, -1
            # )  # self.weight_list[idx]:irreps_channel*hidden, lx_embed:B*L*hidden*(2l+1)
            node_features.append(lx_embed)

        node_features = torch.cat(node_features, dim=2) / self.avg_aggregate_num
        # node_features = self.proj(node_features)
        return node_features


class EdgeDegreeEmbeddingNetwork_eqv2(torch.nn.Module):
    def __init__(
        self,
        irreps_node_embedding,
        avg_aggregate_num=10,
        number_of_basis=32,
        cutoff=15,
        lmax=2,
        time_embed=False,
        **kwargs,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.irreps_node_embedding = (
            o3.Irreps(irreps_node_embedding)
            if isinstance(irreps_node_embedding, str)
            else irreps_node_embedding
        )
        if self.irreps_node_embedding[0][1].l != 0:
            raise ValueError("node embedding must have sph order 0 embedding.")

        self.lmax = self.irreps_node_embedding[-1][1].l
        self.sph_ch = self.irreps_node_embedding[0][0]

        # # Statistics of IS2RE 100K
        _AVG_NUM_NODES = 77.81317
        self.sphere_channels = self.sph_ch
        self.lmax_list = [self.lmax]
        self.mmax_list = [2]
        self.num_resolutions = len(self.lmax_list)
        # self.SO3_rotation = SO3_rotation

        self.SO3_grid = torch.nn.ModuleList()
        for lval in range(max(self.lmax_list) + 1):
            so3_m_grid = torch.nn.ModuleList()
            for m in range(max(self.lmax_list) + 1):
                so3_m_grid.append(
                    SO3_Grid(
                        lval,
                        m,
                        resolution=18,
                        normalization="component",
                    )
                )

            self.SO3_grid.append(so3_m_grid)
        self.mappingReduced = CoefficientMapping([self.lmax], [2])

        self.m_0_num_coefficients = self.mappingReduced.m_size[0]
        self.m_all_num_coefficents = len(self.mappingReduced.l_harmonic)

        # Create edge scalar (invariant to rotations) features
        # Embedding function of the atomic numbers
        self.max_num_elements = 256
        self.edge_channels_list = copy.deepcopy([number_of_basis, 128, 128])
        self.use_atom_edge_embedding = True

        if self.use_atom_edge_embedding:
            self.source_embedding = nn.Embedding(
                self.max_num_elements, self.edge_channels_list[-1]
            )
            self.target_embedding = nn.Embedding(
                self.max_num_elements, self.edge_channels_list[-1]
            )
            nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
            nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)
            self.edge_channels_list[0] = (
                self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]
            )
        else:
            self.source_embedding, self.target_embedding = None, None

        # Embedding function of distance
        self.edge_channels_list.append(self.m_0_num_coefficients * self.sphere_channels)
        self.rad_func = RadialFunction(self.edge_channels_list)

        self.rescale_factor = _AVG_DEGREE

    def _forward(
        self,
        atomic_numbers,
        edge_distance,
        edge_index,
        SO3_edge_rot=None,
        mappingReduced=None,
        attn_mask=None,
    ):
        f_N1, topK = edge_distance.shape[:2]
        edge_distance = edge_distance.reshape(f_N1 * topK, -1)

        if self.use_atom_edge_embedding:
            source_element = atomic_numbers[edge_index[0]]  # Source atom atomic number
            target_element = atomic_numbers[edge_index[1]]  # Target atom atomic number
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            x_edge = torch.cat(
                (edge_distance, source_embedding, target_embedding), dim=1
            )
        else:
            x_edge = edge_distance

        x_edge_m_0 = self.rad_func(x_edge)
        x_edge_m_0 = x_edge_m_0.reshape(
            -1, self.m_0_num_coefficients, self.sphere_channels
        )
        x_edge_m_pad = torch.zeros(
            (
                x_edge_m_0.shape[0],
                (self.m_all_num_coefficents - self.m_0_num_coefficients),
                self.sphere_channels,
            ),
            device=x_edge_m_0.device,
        )
        x_edge_m_all = torch.cat((x_edge_m_0, x_edge_m_pad), dim=1)

        x_edge_embedding = SO3_Embedding(
            0,
            self.lmax_list.copy(),
            self.sphere_channels,
            device=x_edge_m_all.device,
            dtype=x_edge_m_all.dtype,
        )
        x_edge_embedding.set_embedding(x_edge_m_all)
        x_edge_embedding.set_lmax_mmax(self.lmax_list.copy(), self.mmax_list.copy())

        # Reshape the spherical harmonics based on l (degree)
        x_edge_embedding._l_primary(mappingReduced)

        # Rotate back the irreps
        x_edge_embedding._rotate_inv(SO3_edge_rot, mappingReduced)

        # Compute the sum of the incoming neighboring messages for each target node
        out = x_edge_embedding.embedding.reshape(f_N1, topK, -1)
        out = out.masked_fill(attn_mask, 0)
        out = out.reshape(f_N1, topK, (self.lmax + 1) ** 2, -1)

        out = torch.sum(out, dim=1) / self.rescale_factor

        return out

    def forward(
        self,
        node_input,
        node_pos,
        edge_dis,
        atomic_numbers,
        edge_vec,
        batched_data,
        attn_mask,
        edge_scalars,
        time_embed=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        node_pos : postiion
            tensor of shape ``(B, L, 3)``

        edge_scalars : rbf of distance
            tensor of shape ``(B, L, L, number_of_basis)``

        edge_dis : distances of all node pairs
            tensor of shape ``(B, L, L)``


        from sfm.models.psm.equivariant.e2former import EdgeDegreeEmbeddingNetwork_eqv2
        self__irreps_node_embedding = e3nn.o3.Irreps("128x0e+128x1e+128x2e")
        self__number_of_basis = 64
        self__edge_deg_embed_dense = EdgeDegreeEmbeddingNetwork_eqv2(
            self__irreps_node_embedding,
            23.555,
            cutoff=5,
            number_of_basis=self__number_of_basis,
            time_embed=False,
        )
        B = 2
        L = 10
        basis = 64
        pos = torch.randn(B,L,3)
        dist = torch.norm(pos.unsqueeze(dim = 2)-pos.unsqueeze(dim = 1),dim = -1)
        edge_vec = pos.unsqueeze(dim = 2)-pos.unsqueeze(dim = 1)
        atomic_numbers = torch.randint(0,10,(B,L))
        dist_embedding = torch.randn(B,L,L,basis)
        attn_mask = torch.randn(B,L,L,1)>0
        out = self__edge_deg_embed_dense(
        None,
        pos,
        dist,
        batch=None,
        attn_mask=attn_mask,
        atomic_numbers=atomic_numbers,
        edge_vec=edge_vec,
        batched_data=None,
        time_embed=None,
        edge_scalars=dist_embedding,
        )
        print(out.shape)
        """
        f_N1, topK = attn_mask.shape[:2]
        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        self.SO3_edge_rot = torch.nn.ModuleList()
        for i in range(1):
            self.SO3_edge_rot.append(
                SO3_Rotation(
                    _init_edge_rot_mat(edge_vec.reshape(f_N1 * topK, 3)),
                    self.lmax_list[i],
                )
            )

        #######################for memory saving
        x = SO3_Embedding(
            f_N1,
            self.lmax_list,
            self.sphere_channels,
            node_input.device,
            node_input.dtype,
        )
        x.embedding = node_input
        x_embedding = self._forward(
            atomic_numbers,
            edge_scalars,
            edge_index=(
                batched_data["f_sparse_idx_node"].reshape(-1),
                torch.arange(f_N1).reshape(f_N1, -1).repeat(1, topK).reshape(-1),
            ),
            SO3_edge_rot=self.SO3_edge_rot,
            mappingReduced=self.mappingReduced,
            attn_mask=attn_mask,
        )

        return x_embedding


class BOOEmbedding(torch.nn.Module):
    """Bond Orientational Order embedding module.

    Computes rotationally invariant features from the spherical distribution of bonds
    around each node using spherical harmonics, following Steinhardt et al. [1983].
    """

    def __init__(self, max_l=4, hidden_dim=512):
        """
        Args:
            max_l (int): Maximum order of spherical harmonics to use (L in the paper)
        """
        super().__init__()
        self.max_l = max_l
        self.linear = nn.Linear(max_l + 1, hidden_dim)

    def forward(self, edge_vec, edge_mask):
        """
        Args:
            edge_vec (torch.Tensor): Bond vectors of shape (B, N, K, 3) where:
                B is batch size
                N is number of nodes
                K is max number of neighbors
            edge_mask (torch.Tensor): Mask for valid edges of shape (B, N, K, 1)
            batch (torch.Tensor, optional): Batch indices for each node

        Returns:
            torch.Tensor: BOO features of shape (B, N, max_l+1)
        """
        B, N, K, _ = edge_vec.shape

        # Normalize bond vectors
        edge_vec_norm = torch.norm(edge_vec, dim=-1, keepdim=True)
        edge_vec_normalized = edge_vec / (edge_vec_norm + 1e-10)

        # Count valid neighbors per node
        n_neighbors = edge_mask.squeeze(-1).float().sum(dim=-1)  # Shape: (B, N)
        n_neighbors = n_neighbors.clamp(min=1)  # Avoid division by zero

        # Initialize BOO features
        boo_features = []

        for l in range(self.max_l + 1):
            # Compute spherical harmonics Y_l^m for all bonds
            # Shape: (B, N, K, 2l+1)
            Y_lm = e3nn.o3.spherical_harmonics(
                l, edge_vec_normalized, normalize=True, normalization="component"
            )

            # Apply mask and normalize by number of neighbors
            # Shape: (B, N, K, 2l+1)
            Y_lm = Y_lm * edge_mask
            Y_lm = Y_lm / n_neighbors.view(B, N, 1, 1)

            # Sum over neighbors (K dimension)
            # Shape: (B, N, 2l+1)
            q_lm = torch.sum(Y_lm, dim=2)

            # Compute BOO^(l) = sum_m |q_lm|^2
            # Include normalization factor sqrt(4π/(2l+1))
            norm_factor = math.sqrt(4 * math.pi / (2 * l + 1))
            q_lm = q_lm * norm_factor
            boo_l = torch.sum(
                torch.abs(q_lm) ** 2, dim=-1, keepdim=True
            )  # Shape: (B, N, 1)

            boo_features.append(boo_l)

        # Concatenate all BOO features
        # Final shape: (B, N, max_l+1)
        boo = torch.cat(boo_features, dim=-1)
        boo = self.linear(boo)

        return boo


class CoefficientMapping(torch.nn.Module):
    """
    Helper functions for coefficients used to reshape l<-->m and to get coefficients of specific degree or order

    Args:
        lmax_list (list:int):   List of maximum degree of the spherical harmonics
        mmax_list (list:int):   List of maximum order of the spherical harmonics
        device:                 Device of the output
    """

    def __init__(
        self,
        lmax_list: list[int],
        mmax_list: list[int],
    ) -> None:
        super().__init__()

        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions = len(lmax_list)

        # Compute the degree (l) and order (m) for each
        # entry of the embedding

        self.l_harmonic = torch.tensor([]).long()
        self.m_harmonic = torch.tensor([]).long()
        self.m_complex = torch.tensor([]).long()

        self.res_size = torch.zeros([self.num_resolutions]).long()
        offset = 0
        for i in range(self.num_resolutions):
            for lval in range(self.lmax_list[i] + 1):
                mmax = min(self.mmax_list[i], lval)
                m = torch.arange(-mmax, mmax + 1).long()
                self.m_complex = torch.cat([self.m_complex, m], dim=0)
                self.m_harmonic = torch.cat(
                    [self.m_harmonic, torch.abs(m).long()], dim=0
                )
                self.l_harmonic = torch.cat(
                    [self.l_harmonic, m.fill_(lval).long()], dim=0
                )
            self.res_size[i] = len(self.l_harmonic) - offset
            offset = len(self.l_harmonic)

        num_coefficients = len(self.l_harmonic)
        self.to_m = torch.nn.Parameter(
            torch.zeros([num_coefficients, num_coefficients]), requires_grad=False
        )
        self.m_size = torch.zeros([max(self.mmax_list) + 1]).long()

        # The following is implemented poorly - very slow. It only gets called
        # a few times so haven't optimized.
        offset = 0
        for m in range(max(self.mmax_list) + 1):
            idx_r, idx_i = self.complex_idx(m)

            for idx_out, idx_in in enumerate(idx_r):
                self.to_m[idx_out + offset, idx_in] = 1.0
            offset = offset + len(idx_r)
            self.m_size[m] = int(len(idx_r))

            for idx_out, idx_in in enumerate(idx_i):
                self.to_m[idx_out + offset, idx_in] = 1.0
            offset = offset + len(idx_i)

    # Return mask containing coefficients of order m (real and imaginary parts)
    def complex_idx(self, m, lmax: int = -1):
        if lmax == -1:
            lmax = max(self.lmax_list)

        indices = torch.arange(len(self.l_harmonic))
        # Real part
        mask_r = torch.bitwise_and(self.l_harmonic.le(lmax), self.m_complex.eq(m))
        mask_idx_r = torch.masked_select(indices, mask_r)

        mask_idx_i = torch.tensor([]).long()
        # Imaginary part
        if m != 0:
            mask_i = torch.bitwise_and(self.l_harmonic.le(lmax), self.m_complex.eq(-m))
            mask_idx_i = torch.masked_select(indices, mask_i)

        return mask_idx_r, mask_idx_i

    # Return mask containing coefficients less than or equal to degree (l) and order (m)
    def coefficient_idx(self, lmax: int, mmax: int) -> torch.Tensor:
        mask = torch.bitwise_and(self.l_harmonic.le(lmax), self.m_harmonic.le(mmax))
        indices = torch.arange(len(mask))

        return torch.masked_select(indices, mask)


class MessageBlock_escn(torch.nn.Module):
    def __init__(
        self,
        irreps_node_input="256x0e+256x1e+256x2e",
        attn_weight_input_dim: int = 32,  # e.g. rbf(|r_ij|) or relative pos in sequence
        num_attn_heads: int = 8,
        attn_scalar_head: int = 32,
        irreps_head="32x0e+32x1e+32x2e",
        alpha_drop=0.1,
        rescale_degree=False,
        nonlinear_message=False,
        proj_drop=0.1,
        tp_type="v1",
        attn_type="first-order",  ## second-order
        add_rope=True,
        layer_id=0,
        irreps_origin="1x0e+1x1e+1x2e",
        neighbor_weight=None,
        atom_type_cnt=256,
        **kwargs,
    ):
        self.irreps_node_input = (
            o3.Irreps(irreps_node_input)
            if isinstance(irreps_node_input, str)
            else irreps_node_input
        )
        self.scalar_dim = self.irreps_node_input[0][0] // 2  # scalar_dim x 0e
        self.lmax = len(self.irreps_node_input) - 1
        self.num_attention_heads = num_attn_heads
        self.attn_scalar_head = attn_scalar_head
        self.attn_weight_input_dim = attn_weight_input_dim
        self.atom_type_cnt = atom_type_cnt
        self.irreps_head = (
            o3.Irreps(irreps_head) if isinstance(irreps_head, str) else irreps_head
        )
        # new params
        self.tp_type = tp_type
        self.attn_type = attn_type

        super().__init__()

        self.proj_input = SO3_Linear_e2former(
            self.irreps_node_input[0][0],
            self.scalar_dim,
            lmax=self.lmax,
        )
        self.proj_final = SO3_Linear_e2former(
            self.scalar_dim,
            self.irreps_node_input[0][0],
            lmax=self.lmax,
        )
        self.act = torch.nn.SiLU()

        self.sphere_channels = self.scalar_dim
        self.hidden_channels = self.scalar_dim
        self.edge_channels = self.attn_weight_input_dim // 2
        self.lmax_list = [self.lmax]
        self.mmax_list = [2]

        # Embedding function of the atomic numbers
        self.source_embedding = nn.Embedding(self.atom_type_cnt, self.edge_channels)
        self.target_embedding = nn.Embedding(self.atom_type_cnt, self.edge_channels)
        nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
        nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)
        # Embedding function of the edge
        self.fc1_dist = nn.Linear(self.attn_weight_input_dim, self.edge_channels)
        self.fc1_edge_attr = nn.Sequential(
            self.act,
            nn.Linear(
                self.edge_channels,
                self.edge_channels,
            ),
            self.act,
        )

        # Create SO(2) convolution blocks
        self.so2_block_source = SO2Block(
            self.sphere_channels,
            self.hidden_channels,
            self.edge_channels,
            self.lmax_list,
            self.mmax_list,
            self.act,
        )
        self.so2_block_target = SO2Block(
            self.sphere_channels,
            self.hidden_channels,
            self.edge_channels,
            self.lmax_list,
            self.mmax_list,
            self.act,
        )
        self.SO3_grid = torch.nn.ModuleList()
        for lval in range(max(self.lmax_list) + 1):
            so3_m_grid = torch.nn.ModuleList()
            for m in range(max(self.lmax_list) + 1):
                so3_m_grid.append(SO3_Grid(lval, m, resolution=18))

            self.SO3_grid.append(so3_m_grid)
        self.mappingReduced = CoefficientMapping([self.lmax], [2])
        # [0, -1,  0,  2, -2, -1,  0,  1,  2]
        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        # self.SO3_edge_rot = torch.nn.ModuleList()
        # for i in range(self.num_resolutions):
        #     self.SO3_edge_rot.append(SO3_Rotation(edge_rot_mat, self.lmax_list[i]))

    def _forward(
        self,
        x,
        atomic_numbers,
        edge_distance,
        edge_index,
        SO3_edge_rot,
        mappingReduced,
        attn_mask,
    ):
        ###############################################################
        # Compute messages
        ###############################################################
        f_N1, topK = edge_distance.shape[:2]
        edge_distance = edge_distance.reshape(f_N1 * topK, -1)
        # Compute edge scalar features (invariant to rotations)
        # Uses atomic numbers and edge distance as inputs
        x_edge = self.fc1_edge_attr(
            self.fc1_dist(edge_distance)
            + self.source_embedding(atomic_numbers)[edge_index[0]]
            + self.target_embedding(atomic_numbers)[  # Source atom atomic number
                edge_index[1]
            ],  # Target atom atomic number
        )

        # Copy embeddings for each edge's source and target nodes
        x_source = x.clone()
        x_target = x.clone()
        x_source._expand_edge(edge_index[0])
        x_target._expand_edge(edge_index[1])

        # Rotate the irreps to align with the edge
        x_source._rotate(SO3_edge_rot, self.lmax_list, self.mmax_list)
        x_target._rotate(SO3_edge_rot, self.lmax_list, self.mmax_list)

        # Compute messages
        x_source = self.so2_block_source(x_source, x_edge, mappingReduced)
        x_target = self.so2_block_target(x_target, x_edge, mappingReduced)

        # Add together the source and target results
        x_target.embedding = x_source.embedding + x_target.embedding

        # Point-wise spherical non-linearity
        x_target._grid_act(self.SO3_grid, self.act, mappingReduced)

        # Rotate back the irreps
        x_target._rotate_inv(SO3_edge_rot, mappingReduced)

        # Compute the sum of the incoming neighboring messages for each target node
        output = x_target.embedding
        output = output.reshape(f_N1, topK, (self.lmax + 1) ** 2, -1)
        output[attn_mask.squeeze(dim=-1)] = 0
        return torch.sum(output, dim=1)

    def forward(
        self,
        node_pos,
        node_irreps_input,
        edge_dis,
        edge_vec,
        attn_weight,  # e.g. rbf(|r_ij|) or relative pos in sequence
        atomic_numbers,
        poly_dist=None,
        attn_mask=None,  # non-adj is True
        batch=None,
        batched_data=None,
        **kwargs,
    ):
        f_N1, topK = attn_weight.shape[:2]
        num_atoms = node_irreps_input.shape[0]

        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        self.SO3_edge_rot = torch.nn.ModuleList()
        for i in range(1):
            self.SO3_edge_rot.append(
                SO3_Rotation(
                    _init_edge_rot_mat(edge_vec.reshape(f_N1 * topK, 3)),
                    self.lmax_list[i],
                )
            )

        #######################for memory saving
        node_irreps_input = self.proj_input(node_irreps_input)
        x = SO3_Embedding(
            num_atoms,
            self.lmax_list,
            self.sphere_channels,
            node_irreps_input.device,
            node_irreps_input.dtype,
        )
        x.embedding = node_irreps_input
        x_embedding = self._forward(
            x,
            atomic_numbers,
            edge_distance=attn_weight,
            edge_index=(
                batched_data["f_sparse_idx_node"].reshape(-1),
                torch.arange(f_N1).reshape(f_N1, -1).repeat(1, topK).reshape(-1),
            ),
            SO3_edge_rot=self.SO3_edge_rot,
            mappingReduced=self.mappingReduced,
            attn_mask=attn_mask,
        )
        x_embedding = self.proj_final(x_embedding)

        return x_embedding, attn_weight


import copy


class MessageBlock_eqv2(torch.nn.Module):
    """
    SO2EquivariantGraphAttention: Perform MLP attention + non-linear message passing
        SO(2) Convolution with radial function -> S2 Activation -> SO(2) Convolution -> attention weights and non-linear messages
        attention weights * non-linear messages -> Linear

    Args:
        sphere_channels (int):      Number of spherical channels
        hidden_channels (int):      Number of hidden channels used during the SO(2) conv
        num_heads (int):            Number of attention heads
        attn_alpha_head (int):      Number of channels for alpha vector in each attention head
        attn_value_head (int):      Number of channels for value vector in each attention head
        output_channels (int):      Number of output channels
        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution

        SO3_rotation (list:SO3_Rotation): Class to calculate Wigner-D matrices and rotate embeddings
        mappingReduced (CoefficientMappingModule): Class to convert l and m indices once node embedding is rotated
        SO3_grid (SO3_grid):        Class used to convert from grid the spherical harmonic representations

        max_num_elements (int):     Maximum number of atomic numbers
        edge_channels_list (list:int):  List of sizes of invariant edge embedding. For example, [input_channels, hidden_channels, hidden_channels].
                                        The last one will be used as hidden size when `use_atom_edge_embedding` is `True`.
        use_atom_edge_embedding (bool): Whether to use atomic embedding along with relative distance for edge scalar features
        use_m_share_rad (bool):     Whether all m components within a type-L vector of one channel share radial function weights

        activation (str):           Type of activation function
        use_s2_act_attn (bool):     Whether to use attention after S2 activation. Otherwise, use the same attention as Equiformer
        use_attn_renorm (bool):     Whether to re-normalize attention weights
        use_gate_act (bool):        If `True`, use gate activation. Otherwise, use S2 activation.
        use_sep_s2_act (bool):      If `True`, use separable S2 activation when `use_gate_act` is False.

        alpha_drop (float):         Dropout rate for attention weights
    """

    def __init__(
        self,
        irreps_node_input="256x0e+256x1e+256x2e",
        attn_weight_input_dim: int = 32,  # e.g. rbf(|r_ij|) or relative pos in sequence
        num_attn_heads: int = 8,
        attn_scalar_head: int = 32,
        irreps_head="32x0e+32x1e+32x2e",
        alpha_drop=0.1,
        rescale_degree=False,
        nonlinear_message=False,
        proj_drop=0.1,
        tp_type="v1",
        attn_type="first-order",  ## second-order
        add_rope=True,
        layer_id=0,
        irreps_origin="1x0e+1x1e+1x2e",
        neighbor_weight=None,
        atom_type_cnt=256,
        use_atom_edge_embedding: bool = True,
        use_m_share_rad: bool = False,
        use_s2_act_attn: bool = False,
        use_gate_act: bool = False,
        use_attn_renorm: bool = True,
        use_sep_s2_act: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.irreps_node_input = (
            o3.Irreps(irreps_node_input)
            if isinstance(irreps_node_input, str)
            else irreps_node_input
        )
        self.scalar_dim = self.irreps_node_input[0][0]  # scalar_dim x 0e

        self.sphere_channels = self.scalar_dim
        self.hidden_channels = self.scalar_dim // 2
        self.num_heads = 8
        self.attn_alpha_channels = self.scalar_dim // 2
        self.attn_value_channels = self.scalar_dim // self.num_heads
        self.output_channels = self.scalar_dim
        self.lmax = len(self.irreps_node_input) - 1
        self.lmax_list = [self.lmax]
        self.mmax_list = [2]
        self.num_resolutions = len(self.lmax_list)

        self.SO3_grid = torch.nn.ModuleList()
        for lval in range(max(self.lmax_list) + 1):
            so3_m_grid = torch.nn.ModuleList()
            for m in range(max(self.lmax_list) + 1):
                so3_m_grid.append(
                    SO3_Grid(
                        lval,
                        m,
                        resolution=18,
                        normalization="component",
                    )
                )

            self.SO3_grid.append(so3_m_grid)
        self.mappingReduced = CoefficientMapping([self.lmax], [2])

        # Create edge scalar (invariant to rotations) features
        # Embedding function of the atomic numbers
        self.max_num_elements = 256
        self.edge_channels_list = copy.deepcopy(
            [
                attn_weight_input_dim,
                min(128, attn_weight_input_dim),
                min(128, attn_weight_input_dim),
            ]
        )
        self.use_atom_edge_embedding = use_atom_edge_embedding
        self.use_m_share_rad = use_m_share_rad

        if self.use_atom_edge_embedding:
            self.source_embedding = nn.Embedding(
                self.max_num_elements, self.edge_channels_list[-1]
            )
            self.target_embedding = nn.Embedding(
                self.max_num_elements, self.edge_channels_list[-1]
            )
            nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
            nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)
            self.edge_channels_list[0] = (
                self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]
            )
        else:
            self.source_embedding, self.target_embedding = None, None

        self.use_s2_act_attn = use_s2_act_attn
        self.use_attn_renorm = use_attn_renorm
        self.use_gate_act = use_gate_act
        self.use_sep_s2_act = use_sep_s2_act

        assert not self.use_s2_act_attn  # since this is not used

        # Create SO(2) convolution blocks
        extra_m0_output_channels = None
        if not self.use_s2_act_attn:
            extra_m0_output_channels = self.num_heads * self.attn_alpha_channels
            if self.use_gate_act:
                extra_m0_output_channels = (
                    extra_m0_output_channels
                    + max(self.lmax_list) * self.hidden_channels
                )
            else:
                if self.use_sep_s2_act:
                    extra_m0_output_channels = (
                        extra_m0_output_channels + self.hidden_channels
                    )

        if self.use_m_share_rad:
            self.edge_channels_list = [
                *self.edge_channels_list,
                2 * self.sphere_channels * (max(self.lmax_list) + 1),
            ]
            self.rad_func = RadialFunction(self.edge_channels_list)
            expand_index = torch.zeros([(max(self.lmax_list) + 1) ** 2]).long()
            for lval in range(max(self.lmax_list) + 1):
                start_idx = lval**2
                length = 2 * lval + 1
                expand_index[start_idx : (start_idx + length)] = lval
            self.register_buffer("expand_index", expand_index)
        from fairchem.core.models.equiformer_v2.activation import (
            GateActivation,
            S2Activation,
            SeparableS2Activation,
        )
        from fairchem.core.models.equiformer_v2.so2_ops import SO2_Convolution

        self.so2_conv_1 = SO2_Convolution(
            2 * self.sphere_channels,
            self.hidden_channels,
            self.lmax_list,
            self.mmax_list,
            self.mappingReduced,
            internal_weights=(bool(self.use_m_share_rad)),
            edge_channels_list=(
                self.edge_channels_list if not self.use_m_share_rad else None
            ),
            extra_m0_output_channels=extra_m0_output_channels,  # for attention weights and/or gate activation
        )

        if self.use_s2_act_attn:
            self.alpha_norm = None
            self.alpha_act = None
            self.alpha_dot = None
        else:
            if self.use_attn_renorm:
                self.alpha_norm = torch.nn.LayerNorm(self.attn_alpha_channels)
            else:
                self.alpha_norm = torch.nn.Identity()
            self.alpha_act = SmoothLeakyReLU()
            self.alpha_dot = torch.nn.Parameter(
                torch.randn(self.num_heads, self.attn_alpha_channels)
            )
            # torch_geometric.nn.inits.glorot(self.alpha_dot) # Following GATv2
            std = 1.0 / math.sqrt(self.attn_alpha_channels)
            torch.nn.init.uniform_(self.alpha_dot, -std, std)

        self.alpha_dropout = None
        if alpha_drop != 0.0:
            self.alpha_dropout = torch.nn.Dropout(alpha_drop)

        if self.use_gate_act:
            self.gate_act = GateActivation(
                lmax=max(self.lmax_list),
                mmax=max(self.mmax_list),
                num_channels=self.hidden_channels,
            )
        else:
            if self.use_sep_s2_act:
                # separable S2 activation
                self.s2_act = SeparableS2Activation(
                    lmax=max(self.lmax_list), mmax=max(self.mmax_list)
                )
            else:
                # S2 activation
                self.s2_act = S2Activation(
                    lmax=max(self.lmax_list), mmax=max(self.mmax_list)
                )

        self.so2_conv_2 = SO2_Convolution(
            self.hidden_channels,
            self.num_heads * self.attn_value_channels,
            self.lmax_list,
            self.mmax_list,
            self.mappingReduced,
            internal_weights=True,
            edge_channels_list=None,
            extra_m0_output_channels=(
                self.num_heads if self.use_s2_act_attn else None
            ),  # for attention weights
        )

        self.proj = SO3_Linear_e2former(
            self.num_heads * self.attn_value_channels,
            self.output_channels,
            lmax=self.lmax,
        )

        # SO3_LinearV2(
        #                 self.num_heads * self.attn_value_channels,
        #     self.output_channels,
        #     lmax=self.lmax_list[0],
        # )

    def forward(
        self,
        node_pos,
        node_irreps_input,
        edge_dis,
        edge_vec,
        attn_weight,  # e.g. rbf(|r_ij|) or relative pos in sequence
        atomic_numbers,
        poly_dist=None,
        attn_mask=None,  # non-adj is True
        batch=None,
        batched_data=None,
        **kwargs,
    ):
        f_N1, topK = attn_weight.shape[:2]
        num_atoms = node_irreps_input.shape[0]

        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        self.SO3_edge_rot = torch.nn.ModuleList()
        for i in range(1):
            self.SO3_edge_rot.append(
                SO3_Rotation(
                    _init_edge_rot_mat(edge_vec.reshape(f_N1 * topK, 3)),
                    self.lmax_list[i],
                )
            )

        #######################for memory saving
        x = SO3_Embedding(
            num_atoms,
            self.lmax_list,
            self.sphere_channels,
            node_irreps_input.device,
            node_irreps_input.dtype,
        )
        x.embedding = node_irreps_input
        x_embedding = self._forward(
            x,
            atomic_numbers,
            edge_distance=attn_weight,
            edge_index=(
                batched_data["f_sparse_idx_node"].reshape(-1),
                torch.arange(f_N1).reshape(f_N1, -1).repeat(1, topK).reshape(-1),
            ),
            SO3_edge_rot=self.SO3_edge_rot,
            mappingReduced=self.mappingReduced,
            attn_mask=attn_mask,
        )

        return x_embedding, attn_weight

    def _forward(
        self,
        x: torch.Tensor,
        atomic_numbers,
        edge_distance: torch.Tensor,
        edge_index,
        SO3_edge_rot,
        mappingReduced,
        attn_mask,
        node_offset: int = 0,
    ):
        f_N1, topK = edge_distance.shape[:2]
        edge_distance = edge_distance.reshape(f_N1 * topK, -1)

        # Compute edge scalar features (invariant to rotations)
        # Uses atomic numbers and edge distance as inputs
        if self.use_atom_edge_embedding:
            source_element = atomic_numbers[edge_index[0]]  # Source atom atomic number
            target_element = atomic_numbers[edge_index[1]]  # Target atom atomic number
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            x_edge = torch.cat(
                (edge_distance, source_embedding, target_embedding), dim=1
            )
        else:
            x_edge = edge_distance

        x_source = x.clone()
        x_target = x.clone()
        # if gp_utils.initialized():
        #     x_full = gp_utils.gather_from_model_parallel_region(x.embedding, dim=0)
        #     x_source.set_embedding(x_full)
        #     x_target.set_embedding(x_full)
        x_source._expand_edge(edge_index[0])
        x_target._expand_edge(edge_index[1])

        x_message_data = torch.cat((x_source.embedding, x_target.embedding), dim=2)
        x_message = SO3_Embedding(
            0,
            x_target.lmax_list.copy(),
            x_target.num_channels * 2,
            device=x_target.device,
            dtype=x_target.dtype,
        )
        x_message.set_embedding(x_message_data)
        x_message.set_lmax_mmax(self.lmax_list.copy(), self.mmax_list.copy())

        # radial function (scale all m components within a type-L vector of one channel with the same weight)
        if self.use_m_share_rad:
            x_edge_weight = self.rad_func(x_edge)
            x_edge_weight = x_edge_weight.reshape(
                -1, (max(self.lmax_list) + 1), 2 * self.sphere_channels
            )
            x_edge_weight = torch.index_select(
                x_edge_weight, dim=1, index=self.expand_index
            )  # [E, (L_max + 1) ** 2, C]
            x_message.embedding = x_message.embedding * x_edge_weight

        # Rotate the irreps to align with the edge
        x_message._rotate(SO3_edge_rot, self.lmax_list, self.mmax_list)
        # print(x_edge.shape,self.use_m_share_rad,x_message.embedding.shape,self.edge_channels_list)
        # First SO(2)-convolution
        if self.use_s2_act_attn:
            x_message = self.so2_conv_1(x_message, x_edge)
        else:
            x_message, x_0_extra = self.so2_conv_1(x_message, x_edge)

        # Activation
        x_alpha_num_channels = self.num_heads * self.attn_alpha_channels
        if self.use_gate_act:
            # Gate activation
            x_0_gating = x_0_extra.narrow(
                1,
                x_alpha_num_channels,
                x_0_extra.shape[1] - x_alpha_num_channels,
            )  # for activation
            x_0_alpha = x_0_extra.narrow(
                1, 0, x_alpha_num_channels
            )  # for attention weights
            x_message.embedding = self.gate_act(x_0_gating, x_message.embedding)
        else:
            if self.use_sep_s2_act:
                x_0_gating = x_0_extra.narrow(
                    1,
                    x_alpha_num_channels,
                    x_0_extra.shape[1] - x_alpha_num_channels,
                )  # for activation
                x_0_alpha = x_0_extra.narrow(
                    1, 0, x_alpha_num_channels
                )  # for attention weights
                x_message.embedding = self.s2_act(
                    x_0_gating, x_message.embedding, self.SO3_grid
                )
            else:
                x_0_alpha = x_0_extra
                x_message.embedding = self.s2_act(x_message.embedding, self.SO3_grid)
            # x_message._grid_act(self.SO3_grid, self.value_act, self.mappingReduced)

        # Second SO(2)-convolution
        if self.use_s2_act_attn:
            x_message, x_0_extra = self.so2_conv_2(x_message, x_edge)
        else:
            x_message = self.so2_conv_2(x_message, x_edge)

        # Attention weights
        if self.use_s2_act_attn:
            alpha = x_0_extra
        else:
            x_0_alpha = x_0_alpha.reshape(-1, self.num_heads, self.attn_alpha_channels)
            x_0_alpha = self.alpha_norm(x_0_alpha)
            x_0_alpha = self.alpha_act(x_0_alpha)
            alpha = torch.einsum("bik, ik -> bi", x_0_alpha, self.alpha_dot)
        # alpha = torch_geometric.utils.softmax(alpha, edge_index[1])

        alpha = alpha.reshape(f_N1, topK, self.num_heads)
        alpha = alpha.masked_fill(attn_mask, -1e6)
        alpha = torch.nn.functional.softmax(alpha, 1)
        alpha = alpha.masked_fill(attn_mask, 0)

        alpha = alpha.reshape(-1, 1, self.num_heads, 1)
        if self.alpha_dropout is not None:
            alpha = self.alpha_dropout(alpha)

        # Attention weights * non-linear messages
        attn = x_message.embedding
        attn = attn.reshape(
            attn.shape[0],
            attn.shape[1],
            self.num_heads,
            self.attn_value_channels,
        )
        attn = attn * alpha
        attn = attn.reshape(
            attn.shape[0],
            attn.shape[1],
            self.num_heads * self.attn_value_channels,
        )
        x_message.embedding = attn

        # Rotate back the irreps
        x_message._rotate_inv(SO3_edge_rot, self.mappingReduced)

        # # Compute the sum of the incoming neighboring messages for each target node
        # x_message._reduce_edge(edge_index[1] - node_offset, len(x.embedding))
        out = torch.sum(
            x_message.embedding.reshape(f_N1, topK, (self.lmax + 1) ** 2, -1), dim=1
        )
        # Project
        return self.proj(out)


class E2former(torch.nn.Module):
    def __init__(
        self,
        irreps_node_embedding="128x0e+64x1e+32x2e",
        num_layers=6,
        pbc_max_radius=5,
        max_neighbors=20,
        max_radius=15.0,
        basis_type="gaussian",
        number_of_basis=128,
        num_attn_heads=4,
        attn_scalar_head=32,
        irreps_head="32x0e+16x1e+8x2e",
        rescale_degree=False,
        nonlinear_message=False,
        norm_layer="layer",  # the default is deprecated
        alpha_drop=0.1,
        proj_drop=0.0,
        out_drop=0.0,
        drop_path_rate=0.1,
        atom_type_cnt=256,
        tp_type=None,
        attn_type="v0",
        edge_embedtype="default",
        attn_biastype="share",  # add
        ffn_type="default",
        add_rope=True,
        time_embed=False,
        sparse_attn=False,
        dynamic_sparse_attn_threthod=1000,
        avg_degree=_AVG_DEGREE,
        force_head=None,
        decouple_EF=False,
        # mean=None,
        # std=None,
        # scale=None,
        # atomref=None,
        **kwargs,
    ):
        super().__init__()
        self.tp_type = tp_type
        self.attn_type = attn_type
        self.pbc_max_radius = pbc_max_radius  #
        self.max_neighbors = max_neighbors
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.alpha_drop = alpha_drop
        self.proj_drop = proj_drop
        self.out_drop = out_drop
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.add_rope = add_rope
        self.time_embed = time_embed
        self.sparse_attn = sparse_attn
        self.dynamic_sparse_attn_threthod = dynamic_sparse_attn_threthod
        # self.task_mean = mean
        # self.task_std = std
        # self.scale = scale
        # self.register_buffer("atomref", atomref)

        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.num_layers = num_layers
        self.num_attn_heads = num_attn_heads
        self.attn_scalar_head = attn_scalar_head
        self.irreps_head = irreps_head
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message
        self.decouple_EF = decouple_EF
        if "0e" not in self.irreps_node_embedding:
            raise ValueError("sorry, the irreps node embedding must have 0e embedding")

        self.unifiedtokentoembedding = nn.Linear(
            self.irreps_node_embedding[0][0], self.irreps_node_embedding[0][0]
        )

        self.default_node_embedding = torch.nn.Embedding(
            atom_type_cnt, self.irreps_node_embedding[0][0]
        )

        self._node_scalar_dim = self.irreps_node_embedding[0][0]
        self._node_vec_dim = (
            self.irreps_node_embedding.dim - self.irreps_node_embedding[0][0]
        )

        ## this is for f( r_ij )
        self.basis_type = basis_type
        self.attn_biastype = attn_biastype
        self.heads2basis = nn.Linear(
            self.num_attn_heads, self.number_of_basis, bias=True
        )
        if self.basis_type == "gaussian":
            self.rbf = GaussianRadialBasisLayer(
                self.number_of_basis, cutoff=self.max_radius
            )
        # elif self.basis_type == "gaussian_edge":
        #     self.rbf = GaussianLayer_Edgetype(
        #         self.number_of_basis, cutoff=self.max_radius
        #     )
        elif self.basis_type == "gaussiansmear":
            self.rbf = GaussianSmearing(
                self.number_of_basis, cutoff=self.max_radius, basis_width_scalar=2
            )
        elif self.basis_type == "bessel":
            self.rbf = RadialBasis(
                self.number_of_basis,
                cutoff=self.max_radius,
                rbf={"name": "spherical_bessel"},
            )
        else:
            raise ValueError

        # edge
        if (
            "default" in edge_embedtype
            or "highorder" in edge_embedtype
            or "elec" in edge_embedtype
        ):
            self.edge_deg_embed_dense = EdgeDegreeEmbeddingNetwork_higherorder(
                self.irreps_node_embedding,
                avg_degree,
                cutoff=self.max_radius,
                number_of_basis=self.number_of_basis,
                time_embed=self.time_embed,
                use_atom_edge=True,
                use_layer_norm="wolayernorm" not in edge_embedtype,
            )
        elif "eqv2" in edge_embedtype:
            self.edge_deg_embed_dense = EdgeDegreeEmbeddingNetwork_eqv2(
                self.irreps_node_embedding,
                avg_degree,
                cutoff=self.max_radius,
                number_of_basis=self.number_of_basis,
                lmax=len(self.irreps_node_embedding) - 1,
                time_embed=self.time_embed,
            )
        else:
            raise ValueError("please check edge embedtype")

        self.blocks = torch.nn.ModuleList()
        for i in range(self.num_layers):
            blk = TransBlock(
                irreps_node_input=self.irreps_node_embedding,
                irreps_node_output=self.irreps_node_embedding,
                attn_weight_input_dim=self.number_of_basis,
                num_attn_heads=self.num_attn_heads,
                attn_scalar_head=self.attn_scalar_head,
                irreps_head=self.irreps_head,
                rescale_degree=self.rescale_degree,
                nonlinear_message=self.nonlinear_message,
                alpha_drop=self.alpha_drop if i != self.num_layers - 1 else 0,
                proj_drop=self.proj_drop if i != self.num_layers - 1 else 0,
                drop_path_rate=self.drop_path_rate if i != self.num_layers - 1 else 0,
                norm_layer=self.norm_layer,
                tp_type=self.tp_type,
                attn_type=self.attn_type,
                ffn_type=ffn_type,
                layer_id=i,
                add_rope=self.add_rope,
                sparse_attn=self.sparse_attn,
                max_radius=max_radius,
            )
            self.blocks.append(blk)

        self.energy_force_block = None
        if self.decouple_EF:
            self.energy_force_block = TransBlock(
                irreps_node_input=self.irreps_node_embedding,
                irreps_node_output=self.irreps_node_embedding,
                attn_weight_input_dim=self.number_of_basis,
                num_attn_heads=self.num_attn_heads,
                attn_scalar_head=self.attn_scalar_head,
                irreps_head=self.irreps_head,
                rescale_degree=self.rescale_degree,
                nonlinear_message=self.nonlinear_message,
                alpha_drop=0,
                proj_drop=0,
                drop_path_rate=0,
                norm_layer=self.norm_layer,
                tp_type=self.tp_type,
                attn_type="first-order",
                ffn_type=ffn_type,
                layer_id=0,
                add_rope=self.add_rope,
                sparse_attn=self.sparse_attn,
                max_radius=max_radius,
            )

        self.scalar_dim = self.irreps_node_embedding[0][0]

        # self.norm_final = get_norm_layer(norm_layer)(
        #     o3.Irreps(f"{self.scalar_dim}x0e+{self.scalar_dim}x1e")
        # )
        self.lmax = len(self.irreps_node_embedding) - 1
        self.norm_tmp = get_normalization_layer(
            norm_layer, lmax=self.lmax, num_channels=self.scalar_dim
        )
        self.norm_final = get_normalization_layer(
            norm_layer, lmax=self.lmax, num_channels=self.scalar_dim
        )
        if len(self.irreps_node_embedding) == 1:
            self.f_linear = nn.Sequential(
                nn.Linear(self.scalar_dim, self.scalar_dim),
                nn.LayerNorm(self.scalar_dim),
                nn.SiLU(),
                nn.Linear(self.scalar_dim, 3 * self.scalar_dim),
            )

        self.apply(self._init_weights)

    def reset_parameters(self):
        warnings.warn("sorry, output model not implement reset parameters")

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            # # if self.weight_init == "normal":
            # std = 1 / math.sqrt(m.in_features)
            # torch.nn.init.normal_(m.weight, 0, std)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    # def _init_rbf_weights(self, m):
    #     if isinstance(m, RadialProfile):
    #         m.apply(self._uniform_init_linear_weights)

    # def _uniform_init_linear_weights(self, m):
    #     if isinstance(m, torch.nn.Linear):
    #         if m.bias is not None:
    #             torch.nn.init.constant_(m.bias, 0)
    #         std = 1 / math.sqrt(m.in_features)
    #         torch.nn.init.uniform_(m.weight, -std, std)

    @torch.jit.ignore
    def no_weight_decay(self):
        return no_weight_decay(self)

    def forward(
        self,
        batched_data: Dict,
        token_embedding: torch.Tensor,
        mixed_attn_bias=None,
        padding_mask: torch.Tensor = None,
        pbc_expand_batched: Optional[Dict] = None,
        time_embed: Optional[torch.Tensor] = None,
        return_node_irreps=False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass of the PSMEncoder class.
        Args:
            x (torch.Tensor): Input tensor, [L, B, H].
            padding_mask (torch.Tensor): Padding mask, [B, L].
            batched_data (Dict): Input data for the forward pass.
            masked_token_type (torch.Tensor): The masked token type, [B, L].
        Returns:
            torch.Tensor: Encoded tensor, [B, L, H].
        example:
        batch: attn_bias torch.Size([4, 65, 65])
        batch: attn_edge_type torch.Size([4, 64, 64, 1])
        batch: spatial_pos torch.Size([4, 64, 64]) -> shortest path
        batch: in_degree torch.Size([4, 64])
        batch: out_degree torch.Size([4, 64])
        batch: token_id torch.Size([4, 64])
        batch: node_attr torch.Size([4, 64, 1])
        batch: edge_input torch.Size([4, 64, 64, 5, 1])
        batch: energy torch.Size([4])
        batch: forces torch.Size([4, 64, 3])
        batch: pos torch.Size([4, 64, 3])
        batch: node_type_edge torch.Size([4, 64, 64, 2])
        batch: pbc torch.Size([4, 3])
        batch: cell torch.Size([4, 3, 3])
        batch: num_atoms torch.Size([4])
        batch: is_periodic torch.Size([4])
        batch: is_molecule torch.Size([4])
        batch: is_protein torch.Size([4])
        batch: protein_masked_pos torch.Size([4, 64, 3])
        batch: protein_masked_aa torch.Size([4, 64])
        batch: protein_mask torch.Size([4, 64, 3])
        batch: init_pos torch.Size([4, 64, 3])
        batch: time_embed torch.Size([4, 64, node_embedding_scalar])

        example:
        import torch
        from sfm.models.psm.equivariant.e2former import E2former
        func = E2former(
                irreps_node_embedding="24x0e+16x1e+8x2e",
                num_layers=6,
                max_radius=5.0,
                basis_type="gaussian",
                number_of_basis=128,
                num_attn_heads=4,
                attn_scalar_head = 32,
                irreps_head="32x0e+16x1e+8x2e",
                alpha_drop=0,)
        batched_data = {}
        B,L = 4,100

        node_pos = torch.randn(B,L,3)*100
        token_embedding = torch.randn(B,L,24)
        token_id = torch.randint(0,100,(B,L))
        batched_data = {
            "pos":node_pos,
            # "token_embedding":token_embedding,
            "token_id":token_id,
        }
        out = func(batched_data, None, None)


        tr = o3._wigner.wigner_D(1,
                            torch.Tensor([0.8]),
                            torch.Tensor([0.5]),
                            torch.Tensor([0.3]))
        pos = (tr@(node_pos.unsqueeze(-1))).squeeze(-1)
        batched_data = {
            "pos":pos,
            # "token_embedding":token_embedding,
            "token_id":token_id,
        }
        out_tr = func(batched_data, None, None)


        print(torch.max(torch.abs(out_tr[0]-out[0])),
            torch.max(torch.abs(out_tr[1]-(tr@out[1]))))
        """

        tensortype = self.default_node_embedding.weight.dtype
        device = padding_mask.device
        B, L = padding_mask.shape[:2]

        node_pos = batched_data["pos"]
        # node_pos.requires_grad = True
        node_pos = torch.where(
            padding_mask.unsqueeze(dim=-1).repeat(1, 1, 3), 999.0, node_pos
        )

        if (time_embed is not None) and self.time_embed:
            time_embed = time_embed.to(dtype=tensortype)
        else:
            time_embed = None

        node_mask = logical_not(padding_mask)
        atomic_numbers = batched_data["masked_token_type"].reshape(B, L)[node_mask]
        ptr = torch.cat(
            [
                torch.Tensor(
                    [
                        0,
                    ]
                )
                .int()
                .to(device),
                torch.cumsum(torch.sum(node_mask, dim=-1), dim=-1),
            ],
            dim=0,
        )
        f_node_pos = node_pos[node_mask]
        f_N1 = f_node_pos.shape[0]
        f_batch = torch.arange(B).reshape(B, 1).repeat(1, L).to(device)[node_mask]

        # expand_node_mask = node_mask
        expand_node_pos = node_pos
        expand_ptr = ptr
        outcell_index = torch.arange(L).unsqueeze(dim=0).repeat(B, 1).to(device)
        f_exp_node_pos = f_node_pos
        f_outcell_index = torch.arange(len(f_node_pos)).to(device)
        mol_type = 0  # torch.any(batched_data["is_molecule"]):
        L2 = L
        if torch.any(batched_data["pbc"]):
            mol_type = 1
            #  batched_data["outcell_index"] # B*L2
            # batched_data["outcell_index_0"] # B*L2
            # batched_data.update(pbc_expand_batched)
            L2 = pbc_expand_batched["outcell_index"].shape[1]
            outcell_index = pbc_expand_batched["outcell_index"]
            # outcell_index_0 = (torch.arange(B).reshape(B, 1).repeat(1,batched_data["outcell_index"].shape[1] ))
            expand_node_pos = pbc_expand_batched["expand_pos"].float()
            expand_node_pos[
                pbc_expand_batched["expand_mask"]
            ] = 999  # set expand node pos padding to 9999
            expand_node_mask = logical_not(pbc_expand_batched["expand_mask"])
            expand_ptr = torch.cat(
                [
                    torch.Tensor(
                        [
                            0,
                        ]
                    )
                    .int()
                    .to(device),
                    torch.cumsum(torch.sum(expand_node_mask, dim=-1), dim=-1),
                ],
                dim=0,
            )
            f_exp_node_pos = expand_node_pos[expand_node_mask]
            f_outcell_index = (outcell_index + ptr[:B, None])[
                expand_node_mask
            ]  # e.g. n1*hidden [flatten_outcell_index]  -> n2*hidden

        # print(f_outcell_index.shape)
        # if torch.max(f_outcell_index)>=ptr[-1]:
        #     raise ValueError("sorry please check your code")
        # if torch.any(batched_data["is_protein"]):
        #     mol_type = 2
        batched_data["mol_type"] = mol_type

        edge_vec = node_pos.unsqueeze(2) - expand_node_pos.unsqueeze(1)
        dist = torch.norm(edge_vec, dim=-1)  # B*L*L Attention: ego-connection is 0 here
        dist = torch.where(dist < 1e-4, 1000, dist)
        # dist_embedding = self.rbf(dist.reshape(-1)).reshape(B, L, L2, self.number_of_basis)  # [B, L, L, number_of_basis]
        _, neighbor_indices = dist.sort(dim=-1)
        topK = min(L2, self.max_neighbors)
        neighbor_indices = neighbor_indices[:, :, :topK]  # Shape: B*L*K
        # neighbor_indices = torch.arange(topK).reshape(1,1,topK).repeat(B,L,1).to(device)
        # neighbor_indices = torch.arange(K).to(device).reshape(1,1,K).repeat(B,L,1)
        dist = torch.gather(dist, dim=-1, index=neighbor_indices)  # Shape: B*L*topK
        attn_mask = (
            dist > (self.max_radius if mol_type != 1 else self.pbc_max_radius)
        ) | (dist < 1e-4)
        attn_mask = attn_mask[node_mask].unsqueeze(dim=-1)
        # print("sum of attn mask: ",attn_mask.shape, torch.mean(torch.sum(attn_mask,dim = (-2,-1)).float()))
        f_dist = dist[node_mask]  # flattn_N* topK*
        f_dist_embedding = self.rbf(f_dist)  # flattn_N* topK* self.number_of_basis)
        poly_dist = polynomial(
            f_dist, self.max_radius if mol_type != 1 else self.pbc_max_radius
        )

        f_sparse_idx_node = (
            torch.gather(
                outcell_index.unsqueeze(1).repeat(1, L, 1), 2, neighbor_indices
            )
            + ptr[:B, None, None]
        )[node_mask]
        f_sparse_idx_node = torch.clamp(f_sparse_idx_node, max=ptr[B] - 1)
        f_sparse_idx_expnode = (neighbor_indices + expand_ptr[:B, None, None])[
            node_mask
        ]
        f_sparse_idx_expnode = torch.clamp(f_sparse_idx_expnode, max=expand_ptr[B] - 1)
        f_edge_vec = f_node_pos.unsqueeze(dim=1) - f_exp_node_pos[f_sparse_idx_expnode]

        # print(torch.max(f_sparse_idx_node),torch.max(f_sparse_idx_expnode),torch.max(ptr),torch.max(expand_ptr))
        # # this line could use to check the index's correctness
        # batch_indices = torch.arange(B).unsqueeze(1).unsqueeze(2).expand(B, L, topK)
        # test_edge_vec = (node_pos[:,:L].unsqueeze(dim = 2)-expand_node_pos[batch_indices,neighbor_indices])[node_mask]
        # print('test edge vec ',torch.sum(torch.abs(edge_vec-test_edge_vec)[~attn_mask.squeeze()]))

        # # this line could use to check the index's correctness
        # batch_indices = torch.arange(B).unsqueeze(1).unsqueeze(2).expand(B, L, topK)
        # test_src_ne = atomic_numbers[(torch.arange(B).reshape(B, 1).repeat(1,L2)),
        #                              outcell_index][batch_indices,neighbor_indices][node_mask]
        # src_ne = atomic_numbers[node_mask][flatten_sparse_indices_innode]
        # print('test atomic numbers',torch.sum(torch.abs(test_src_ne-src_ne)[~attn_mask.squeeze()]))

        # node_mask is used for node_embedding -> f_N*hidden
        # f_node_irreps = token_embedding[node_mask]
        if token_embedding is not None:
            f_atom_embedding = self.unifiedtokentoembedding(
                token_embedding[node_mask]
            )  # [L, B, D] => [B, L, D]
        else:
            f_atom_embedding = self.default_node_embedding(atomic_numbers)

        _edge_inter_mask = logical_not(attn_mask).reshape(
            attn_mask.shape[0], 1, topK
        ) + logical_not(attn_mask).reshape(attn_mask.shape[0], topK, 1)
        _edge_inter_mask[:, torch.arange(topK), torch.arange(topK)] = True
        edge_inter_mask = torch.zeros(f_N1, topK, topK, device=token_embedding.device)
        edge_inter_mask = edge_inter_mask.masked_fill(~_edge_inter_mask, float("-inf"))
        edge_inter_mask = (
            edge_inter_mask.unsqueeze(dim=1)
            .repeat(1, self.num_attn_heads, 1, 1)
            .reshape(-1, topK, topK)
        )

        coeffs = E2TensorProductArbitraryOrder.get_coeffs()
        Y_powers = [
            coeffs[0] * torch.ones_like(f_node_pos.narrow(-1, 0, 1).unsqueeze(dim=-1))
        ]
        # Y is pos. Precompute spherical harmonics for all orders
        for i in range(1, self.lmax + 1):
            Y_powers.append(
                coeffs[i]
                * e3nn.o3.spherical_harmonics(
                    i, f_node_pos, normalize=False, normalization="integral"
                ).unsqueeze(-1)
            )

        exp_Y_powers = [
            coeffs[0]
            * torch.ones_like(f_exp_node_pos.narrow(-1, 0, 1).unsqueeze(dim=-1))
        ]
        # Y is pos. Precompute spherical harmonics for all orders
        for i in range(1, self.lmax + 1):
            exp_Y_powers.append(
                coeffs[i]
                * e3nn.o3.spherical_harmonics(
                    i, f_exp_node_pos, normalize=False, normalization="integral"
                ).unsqueeze(-1)
            )

        edge_vec_powers = [
            coeffs[0] * torch.ones_like(f_edge_vec.narrow(-1, 0, 1).unsqueeze(dim=-1))
        ]
        # Y is pos. Precompute spherical harmonics for all orders
        for i in range(1, self.lmax + 1):
            edge_vec_powers.append(
                e3nn.o3.spherical_harmonics(
                    i, f_edge_vec, normalize=True, normalization="integral"
                ).unsqueeze(-1)
            )
        edge_vec_powers = torch.cat(edge_vec_powers, dim=-2)

        batched_data.update(
            {
                "f_sparse_idx_node": f_sparse_idx_node,
                "f_sparse_idx_expnode": f_sparse_idx_expnode,
                "f_exp_node_pos": f_exp_node_pos,
                "f_outcell_index": f_outcell_index,
                "edge_inter_mask": edge_inter_mask,  # used for escaip attention
                "Y_powers": Y_powers,
                "exp_Y_powers": exp_Y_powers,
                "edge_vec_powers": edge_vec_powers,
            }
        )
        # f_N1 = f_node_pos.shape[0]
        # edge_mask = ~attn_mask.reshape(-1)
        # data = {"pos":f_node_pos,
        #         "atomic_numbers":atomic_numbers,
        #         "batch":f_batch,
        #         "natoms":torch.sum(node_mask,dim = -1),
        #         "node_offset":0,
        #         "atomic_numbers_full":atomic_numbers,
        #         "batch_full":f_batch,
        #         "edge_index":torch.stack([f_sparse_idx_node.reshape(-1).to(device)[edge_mask],
        #                       torch.arange(f_N1).reshape(f_N1,-1).repeat(1,topK).reshape(-1).to(device)[edge_mask]
        #                         ],dim = 0),
        #         "edge_distance":f_dist.reshape(f_N1*topK)[edge_mask],
        #         "edge_distance_vec":f_edge_vec.reshape(f_N1*topK,3)[edge_mask],

        #         }
        # return self.decoder(Data(**data))
        # if torch.any(torch.isnan(atom_embedding)):assert(False)
        # not use sparse mode
        edge_degree_embedding_dense = self.edge_deg_embed_dense(
            f_atom_embedding,
            f_node_pos,
            f_dist,
            edge_scalars=f_dist_embedding,
            edge_vec=f_edge_vec,
            batch=None,
            attn_mask=attn_mask,
            atomic_numbers=atomic_numbers,
            batched_data=batched_data,
            time_embed=time_embed,
        )

        f_node_irreps = edge_degree_embedding_dense
        # node_irreps = torch.zeros(B,L,9,self.irreps_node_embedding[0][0],device = device)
        f_node_irreps[:, 0, :] = f_node_irreps[:, 0, :] + f_atom_embedding
        node_irreps_his = torch.zeros(
            (B, L, (self.lmax + 1) ** 2, self._node_scalar_dim), device=device
        )

        for i, blk in enumerate(self.blocks):
            f_node_irreps, f_dist_embedding = blk(
                node_pos=f_node_pos,
                node_irreps=f_node_irreps,
                edge_dis=f_dist,
                poly_dist=poly_dist,
                edge_vec=f_edge_vec,
                attn_weight=f_dist_embedding,
                atomic_numbers=atomic_numbers,
                attn_mask=attn_mask,
                batched_data=batched_data,
                add_rope=self.add_rope,
                sparse_attn=self.sparse_attn,
                batch=f_batch,  #
            )
            if i == len(self.blocks) - 2:
                node_irreps_his[node_mask] = self.norm_tmp(
                    f_node_irreps
                )  # the part of order 0

            # if torch.any(torch.isnan(node_irreps)):assert(False)

        f_node_irreps_final = self.norm_final(f_node_irreps)
        node_irreps = torch.zeros(
            (B, L, (self.lmax + 1) ** 2, self._node_scalar_dim), device=device
        )
        node_irreps[node_mask] = f_node_irreps  # the part of order 0

        node_attr = torch.zeros((B, L, self._node_scalar_dim), device=device)
        node_vec = torch.zeros((B, L, 3, self._node_scalar_dim), device=device)
        if not self.decouple_EF:
            node_attr[node_mask] = f_node_irreps_final[:, 0]
            node_vec[node_mask] = f_node_irreps_final[:, 1:4]  # the part of order 0
        else:
            node_attr[node_mask] = self.energy_force_block.ffn_s2(f_node_irreps_final)[
                :, 0
            ]
            node_vec[node_mask] = self.energy_force_block.ga(
                node_pos=f_node_pos,
                node_irreps_input=f_node_irreps_final,
                edge_dis=f_dist,
                poly_dist=poly_dist,
                edge_vec=f_edge_vec,
                attn_weight=f_dist_embedding,
                atomic_numbers=atomic_numbers,
                attn_mask=attn_mask,
                batched_data=batched_data,
                add_rope=self.add_rope,
                sparse_attn=self.sparse_attn,
            )[0][:, 1:4]
        if return_node_irreps:
            return node_attr, node_vec, node_irreps, node_irreps_his

        return node_attr, node_vec
