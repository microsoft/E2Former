# -*- coding: utf-8 -*-
import logging
import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

# from sfm.logging import logging


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class FairseqDropout(nn.Module):
    def __init__(self, p, module_name=None):
        super().__init__()
        self.p = p
        self.module_name = module_name
        self.apply_during_inference = False

    def forward(self, x, inplace: bool = False):
        if self.p > 0 and (self.training or self.apply_during_inference):
            return torch.nn.functional.dropout(
                x, p=self.p, training=True, inplace=inplace
            )
        else:
            return x

    def make_generation_fast_(
        self,
        name: str,
        retain_dropout: bool = False,
        retain_dropout_modules: Optional[List[str]] = None,
        **kwargs,
    ):
        if retain_dropout:
            if retain_dropout_modules is not None and self.module_name is None:
                logging.warning(
                    "Cannot enable dropout during inference for module {} "
                    "because module_name was not set".format(name)
                )
            elif (
                retain_dropout_modules is None  # if None, apply to all modules
                or self.module_name in retain_dropout_modules
            ):
                logging.info(
                    "Enabling dropout during inference for module: {}".format(name)
                )
                self.apply_during_inference = True
            else:
                logging.info("Disabling dropout for module: {}".format(name))


def quant_noise(module, p, block_size):
    """
    Wraps modules and applies quantization noise to the weights for
    subsequent quantization with Iterative Product Quantization as
    described in "Training with Quantization Noise for Extreme Model Compression"

    Args:
        - module: nn.Module
        - p: amount of Quantization Noise
        - block_size: size of the blocks for subsequent quantization with iPQ

    Remarks:
        - Module weights must have the right sizes wrt the block size
        - Only Linear, Embedding and Conv2d modules are supported for the moment
        - For more detail on how to quantize by blocks with convolutional weights,
          see "And the Bit Goes Down: Revisiting the Quantization of Neural Networks"
        - We implement the simplest form of noise here as stated in the paper
          which consists in randomly dropping blocks
    """

    # if no quantization noise, don't register hook
    if p <= 0:
        return module

    # supported modules
    assert isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d))

    # test whether module.weight has the right sizes wrt block_size
    is_conv = module.weight.ndim == 4

    # 2D matrix
    if not is_conv:
        assert (
            module.weight.size(1) % block_size == 0
        ), "Input features must be a multiple of block sizes"

    # 4D matrix
    else:
        # 1x1 convolutions
        if module.kernel_size == (1, 1):
            assert (
                module.in_channels % block_size == 0
            ), "Input channels must be a multiple of block sizes"
        # regular convolutions
        else:
            k = module.kernel_size[0] * module.kernel_size[1]
            assert k % block_size == 0, "Kernel size must be a multiple of block size"

    def _forward_pre_hook(mod, input):
        # no noise for evaluation
        if mod.training:
            if not is_conv:
                # gather weight and sizes
                weight = mod.weight
                in_features = weight.size(1)
                out_features = weight.size(0)

                # split weight matrix into blocks and randomly drop selected blocks
                mask = torch.zeros(
                    in_features // block_size * out_features, device=weight.device
                )
                mask.bernoulli_(p)
                mask = mask.repeat_interleave(block_size, -1).view(-1, in_features)

            else:
                # gather weight and sizes
                weight = mod.weight
                in_channels = mod.in_channels
                out_channels = mod.out_channels

                # split weight matrix into blocks and randomly drop selected blocks
                if mod.kernel_size == (1, 1):
                    mask = torch.zeros(
                        int(in_channels // block_size * out_channels),
                        device=weight.device,
                    )
                    mask.bernoulli_(p)
                    mask = mask.repeat_interleave(block_size, -1).view(-1, in_channels)
                else:
                    mask = torch.zeros(
                        weight.size(0), weight.size(1), device=weight.device
                    )
                    mask.bernoulli_(p)
                    mask = (
                        mask.unsqueeze(2)
                        .unsqueeze(3)
                        .repeat(1, 1, mod.kernel_size[0], mod.kernel_size[1])
                    )

            # scale weights and apply mask
            mask = mask.to(
                torch.bool
            )  # x.bool() is not currently supported in TorchScript
            s = 1 / (1 - p)
            mod.weight.data = s * weight.masked_fill(mask, 0)

    module.register_forward_pre_hook(_forward_pre_hook)
    return module


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin, seq_len):
    if len(x.shape) == 3:
        cos = cos[:, :seq_len, :]
        sin = sin[:, :seq_len, :]

        return (x * cos) + (rotate_half(x) * sin)
    elif len(x.shape) == 4:
        cos = cos[:, None, :seq_len, :]
        sin = sin[:, None, :seq_len, :]

        return (x * cos) + (rotate_half(x) * sin)
    else:
        raise ValueError(
            "Input tensor must have 3 or 4 dimensions, but got {}".format(x.shape)
        )


class SFMRotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        dim,
        max_position_embeddings=16384,
        base=500000,
        # base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device)
                / self.dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=True)
        self.max_seq_len_cached = max_position_embeddings

    def apply_rotary_pos_emb(self, q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        """Applies Rotary Position Embedding to the query and key tensors.

        Args:
            q (`torch.Tensor`): The query tensor.
            k (`torch.Tensor`): The key tensor.
            cos (`torch.Tensor`): The cosine part of the rotary embedding.
            sin (`torch.Tensor`): The sine part of the rotary embedding.
            position_ids (`torch.Tensor`, *optional*):
                Deprecated and unused.
            unsqueeze_dim (`int`, *optional*, defaults to 1):
                The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
                sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
                that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
                k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
                cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
                the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
        Returns:
            `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
        """
        # cos = cos.unsqueeze(unsqueeze_dim)
        # sin = sin.unsqueeze(unsqueeze_dim)
        sLen = q.shape[-2]
        q_embed = (q * cos[:, :sLen, :]) + (rotate_half(q) * sin[:, :sLen, :])
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    def forward(self, q, k, v, position_ids=None, nhead=1):
        """
        Args:
            q: [bs*num_attention_heads, tgt_len, head_size]
            k: [bs*num_attention_heads, seq_len, head_size]
            v: [bs*num_attention_heads, seq_len, head_size]
            position_ids: [bs, seq_len]
        return:
            q: [bs*num_attention_heads, tgt_len, head_size]
            k: [bs*num_attention_heads, seq_len, head_size]
        """
        with torch.no_grad():
            if position_ids is None:
                position_ids = (
                    torch.arange(v.shape[-2], device=q.device)
                    .type_as(self.inv_freq)
                    .unsqueeze(0)
                    .repeat(v.shape[0], 1)
                )
            else:
                max_seq_len = position_ids.size()[-1]
                position_ids = (
                    position_ids.unsqueeze(1)
                    .repeat(1, nhead, 1)
                    .reshape(-1, max_seq_len)
                )

            # x: [bs, num_attention_heads, seq_len, head_size]
            inv_freq_expanded = (
                self.inv_freq[None, :, None]
                .float()
                .expand(position_ids.shape[0], -1, 1)
            )
            position_ids_expanded = position_ids[:, None, :].float()
            device_type = v.device.type
            device_type = (
                device_type
                if isinstance(device_type, str) and device_type != "mps"
                else "cpu"
            )
            with torch.autocast(device_type=device_type, enabled=False):
                freqs = (
                    inv_freq_expanded.float() @ position_ids_expanded.float()
                ).transpose(1, 2)
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos()
                sin = emb.sin()

            cos, sin = cos.to(dtype=v.dtype), sin.to(dtype=v.dtype)

        q, k = self.apply_rotary_pos_emb(q, k, cos, sin)
        return q, k


class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        self_attention=True,
        q_noise=0.0,
        qn_block_size=8,
        d_tilde=1,
        k_bias=False,
        q_bias=True,
        v_bias=True,
        o_bias=True,
        add_rope=False,
        layer_norm=False,
        use_smooth_softmax=False,
        smooth_factor=0.0,
        use_no_pre_cutoff_softmax=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = (
            (self.head_dim / d_tilde) ** 0.5
        ) / self.head_dim  # when d_tilt == 1, match with original transformer scale

        self.self_attention = self_attention

        assert self.self_attention, "Only support self attention"

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        self.k_proj = quant_noise(
            nn.Linear(self.kdim, embed_dim, bias=k_bias), q_noise, qn_block_size
        )
        self.v_proj = quant_noise(
            nn.Linear(self.vdim, embed_dim, bias=q_bias), q_noise, qn_block_size
        )
        self.q_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=v_bias), q_noise, qn_block_size
        )

        self.out_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=o_bias), q_noise, qn_block_size
        )

        if layer_norm:
            self.layer_norm = torch.nn.LayerNorm(embed_dim)
        else:
            self.layer_norm = None

        self.reset_parameters(d_tilde)

        self.onnx_trace = False

        self.rot_emb = None
        if add_rope:
            self.rot_emb = SFMRotaryEmbedding(dim=self.head_dim)

        self.use_smooth_softmax = use_smooth_softmax
        self.smooth_factor = smooth_factor
        self.use_no_pre_cutoff_softmax = use_no_pre_cutoff_softmax

    def prepare_for_onnx_export_(self):
        raise NotImplementedError

    def reset_parameters(self, d_tilde=1):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(
                self.k_proj.weight, gain=1.0 / (math.sqrt(2 * d_tilde))
            )
            nn.init.xavier_uniform_(
                self.v_proj.weight, gain=1.0 / (math.sqrt(2 * d_tilde))
            )
            nn.init.xavier_uniform_(
                self.q_proj.weight, gain=1.0 / (math.sqrt(2 * d_tilde))
            )
        else:
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1.0 / math.sqrt(d_tilde))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1.0 / math.sqrt(d_tilde))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1.0 / math.sqrt(d_tilde))

        nn.init.xavier_uniform_(self.out_proj.weight, gain=1.0 / math.sqrt(d_tilde))
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.layer_norm is not None:
            self.layer_norm.reset_parameters()

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    def forward(
        self,
        query,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        attn_bias: Optional[Tensor] = None,
        local_attention_weight: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
        pbc_expand_batched: Optional[Dict[str, torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
        is_protein: Optional[torch.Tensor] = None,
        math_kernel: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()

        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert src_len, bsz == value.shape[:2]

        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)

        q *= self.scaling

        if pbc_expand_batched is not None:
            outcell_index = pbc_expand_batched["outcell_index"]
            expand_mask = pbc_expand_batched["expand_mask"]
            key_padding_mask = expand_mask

            outcell_index = (
                outcell_index.transpose(1, 0).unsqueeze(-1).expand(-1, -1, embed_dim)
            )
            k = torch.gather(k, dim=0, index=outcell_index)
            v = torch.gather(v, dim=0, index=outcell_index)

            # k = torch.cat([k, expand_k], dim=0)
            # v = torch.cat([v, expand_v], dim=0)

            src_len = k.size()[0]
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len
        else:
            outcell_index = None
            expand_mask = None
            key_padding_mask = (
                key_padding_mask  # torch.cat([key_padding_mask, expand_mask], dim=1)
            )
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        if k is not None:
            k = (
                k.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        assert k is not None
        assert k.size(1) == src_len

        # # This is part of a workaround to get around fork/join parallelism
        # # not supporting Optional types.
        # if key_padding_mask is not None and key_padding_mask.dim() == 0:
        #     key_padding_mask = None

        # add rope
        if self.rot_emb and is_protein.any():
            is_protein = (
                is_protein.unsqueeze(1)
                .repeat(1, self.num_heads, 1)
                .view(bsz * self.num_heads, tgt_len, 1)
            )
            q_rope, k_rope = self.rot_emb(q, k, v, position_ids, self.num_heads)
            q = torch.where(is_protein, q_rope, q)
            k = torch.where(is_protein, k_rope, k)

        # if key_padding_mask is not None:
        #     if outcell_index is not None:
        #         assert expand_mask is not None
        #         key_padding_mask = expand_mask #torch.cat([key_padding_mask, expand_mask], dim=1)
        #     assert key_padding_mask.size(0) == bsz
        #     assert key_padding_mask.size(1) == src_len

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_bias is not None:
            attn_weights += attn_bias.contiguous().view(
                bsz * self.num_heads, tgt_len, src_len
            )

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if local_attention_weight is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if self.use_smooth_softmax:
                attn_weights = (
                    attn_weights + self.smooth_factor
                ) * local_attention_weight.unsqueeze(1) - self.smooth_factor
            elif self.use_no_pre_cutoff_softmax:
                pass
            else:
                attn_weights = attn_weights.masked_fill(
                    local_attention_weight.unsqueeze(1) <= 1e-5, float("-inf")
                )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = nn.functional.softmax(attn_weights, dim=-1)

        if local_attention_weight is not None:
            attn_weights_float = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights_float = attn_weights_float * local_attention_weight.unsqueeze(
                1
            )
            attn_weights_float = attn_weights_float.view(
                bsz * self.num_heads, tgt_len, src_len
            )

        attn_weights = attn_weights_float.type_as(attn_weights)

        attn_probs = self.dropout_module(attn_weights)

        assert v is not None
        attn = torch.bmm(attn_probs, v)

        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)

        if self.layer_norm is not None:
            attn = self.layer_norm(attn)

        attn = self.out_proj(attn)

        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights


class E2DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
        self,
        embedding_dim,
        ffn_embedding_dim,
        num_attention_heads,
        dropout=0.1,
        use_smooth_softmax=False,
        use_no_pre_cutoff_softmax=True,  # tobetest
        smooth_factor=20,
        only_use_rotary_embedding_for_protein=True,
        **kwargs,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

        # NOTICE
        self.attn = MultiheadAttention(
            embedding_dim,
            num_attention_heads,
            dropout=dropout,
            k_bias=False,
            q_bias=False,
            v_bias=False,
            o_bias=False,
            add_rope=True,
            use_no_pre_cutoff_softmax=use_no_pre_cutoff_softmax,
            use_smooth_softmax=use_smooth_softmax,
            smooth_factor=smooth_factor,
        )

        self.norm2 = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, ffn_embedding_dim, bias=False),
            nn.SiLU(),
            nn.LayerNorm(ffn_embedding_dim, eps=1e-6),
            nn.Linear(ffn_embedding_dim, embedding_dim, bias=False),
        )
        self.adaLN_modulation = nn.Sequential(
            # nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim, bias=False),
            nn.SiLU(),
            nn.LayerNorm(embedding_dim, eps=1e-6),
            nn.Linear(embedding_dim, 6 * embedding_dim, bias=False),
        )
        self.adjust_dit_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim, bias=False),
        )
        self.adjust_dit_inverse = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim, bias=False),
            nn.LayerNorm(embedding_dim, eps=1e-6),
        )

    def forward(
        self,
        x,
        c,
        padding_mask,
        batched_data,
        local_attention_weight=None,
        pbc_expand_batched=None,
        mixed_attn_bias=None,
        # ifbackprop=False,
    ):
        # math_kernel = ifbackprop and pbc_expand_batched is not None

        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c).chunk(6, dim=2)
        # if x.shape[-1] != self.psm_config.embedding_dim:
        x = self.adjust_dit_forward(x)
        x = x + gate_msa * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa).transpose(0, 1),
            key_padding_mask=padding_mask,
            is_protein=batched_data["is_protein"],
            position_ids=batched_data["position_ids"],
            pbc_expand_batched=pbc_expand_batched,
            attn_bias=mixed_attn_bias,
            local_attention_weight=local_attention_weight,
            # math_kernel=math_kernel,
        )[0].transpose(0, 1)

        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = self.adjust_dit_inverse(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda")
    model = E2DiTBlock(
        embedding_dim=384,
        ffn_embedding_dim=384,
        num_attention_heads=8,
        dropout=0.1,
        use_smooth_softmax=False,
        smooth_factor=20,
        only_use_rotary_embedding_for_protein=True,
    ).to(device)

    B = 4
    N = 128
    model(
        torch.randn(B, N, 384).to(device),
        torch.randn(B, N, 384).to(device),
        (torch.randn(B, N) > 0).to(device),
        batched_data={
            "is_protein": (torch.ones(B, N) > 0).to(device),
            "position_ids": torch.randint(
                0,
                100,
                (
                    B,
                    N,
                ),
            ).to(device),
        },
        pbc_expand_batched=None,
        mixed_attn_bias=None,
        ifbackprop=True,
    )
