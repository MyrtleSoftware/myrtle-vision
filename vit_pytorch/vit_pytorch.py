from contextlib import nullcontext
from typing import Optional
from typing import Union

import torch
import torch.autograd.profiler as profiler
from quantize import ModelQuantizer
from quantize import QFormat
from torch import nn
from torch.quantization import DeQuantStub
from torch.quantization import QuantStub

MIN_NUM_PATCHES = 16


class Residual(nn.Module):
    def __init__(
        self,
        fn: nn.Module,
    ):
        super().__init__()
        self.fn = fn
        self.res_add = torch.nn.quantized.FloatFunctional()

    def forward(self, x: torch.Tensor):
        return self.res_add.add(self.fn(x), x)


class PreNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        fn: nn.Module,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor):
        return self.fn(self.norm(x))


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

        self.dequant_qkv = DeQuantStub()
        self.quant_out = QuantStub()
        # Add Identity layer in order to attach forward hook and build
        # attention maps
        self.attn_output = nn.Identity()

    def forward(self, x: torch.Tensor):
        b_dim, n_dim, c_dim = x.shape
        qkv = self.dequant_qkv(self.to_qkv(x))
        qkv = qkv.reshape(
            b_dim, n_dim, 3, self.heads, c_dim // self.heads
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_output(attn)

        out = (attn @ v).transpose(1, 2).reshape(b_dim, n_dim, c_dim)
        out = self.quant_out(out)
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float,
        profile: bool,
    ):
        super().__init__()
        # Create context managers when profiling model
        if profile:
            self.cm_attention = profiler.record_function(
                "transformer:attention"
            )
            self.cm_feedforward = profiler.record_function(
                "transformer:feedforward"
            )
        else:
            self.cm_attention = nullcontext()
            self.cm_feedforward = nullcontext()

        # Create list of Transformer blocks
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.Sequential(
                    Residual(
                        PreNorm(
                            dim,
                            Attention(
                                dim,
                                heads=heads,
                                dim_head=dim_head,
                                dropout=dropout,
                            ),
                        )
                    ),
                    Residual(
                        PreNorm(
                            dim,
                            FeedForward(
                                dim,
                                mlp_dim,
                                dropout=dropout,
                            ),
                        )
                    ),
                )
            )

    def forward(self, x: torch.Tensor):
        for transformer_block in self.layers:
            with self.cm_attention:
                x = transformer_block[0](x)
            with self.cm_feedforward:
                x = transformer_block[1](x)
        return x


class ViT(nn.Module):
    def __init__(
        self,
        *,
        image_size: int,
        patch_size: int,
        num_classes: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        pool: str = "cls",
        channels: int = 3,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        profile: bool = False,
        q_format: Optional[Union[str, QFormat]] = None,
    ):
        super().__init__()
        assert (
            image_size % patch_size == 0
        ), "Image dimensions must be divisible by the patch size."
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert num_patches > MIN_NUM_PATCHES, (
            f"your number of patches ({num_patches}) is way too small for "
            f"attention to be effective (at least 16). Try decreasing your "
            f"patch size"
        )
        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"
        self.patch_size = patch_size

        # Create context managers when profiling model
        if profile:
            self.cm_patch_to_embedding = profiler.record_function(
                "patch_to_embedding"
            )
            self.cm_transformer = profiler.record_function("transformer")
            self.cm_mlp_head = profiler.record_function("mlp_head")
        else:
            self.cm_patch_to_embedding = nullcontext()
            self.cm_transformer = nullcontext()
            self.cm_mlp_head = nullcontext()

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            profile,
        )

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
        )

        self.quant_img = QuantStub()
        self.quant_pos_embedding = QuantStub()
        self.quant_cls_token = QuantStub()
        self.dequant_output = DeQuantStub()
        self.cls_token_cat = torch.nn.quantized.FloatFunctional()
        self.pos_embedding_add = torch.nn.quantized.FloatFunctional()
        self.quantizer = ModelQuantizer(self)
        self.quantizer.prepare_qat(
            q_format if q_format is not None else QFormat.FP32
        )

    def forward(self, img: torch.Tensor):
        b_dim, c_dim, h_dim, w_dim = img.shape
        p = self.patch_size

        x = (
            img.reshape(b_dim, c_dim, h_dim // p, p, w_dim // p, p)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(b_dim, (h_dim // p) * (w_dim // p), p * p * c_dim)
        )
        x = self.quant_img(x)
        with self.cm_patch_to_embedding:
            x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        # Add lass token at the beginning of the input sequence
        cls_tokens = self.cls_token.repeat(b, 1, 1)
        cls_tokens = self.quant_cls_token(cls_tokens)
        x = self.cls_token_cat.cat((cls_tokens, x), dim=1)
        # Add the positional embedding
        x = self.pos_embedding_add.add(
            x,
            self.quant_pos_embedding(
                self.pos_embedding.repeat(x.size(0), 1, 1)
            ),
        )
        x = self.dropout(x)

        with self.cm_transformer:
            x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)

        with self.cm_mlp_head:
            output = self.mlp_head(x)
        output = self.dequant_output(output)

        return output

    def convert(self) -> None:
        self.quantizer.convert()
