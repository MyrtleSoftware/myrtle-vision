from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.quantization import DeQuantStub
from torch.quantization import QuantStub
from myrtle_vision.models.vit_pytorch import ViT


class DistillableViT(ViT):
    def __init__(self, *args, **kwargs):
        super(DistillableViT, self).__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs
        self.dim = kwargs["dim"]
        self.num_classes = kwargs["num_classes"]
        self.quant_distill_token = QuantStub()
        self.distill_token_cat = torch.nn.quantized.FloatFunctional()
        self.dequant_out = DeQuantStub()
        self.dequant_distill_tokens = DeQuantStub()

    def to_vit(self):
        # Convert DistillableViT instance to a ViT instance
        v = ViT(*self.args, **self.kwargs)
        v.load_state_dict(self.state_dict())
        return v

    def _attend(self, x: torch.Tensor):
        x = self.dropout(x)
        with self.cm_transformer:
            x = self.transformer(x)
        return x

    def forward(
        self,
        img: torch.Tensor,
        distill_token: Optional[nn.Parameter] = None,
    ):
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

        if distill_token is not None:
            # Add distillation token at the end of the input sequence
            distill_tokens = distill_token.repeat(b, 1, 1)
            distill_tokens = self.quant_distill_token(distill_tokens)
            x = self.distill_token_cat.cat((x, distill_tokens), dim=1)

        x = self._attend(x)

        if distill_token is not None:
            x, distill_tokens = x[:, :-1], x[:, -1]

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)
        with self.cm_mlp_head:
            out = self.mlp_head(x)
        out = self.dequant_out(out)

        if distill_token is not None:
            distill_tokens = self.dequant_distill_tokens(distill_tokens)
            return out, distill_tokens
        else:
            return out


class DistillWrapper(nn.Module):
    # knowledge distillation wrapper
    def __init__(
        self,
        *,
        teacher: nn.Module,
        student: DistillableViT,
        temperature: float = 1.0,
        alpha: float = 0.5,
    ):
        super().__init__()
        assert isinstance(
            student, DistillableViT
        ), "student must be a vision transformer"

        self.teacher = teacher
        self.student = student

        dim = student.dim
        num_classes = student.num_classes
        self.temperature = temperature
        self.alpha = alpha

        self.distillation_token = nn.Parameter(torch.randn(1, 1, dim))

        self.distill_mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
        )

    def forward(
        self,
        img: torch.Tensor,
        labels: torch.Tensor,
        temperature: Optional[float] = None,
        alpha: Optional[float] = None,
        **kwargs,
    ):
        b, *_ = img.shape
        alpha = alpha if alpha is not None else self.alpha
        T = temperature if temperature is not None else self.temperature

        with torch.no_grad():
            teacher_logits = self.teacher(img)

        student_logits, distill_tokens = self.student(
            img,
            distill_token=self.distillation_token,
            **kwargs,
        )
        distill_logits = self.distill_mlp(distill_tokens)

        loss = F.cross_entropy(student_logits, labels)

        distill_loss = F.kl_div(
            F.log_softmax(distill_logits / T, dim=-1),
            F.softmax(teacher_logits / T, dim=-1).detach(),
            reduction="batchmean",
        )
        distill_loss *= T ** 2

        return loss * alpha + distill_loss * (1 - alpha)
