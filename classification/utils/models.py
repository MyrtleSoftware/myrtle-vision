import argparse
import re

import timm
import torch
from quantize import QFormat
from torchvision.models import resnet50
from utils.utils import parse_config
from vit_pytorch.distill import DistillableViT
from vit_pytorch.distill import DistillWrapper
from vit_pytorch.vit_pytorch import ViT


def get_teacher(num_classes, weights_path):
    # Load pre-trained weights
    teacher = resnet50(num_classes=num_classes)
    teacher.load_state_dict(torch.load(weights_path, map_location="cpu"))
    # freeze teacher parameters
    for param in teacher.parameters():
        param.requires_grad = False
    teacher.eval()
    return teacher


def get_models(config, profile=False):
    vit_config = config["vit_config"]
    data_config = parse_config(config["data_config_path"])

    vit_kwargs = {
        "image_size": vit_config["image_size"],
        "patch_size": vit_config["patch_size"],
        "num_classes": data_config["number_of_classes"],
        "dim": vit_config["embed_dim"],
        "depth": vit_config["depth"],
        "heads": vit_config["heads"],
        "mlp_dim": vit_config["mlp_dim"],
        "dropout": vit_config["dropout"],
        "emb_dropout": vit_config["emb_dropout"],
        "profile": profile,
        "apply_prune_mask": vit_config["apply_prune_mask"],
        "q_format": QFormat[vit_config["q_format"]],
    }
    if "distiller_config" in config:
        # prepare student and teacher
        vit = DistillableViT(**vit_kwargs)
        distiller_config = config["distiller_config"]
        teacher = get_teacher(
            num_classes=data_config["number_of_classes"],
            weights_path=distiller_config["teacher_weights_path"],
        )
        # prepare distiller wrapper
        distiller = DistillWrapper(
            student=vit,
            teacher=teacher,
            temperature=distiller_config["temperature"],
            alpha=distiller_config["alpha"],
        )
        return vit, distiller
    else:
        return ViT(**vit_kwargs), None


def prepare_model_and_load_ckpt(
    train_config,
    model,
    optimizer=None,
    lr_scheduler=None,
):
    if train_config["checkpoint_path"] != "":
        # resume training from checkpoint file
        iteration = load_checkpoint(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            filepath="checkpoints/" + train_config["checkpoint_path"],
        )
    else:
        # train from scratch
        iteration = 0

    return iteration


def get_optimizer_args(train_config):
    optimizer_args = argparse.Namespace

    # Optimizer parameters
    optimizer_args.opt = train_config["optimizer"]
    optimizer_args.opt_eps = train_config["opt_eps"]
    optimizer_args.opt_betas = train_config["opt_betas"]
    optimizer_args.clip_grad = train_config["clip_grad"]
    optimizer_args.momentum = train_config["momentum"]
    optimizer_args.weight_decay = train_config["weight_decay"]
    # Learning rate schedule parameters
    optimizer_args.sched = train_config["scheduler"]
    lr = train_config["lr"] * train_config["global_batch_size"] / 512.0
    optimizer_args.lr = lr
    optimizer_args.lr_noise = train_config["lr_noise"]
    optimizer_args.lr_noise_pct = train_config["lr_noise_pct"]
    optimizer_args.lr_noise_std = train_config["lr_noise_std"]
    optimizer_args.warmup_lr = train_config["warmup_lr"]
    optimizer_args.min_lr = train_config["min_lr"]
    optimizer_args.epochs = train_config["epochs"]
    optimizer_args.decay_epochs = train_config["decay_epochs"]
    optimizer_args.warmup_epochs = train_config["warmup_epochs"]
    optimizer_args.cooldown_epochs = train_config["cooldown_epochs"]
    optimizer_args.patience_epochs = train_config["patience_epochs"]
    optimizer_args.decay_rate = train_config["decay_rate"]

    return optimizer_args


def save_checkpoint(
    model,
    optimizer,
    lr_scheduler,
    iteration,
    filepath,
):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "iteration": iteration,
    }
    torch.save(ckpt, filepath)


def load_checkpoint(
    model, optimizer, lr_scheduler, filepath
):
    checkpoint = torch.load(filepath, map_location="cpu")

    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    iteration = checkpoint["iteration"]

    return iteration


def apply_rules(name, rules):
    """Apply the first matching rule (regex substitution) to name.
    """
    for pattern, replacement in rules:
        m = re.match(pattern, name)
        if m is not None:
            return re.sub(pattern, replacement, name)
    return name


def rename_timm_state_dict(timm_model_name, vit_config, num_classes):
    """Returns a state dict with weights from a pretrained timm model.
    """
    rules = [
        # Input
        ## Positional embedding
        (r"pos_embed", r"pos_embedding"),
        ## Patch embedding
        (r"patch_embed\.proj\.weight", r"patch_to_embedding.weight"),
        (r"patch_embed\.proj\.bias", r"patch_to_embedding.bias"),

        # Transformer layers
        ## Self-attention
        ### norm
        (r"blocks\.([0-9]+)\.norm1\.weight", r"transformer.layers.\1.0.fn.norm.weight"),
        (r"blocks\.([0-9]+)\.norm1\.bias", r"transformer.layers.\1.0.fn.norm.bias"),
        ### to_qkv
        (r"blocks\.([0-9]+)\.attn\.qkv\.weight", r"transformer.layers.\1.0.fn.fn.to_qkv.weight"),
        (r"blocks\.([0-9]+)\.attn\.qkv\.bias", r"transformer.layers.\1.0.fn.fn.to_qkv.bias"),
        ### proj
        (r"blocks\.([0-9]+)\.attn\.proj\.weight", r"transformer.layers.\1.0.fn.fn.to_out.0.weight"),
        (r"blocks\.([0-9]+)\.attn\.proj\.bias", r"transformer.layers.\1.0.fn.fn.to_out.0.bias"),

        # Feedforward
        ## norm
        (r"blocks\.([0-9]+)\.norm2\.weight", r"transformer.layers.\1.1.fn.norm.weight"),
        (r"blocks\.([0-9]+)\.norm2\.bias", r"transformer.layers.\1.1.fn.norm.bias"),
        ## fc1
        (r"blocks\.([0-9]+)\.mlp\.fc1\.weight", r"transformer.layers.\1.1.fn.fn.net.0.weight"),
        (r"blocks\.([0-9]+)\.mlp\.fc1\.bias", r"transformer.layers.\1.1.fn.fn.net.0.bias"),
        ## fc2
        (r"blocks\.([0-9]+)\.mlp\.fc2\.weight", r"transformer.layers.\1.1.fn.fn.net.3.weight"),
        (r"blocks\.([0-9]+)\.mlp\.fc2\.bias", r"transformer.layers.\1.1.fn.fn.net.3.bias"),

        # Classifier head
        ## norm
        (r"norm\.weight", r"mlp_head.0.weight"),
        (r"norm\.bias", r"mlp_head.0.bias"),
        ## fc
        (r"head\.weight", r"mlp_head.1.weight"),
        (r"head\.bias", r"mlp_head.1.bias"),
    ]

    timm_vit = timm.create_model(timm_model_name, pretrained=True, num_classes=num_classes)

    # timm state_dict with renamed keys
    state_dict = {}
    for key in timm_vit.state_dict():
        new_key = apply_rules(key, rules)
        # Special case for patch embedding which we implement as a Linear layer
        # rather than a Conv2D.
        if new_key == "patch_to_embedding.weight":
            embed_dim = vit_config["embed_dim"]
            mlp_dim = vit_config["mlp_dim"]
            patch_dim = vit_config["patch_size"]**2 * 3
            # (O,I,H,W) -> (O,(H,W,I))
            state_dict[new_key] = timm_vit.state_dict()[key].permute(0, 2, 3, 1).reshape(embed_dim, patch_dim)
        else:
            state_dict[new_key] = timm_vit.state_dict()[key]
    return state_dict
