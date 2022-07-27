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
            filepath=train_config["checkpoint_path"],
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

def layer_substitution_rules():
    """
    Returns a list of tuples that are used to replace the names of the timm
    vision transformer layers with the names that are used in this repository.
    """
    rules = [
    (r'pos_embed', r'pos_embedding'),
    (r'patch_embed.proj.weight', r'patch_to_embedding.weight'),
    (r'patch_embed.proj.bias', r'patch_to_embedding.bias'),
    # mlp head 0
    (r'norm.weight', r'mlp_head.0.weight'),
    (r'norm.bias', r'mlp_head.0.bias'),
    ## mlp head 1
    (r'head.weight', r'mlp_head.1.weight'),
    (r'head.bias', r'mlp_head.1.bias'),
    ]

    for depth in range(12):
        rules.extend([
            # norm
            (r'blocks.'+str(depth)+r'.norm1.weight', r'transformer.layers.'+str(depth)+r'.0.fn.norm.weight'),
            (r'blocks.'+str(depth)+r'.norm1.bias', r'transformer.layers.'+str(depth)+r'.0.fn.norm.bias'),
            # qkv
            (r'blocks.'+str(depth)+r'.attn.qkv.weight', r'transformer.layers.'+str(depth)+r'.0.fn.fn.to_qkv.weight'),
            (r'blocks.'+str(depth)+r'.attn.qkv.bias', r'transformer.layers.'+str(depth)+r'.0.fn.fn.to_qkv.bias'),
            # proj
            (r'blocks.'+str(depth)+r'.attn.proj.weight', r'transformer.layers.'+str(depth)+r'.0.fn.fn.to_out.0.weight'),
            (r'blocks.'+str(depth)+r'.attn.proj.bias', r'transformer.layers.'+str(depth)+r'.0.fn.fn.to_out.0.bias'),
            # norm
            (r'blocks.'+str(depth)+r'.norm2\.weight', r'transformer.layers.'+str(depth)+r'.1.fn.norm.weight'),
            (r'blocks.'+str(depth)+r'.norm2\.bias', r'transformer.layers.'+str(depth)+r'.1.fn.norm.bias'),
            # fcn
            (r'blocks.'+str(depth)+r'.mlp.fc1.weight', r'transformer.layers.'+str(depth)+r'.1.fn.fn.net.0.weight'),
            (r'blocks.'+str(depth)+r'.mlp.fc1.bias', r'transformer.layers.'+str(depth)+r'.1.fn.fn.net.0.bias'),
            # fcn
            (r'blocks.'+str(depth)+r'.mlp.fc2.weight', r'transformer.layers.'+str(depth)+r'.1.fn.fn.net.3.weight'),
            (r'blocks.'+str(depth)+r'.mlp.fc2.bias', r'transformer.layers.'+str(depth)+r'.1.fn.fn.net.3.bias'),
        ])
    return rules


def apply_rules(name, rules):
    for pattern, replacement in rules:
        name = re.sub(pattern, replacement, name)
    return name


def pretrained_backbone_name(vit_config):
    model_name = vit_config["vit_name"]+"_patch"+str(vit_config["patch_size"])+"_"+str(vit_config["image_size"])
    return model_name


def pretrained_backbone_exists(backbone_name):
    all_backbone_names = timm.list_models(pretrained=True)
    return backbone_name in all_backbone_names


def get_pretrained_backbone_weights(model_name, vit_config):

    # Get the layer substitution rules
    rules = layer_substitution_rules()

    num_classes = vit_config["num_classes"]
    timm_vit = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    timm_vit.eval()

    pretrained_state_dict = {}

    for key in list(timm_vit.state_dict()):

        new_key = apply_rules(key, rules)

        if new_key == "patch_to_embedding.weight":
            embed_dim = vit_config["embed_dim"]
            mlp_dim = vit_config["mlp_dim"]
            patch_dim = vit_config["patch_size"]**2 * 3
            pretrained_state_dict[new_key] = timm_vit.state_dict()[key].permute(0, 2, 3, 1).reshape(embed_dim, patch_dim)
        else:
            pretrained_state_dict[new_key] = timm_vit.state_dict()[key]
    return pretrained_state_dict
