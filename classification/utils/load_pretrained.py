import re

import timm


def layer_substitution_rules():
    """
    This function returns a list with tuples that are used to
    replace the names of the timm vision transformer layers into
    the names that are used in this repository.
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
        ]
        )
    return rules


def apply_rules(name, rules):
    for pattern, replacement in rules:
        name = re.sub(pattern, replacement, name)
    return name


def pretrained_backbone_name(vit_config):
    """
    """
    model_name = vit_config["vit_name"]+"_patch"+str(vit_config["patch_size"])+"_"+str(vit_config["image_size"])
    return model_name


def pretrained_backbone_exists(backbone_name):
    """
    """
    all_backbone_names = timm.list_models(pretrained=True)
    return backbone_name in all_backbone_names
    

def get_pretrained_backbone_weights(model_name, vit_config):
    """
    """

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
            pretrained_state_dict[new_key] = timm_vit.state_dict()[key].reshape(embed_dim, patch_dim)
        else:
            pretrained_state_dict[new_key] = timm_vit.state_dict()[key]
    return pretrained_state_dict


