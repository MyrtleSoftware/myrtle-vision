import argparse
import json

import numpy as np
import onnx
import onnxruntime as ort
import torch
from utils.data_loader import Resisc45Loader
from utils.models import get_models
from utils.models import prepare_model_and_load_ckpt
from utils.utils import parse_config


@torch.no_grad()
def export_model(model, sample, output_filename):
    # Add batch dimension
    sample = sample.unsqueeze(0)
    model.eval()

    example_ins = (sample,)
    example_outs = model(*example_ins)
    input_names = ["img"]
    output_names = ["label_probs"]
    dynamic_axes = {
        "img": {0: "batch_size"},
        "label_probs": {0: "batch_size"},
    }
    torch.onnx.export(
        model,
        example_ins,
        output_filename,
        export_params=True,
        verbose=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=11,
    )


@torch.no_grad()
def check_equivalence(torch_model, sample, onnx_filename):
    # Add batch dimension
    sample = sample.unsqueeze(0)
    torch_model.eval()
    torch_output = torch_model(sample).numpy()

    onnx_session = ort.InferenceSession(onnx_filename)
    [onnx_output] = onnx_session.run(None, {"img": np.array(sample)})

    if np.allclose(torch_output, onnx_output, atol=1e-06):
        print(
            "\nThe outputs from PyTorch and ONNX models are element-wise"
            " equal within a tolerance"
        )
    else:
        print(
            "\nThe outputs from PyTorch and ONNX models are NOT element-wise"
            " equal within a tolerance"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, help="JSON file for configuration"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="vit.onnx",
        help="Path to save the ONNX model",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)

    # parse data config
    train_config = config["train_config"]
    data_config = parse_config(config["data_config_path"])
    dataset_path = data_config["dataset_path"]
    label_map_path = data_config["label_map"]
    device = "cpu"

    # load validation set
    valset = Resisc45Loader(
        mode="eval",
        dataset_path=dataset_path,
        imagepaths=data_config["valid_files"],
        label_map_path=label_map_path,
        transform_config=data_config["transform_ops_val"],
    )

    # Remove dropout
    config["vit_config"]["dropout"] = 0.0
    config["vit_config"]["emb_dropout"] = 0.0
    # Instantiate models
    vit, _ = get_models(config)
    vit = vit.to(device)

    # Load pre-trained weights
    assert (
        train_config["checkpoint_path"] != ""
    ), "Must provide a checkpoint path in the config file"
    prepare_model_and_load_ckpt(train_config=train_config, model=vit)

    export_model(vit, valset[0][0], args.output)
    print(f"\nModel successfully saved to {args.output}")

    onnx_model = onnx.load(args.output)
    onnx.checker.check_model(onnx_model)
    print("\nONNX checker passes")

    # Check output equivalence between PyTorch and ONNX models
    check_equivalence(vit, valset[1][0], args.output)
