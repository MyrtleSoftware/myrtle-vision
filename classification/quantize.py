import enum

import torch
from qtorch import FixedPoint
from qtorch import FloatingPoint
from qtorch.quant import Quantizer as QTorchQuantizer
from torch.quantization import DeQuantStub
from torch.quantization import prepare_qat
from torch.quantization import QConfig
from torch.quantization import QuantStub


class QFormat(enum.IntEnum):
    """Quantization formats supported by ViT."""

    FP32 = 0
    PyTorchINT8 = 1
    FP16_16 = 2
    FP16_32 = 3
    TF32 = 4


class NumberFormat(enum.Enum):
    SymmetricInt8 = enum.auto()
    AsymmetricInt8 = enum.auto()
    HalfPrecisionFloat = enum.auto()
    SinglePrecisionFloat = enum.auto()
    TensorFloat32 = enum.auto()
    FixedPoint11Integral2 = enum.auto()
    FixedPoint11Integral3 = enum.auto()
    FixedPoint11Integral4 = enum.auto()

    @staticmethod
    def quantizer(number_format):
        """Returns a module that simulates `number_format`

        The module expects as input a single-precision floating point tensor
        and returns a single-precision floating point tensor that is
        constrained to `number_format`.

        Args:
            number_format: The number format to simulate.
            dim: Used for block number formats. It denotes the dimension to
                compute the blocks over.
        """
        if number_format == NumberFormat.HalfPrecisionFloat:
            return QTorchQuantizer(
                FloatingPoint(exp=5, man=10),
                forward_rounding="nearest",
            )
        elif number_format == NumberFormat.SinglePrecisionFloat:
            return torch.nn.Identity()
        elif number_format == NumberFormat.TensorFloat32:
            return QTorchQuantizer(
                FloatingPoint(exp=8, man=10),
                forward_rounding="nearest",
            )
        elif number_format == NumberFormat.FixedPoint11Integral2:
            return QTorchQuantizer(
                FixedPoint(wl=11, fl=9),
                forward_rounding="nearest",
            )
        elif number_format == NumberFormat.FixedPoint11Integral3:
            return QTorchQuantizer(
                FixedPoint(wl=11, fl=8),
                forward_rounding="nearest",
            )
        elif number_format == NumberFormat.FixedPoint11Integral4:
            return QTorchQuantizer(
                FixedPoint(wl=11, fl=7),
                forward_rounding="nearest",
            )

        raise NotImplementedError(number_format)


class QuantizerFunction(torch.autograd.Function):
    """A QConfig Observer-compatible Function for fake quantization."""

    @staticmethod
    def forward(ctx, X, quant):
        dtype = X.dtype
        assert X.is_floating_point()
        result = quant(X.data.float()).to(dtype)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class Quantizer(torch.nn.Module):
    """A QConfig Observer-compatible module for fake quantization.

    Args:
        number_format: See `NumberFormat.quantizer`.
        dim: See `NumberFormat.quantizer`.
    """

    def __init__(self, number_format):
        super().__init__()
        self._number_format = number_format
        self._quant = NumberFormat.quantizer(number_format)

    def get_qparams(self):
        raise NotImplementedError()

    def forward(self, X):
        return QuantizerFunction.apply(X, self._quant)

    def forward_pre_hook(self, module, input):
        assert (
            len(input) == 1
        ), f"{self.__class__.__name__} only supports single tensor input"
        return self(input[0])

    def __repr__(self):
        return self.__class__.__name__ + f"({self._number_format})"


class QLinear(torch.nn.Linear):
    """A Linear wrapper that applies `weight_fake_quant` in `from_float`."""

    def __init__(self, *args, activation_post_process=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.activation_post_process = activation_post_process

    def forward(self, input):
        output = super().forward(input)
        if self.activation_post_process is not None:
            output = self.activation_post_process(output)
        return output

    @classmethod
    def from_float(cls, mod, qconfig=None):
        q_mod = cls(
            in_features=mod.in_features,
            out_features=mod.out_features,
            bias=mod.bias is not None,
        )
        q_mod.weight.data = mod.weight_fake_quant(mod.weight).data
        q_mod.bias = mod.bias
        return q_mod


class QLayerNorm(torch.nn.LayerNorm):
    """A LayerNorm wrapper that applies `weight_fake_quant` in `from_float`."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = super().forward(input)
        return output

    @classmethod
    def from_float(cls, mod):
        q_mod = cls(
            normalized_shape=mod.normalized_shape,
            eps=mod.eps,
            elementwise_affine=mod.elementwise_affine,
        )
        weight_fake_quant = mod.qconfig.weight()
        q_mod.weight.data = weight_fake_quant(mod.weight).data
        q_mod.bias = mod.bias
        return q_mod


class QGELU(torch.nn.Module):
    """
    GELU wrapper that de-quantizes the input and re-quantizes the output of the
    GELU activation, since there isn't a quantized version supported in PyTorch
    """

    def __init__(self):
        super().__init__()
        self.gelu = torch.nn.GELU()
        self.dequant_input = DeQuantStub()
        self.quant_activation = QuantStub()

    def forward(self, input):
        input = self.dequant_input(input)
        activation = self.gelu(input)
        return self.quant_activation(activation)


class ModelQuantizer:
    def __init__(self, model):
        self.model = model

    def prepare_qat(self, q_format):
        """Make the model simulate `q_format`."""
        if hasattr(self, "q_format") and self.q_format != QFormat.FP32:
            raise ValueError("model already quantized")

        if isinstance(q_format, str):
            q_format = QFormat[q_format]

        if q_format == QFormat.FP32:
            # all floating point: already "quantized" to this
            pass
        elif q_format == QFormat.PyTorchINT8:
            self._prepare_qat_pytorch_int8()
        elif q_format == QFormat.FP16_16:
            self._prepare_qat_fp16_16()
        elif q_format == QFormat.FP16_32:
            self._prepare_qat_fp16_32()
        elif q_format == QFormat.TF32:
            self._prepare_qat_tf32()
        else:
            raise NotImplementedError(f"unknown q_format={q_format}")

        self.q_format: QFormat = q_format

    def _activation_pre_process(self, module, quantizer):
        # ensure activations are pre-processed by inserting a QuantStub before
        # the module
        stub = QuantStub()
        stub.qconfig = QConfig(activation=lambda: quantizer, weight=None)
        return torch.nn.Sequential(stub, module)

    def _reassign_attrs(self, reassign):
        for name, mod in reassign.items():
            inner = self.model
            names = name.split(".")
            for sub_name in names[:-1]:
                inner = getattr(inner, sub_name)
            setattr(inner, names[-1], mod)

    def _prepare_qat_pytorch_int8(self):
        # PyTorch quantization format, use built-in tooling

        # PyTorch doesn't support quantized GELU, so substitute it with QGELU
        # wrapper
        reassign = {}
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.GELU):
                reassign[name] = QGELU()
        self._reassign_attrs(reassign)

        # add qconfig
        self.model.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.MinMaxObserver.with_args(
                reduce_range=False
            ),
            weight=torch.quantization.MinMaxObserver.with_args(
                qscheme=torch.per_tensor_symmetric, dtype=torch.qint8
            ),
        )

        prepare_qat(self.model, inplace=True)

    def _prepare_qat_fp16_16(self):
        reassign = {}

        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.quantized.FloatFunctional):
                module.qconfig = QConfig(
                    activation=lambda: Quantizer(
                        NumberFormat.HalfPrecisionFloat
                    ),
                    weight=None,
                )
            elif isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm)):
                module.qconfig = QConfig(
                    activation=lambda: Quantizer(
                        NumberFormat.HalfPrecisionFloat
                    ),
                    weight=lambda **kwargs: Quantizer(NumberFormat.HalfPrecisionFloat),
                )
                reassign[name] = self._activation_pre_process(
                    module, Quantizer(NumberFormat.HalfPrecisionFloat)
                )
            elif isinstance(module, torch.nn.GELU):
                module.qconfig = QConfig(
                    activation=lambda: Quantizer(
                        NumberFormat.HalfPrecisionFloat
                    ),
                    weight=None,
                )
                reassign[name] = self._activation_pre_process(
                    module, Quantizer(NumberFormat.HalfPrecisionFloat)
                )

        self._reassign_attrs(reassign)

        prepare_qat(self.model, inplace=True)

    def _prepare_qat_fp16_32(self):
        # fp16.32 uses FP32 for accumulation and FP16 for weights
        reassign = {}

        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm)):
                module.qconfig = QConfig(
                    activation=lambda: Quantizer(
                        NumberFormat.SinglePrecisionFloat
                    ),
                    weight=lambda **kwargs: Quantizer(NumberFormat.HalfPrecisionFloat),
                )
                reassign[name] = self._activation_pre_process(
                    module, Quantizer(NumberFormat.HalfPrecisionFloat)
                )

        self._reassign_attrs(reassign)

        prepare_qat(self.model, inplace=True)

    def _prepare_qat_tf32(self):
        # TF32 uses FP32 for accumulation
        reassign = {}

        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm)):
                module.qconfig = QConfig(
                    activation=lambda: Quantizer(
                        NumberFormat.SinglePrecisionFloat
                    ),
                    weight=lambda **kwargs: Quantizer(NumberFormat.TensorFloat32),
                )
                reassign[name] = self._activation_pre_process(
                    module, Quantizer(NumberFormat.TensorFloat32)
                )

        self._reassign_attrs(reassign)

        prepare_qat(self.model, inplace=True)

    def convert(self):
        if self.q_format == QFormat.FP32:
            # all floating point: already "quantized" to this
            pass
        elif self.q_format == QFormat.PyTorchINT8:
            torch.quantization.convert(self.model, inplace=True)
        elif self.q_format in [
            QFormat.FP16_16,
            QFormat.FP16_32,
            QFormat.TF32,
        ]:
            mapping = {
                torch.nn.qat.modules.linear.Linear: QLinear,
                torch.nn.modules.normalization.LayerNorm: QLayerNorm,
            }
            torch.quantization.convert(
                self.model, mapping=mapping, inplace=True
            )
        else:
            raise NotImplementedError(f"unknown q_format={self.q_format}")
