import torch
import torch.nn as nn
from zeta.nn.modules.simple_rmsnorm import SimpleRMSNorm
import torch.nn.functional as F


def activation_quant(x: torch.Tensor):
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y


def weight_quant(w: torch.Tensor):
    scale = w.abs().mean()
    e = w.mean()
    u = (w - e).sign() * scale
    return u


class LBitLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        device=None,
        dtype=None,
    ):
        super(LBitLinear, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.register_buffer("w_quant", self.weight.detach().clone())

    def __repr__(self):
        return f"LBitLinear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"

    def precompute_quantized_weights(self):
        with torch.no_grad():
            self.w_quant.copy_(weight_quant(self.weight))

    def forward(self, x):
        x_norm = SimpleRMSNorm(self.in_features)(x)
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        y = F.linear(x_quant, self.w_quant, self.bias)
        return y


def count_layers(module, layer_type=nn.Module):
    count = 0
    for child in module.children():
        if isinstance(child, layer_type):
            count += 1
        count += count_layers(child, layer_type)
    return count


def replace_modules(model, old_class=nn.Linear, new_class=LBitLinear, max_layers=225):
    replaced_layers = 0
    replaced_params = 0
    total_params = sum(p.numel() for p in model.parameters())
    total_layers = count_layers(model, old_class)

    def replace_layer(module):
        nonlocal replaced_layers, replaced_params
        for name, child in module.named_children():
            if isinstance(child, old_class):
                if replaced_layers < max_layers:
                    kwargs = {
                        "in_features": child.in_features,
                        "out_features": child.out_features,
                        "bias": child.bias is not None,
                    }
                    new_module = new_class(**kwargs)
                    new_module.weight = child.weight
                    if child.bias is not None:
                        new_module.bias = child.bias
                    setattr(module, name, new_module)
                    replaced_layers += 1
                    replaced_params += sum(p.numel()
                                           for p in child.parameters())
            else:
                replace_layer(child)

    replace_layer(model)
    print(
        f"Replaced {replaced_layers} layers out of {total_layers} total {old_class.__name__} layers")
    print(
        f"Replaced {replaced_params} parameters out of {total_params} total parameters")
