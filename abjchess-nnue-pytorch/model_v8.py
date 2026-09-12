"""Torch model used by the AB-JChess V8 trainer.

The module is intentionally named and versioned separately from generic model
implementations.  The tensor shapes mirror
the fixed native evaluator: a 2,048-wide feature transformer, sixteen
layer-stack heads, and the squared/clipped 16 -> 32 -> 1 head used by the
runtime.
"""

from __future__ import annotations

import copy
import math
import numbers
from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch import nn
import torch.nn.functional as F

try:
    import pytorch_lightning as pl
    _LightningBase = pl.LightningModule
except Exception:  # pragma: no cover - doctor/static tools can omit Lightning
    pl = None
    _LightningBase = nn.Module

try:
    import ranger21  # type: ignore
except Exception:  # pragma: no cover - AdamW fallback is sufficient for smoke tests
    ranger21 = None

from feature_transformer import DoubleFeatureTransformerSlice
import features_v8
from head_balance_v8 import compute_head_weights


DEFAULT_L1 = features_v8.L1_DIM
DEFAULT_L2 = features_v8.L2_DIM
DEFAULT_L3 = features_v8.L3_DIM
L1, L2, L3 = DEFAULT_L1, DEFAULT_L2, DEFAULT_L3

_UINT64_MASK = (1 << 64) - 1
_HEAD_SEED_STEP = 0x9E3779B1
_OUTPUT_WEIGHT_RANGE = 127 * 127 / 9600


def _normalized_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise TypeError("seed must be an integer, not a boolean")
    return int(seed) & _UINT64_MASK


def _cpu_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def dims_for_feature_set(feature_set: Any) -> tuple[int, int, int]:
    return (
        int(getattr(feature_set, "l1", None) or DEFAULT_L1),
        int(getattr(feature_set, "l2", None) or DEFAULT_L2),
        int(getattr(feature_set, "l3", None) or DEFAULT_L3),
    )


def coalesce_ft_weights(model: Any, layer: nn.Module) -> torch.Tensor:
    """Fold factor rows into each of the 31,776 real feature rows."""

    weight = layer.weight.data
    gather = model.feature_set.get_virtual_to_real_features_gather_indices()
    result = weight.new_zeros((model.feature_set.num_real_features, weight.shape[1]))
    for real_index, factors in enumerate(gather):
        result[real_index, :] = sum((weight[index, :] for index in factors), weight.new_zeros(weight.shape[1]))
    return result


def get_parameters(layers: Iterable[nn.Module]) -> list[nn.Parameter]:
    return [parameter for layer in layers for parameter in layer.parameters()]


@dataclass(frozen=True)
class ForwardComponents:
    accumulator: torch.Tensor
    psqt: torch.Tensor
    head: torch.Tensor
    prediction: torch.Tensor

    def __iter__(self):
        return iter((self.accumulator, self.psqt, self.head, self.prediction))


class LayerStacks(nn.Module):
    def __init__(self, count: int, l1: int, l2: int, l3: int, use_jieqi_squared: bool = True) -> None:
        super().__init__()
        self.count = int(count)
        self.l1_dim, self.l2_dim, self.l3_dim = int(l1), int(l2), int(l3)
        self.use_jieqi_squared = bool(use_jieqi_squared)
        self.l1_outputs = self.l2_dim + 1 if self.use_jieqi_squared else self.l2_dim
        self.l2_inputs = self.l2_dim * 2 if self.use_jieqi_squared else self.l2_dim
        self.l1 = nn.Linear(2 * self.l1_dim, self.l1_outputs * self.count)
        self.l1_fact = nn.Linear(2 * self.l1_dim, self.l1_outputs, bias=False)
        self.l2 = nn.Linear(self.l2_inputs, self.l3_dim * self.count)
        self.output = nn.Linear(self.l3_dim, self.count)
        self.idx_offset: torch.Tensor | None = None
        self._init_layers()

    def _init_layers(self) -> None:
        with torch.no_grad():
            self.l1_fact.weight.zero_()
            self.output.bias.zero_()
            if self.use_jieqi_squared:
                for index in range(self.count):
                    direct = index * self.l1_outputs + self.l2_dim
                    self.l1.weight[direct].zero_()
                    self.l1.bias[direct].zero_()
                self.l1_fact.weight[self.l2_dim].zero_()

    def reset_independent(self, seed: int) -> None:
        """Reset every dense head from a deterministic, head-specific seed."""

        base_seed = _normalized_seed(seed)
        with torch.no_grad():
            for index in range(self.count):
                generator = _cpu_generator((base_seed + index * _HEAD_SEED_STEP) & _UINT64_MASK)
                l1_weight = torch.empty((self.l1_outputs, self.l1.in_features), dtype=torch.float32)
                l2_weight = torch.empty((self.l3_dim, self.l2.in_features), dtype=torch.float32)
                output_weight = torch.empty((1, self.output.in_features), dtype=torch.float32)
                nn.init.kaiming_uniform_(l1_weight, a=math.sqrt(5), generator=generator)
                nn.init.kaiming_uniform_(l2_weight, a=math.sqrt(5), generator=generator)
                nn.init.orthogonal_(output_weight, generator=generator)
                output_weight.mul_(_OUTPUT_WEIGHT_RANGE)
                l1_start = index * self.l1_outputs
                l2_start = index * self.l3_dim
                self.l1.weight[l1_start:l1_start + self.l1_outputs].copy_(
                    l1_weight.to(device=self.l1.weight.device, dtype=self.l1.weight.dtype))
                self.l2.weight[l2_start:l2_start + self.l3_dim].copy_(
                    l2_weight.to(device=self.l2.weight.device, dtype=self.l2.weight.dtype))
                self.output.weight[index:index + 1].copy_(
                    output_weight.to(device=self.output.weight.device, dtype=self.output.weight.dtype))
                self.l1.bias[l1_start:l1_start + self.l1_outputs].zero_()
                self.l2.bias[l2_start:l2_start + self.l3_dim].zero_()
                self.output.bias[index:index + 1].zero_()
                if self.use_jieqi_squared:
                    direct = l1_start + self.l2_dim
                    self.l1.weight[direct].zero_()
                    self.l1.bias[direct].zero_()
            if self.use_jieqi_squared:
                # The direct channel has a shared factor row.  Keep it at the
                # native zero baseline on every reset, including resets after
                # a caller has mutated the layer in-place.
                self.l1_fact.weight[self.l2_dim].zero_()

    # The public spelling makes it explicit that head zero is not copied.
    def reset_independent_heads(self, seed: int) -> None:
        self.reset_independent(seed)

    def _normalize_selection(self, ls_indices: torch.Tensor, blend: torch.Tensor | None,
                             batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        indices = torch.as_tensor(ls_indices, device=device)
        if indices.ndim == 2 and tuple(indices.shape) == (batch_size, 1):
            indices = indices[:, 0]
        elif indices.ndim != 1 or indices.shape[0] != batch_size:
            raise ValueError("layer-stack bucket tensor must have shape [batch, 1] or [batch]")
        if indices.dtype not in {
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
        }:
            raise ValueError("layer-stack bucket tensor must be integral")
        indices = indices.long()
        if bool(((indices < 0) | (indices >= self.count)).any().detach().item()):
            raise ValueError(f"layer-stack bucket index is outside [0, {self.count})")
        if blend is None:
            alpha = torch.zeros((batch_size,), dtype=torch.float32, device=device)
        else:
            alpha = torch.as_tensor(blend, device=device)
            if alpha.ndim == 2 and tuple(alpha.shape) == (batch_size, 1):
                alpha = alpha[:, 0]
            elif alpha.ndim != 1 or alpha.shape[0] != batch_size:
                raise ValueError("layer-stack blend tensor must have shape [batch, 1] or [batch]")
            alpha = alpha.to(dtype=torch.float32)
            if not bool(torch.isfinite(alpha).all().detach().item()):
                raise ValueError("layer-stack blend must be finite")
            if bool(((alpha < 0.0) | (alpha > 1.0)).any().detach().item()):
                raise ValueError("layer-stack blend must be in [0, 1]")
            # Head 15 is an endpoint even if a malformed producer supplied a
            # non-zero alpha; normalising here keeps the runtime contract
            # deterministic and avoids indexing a seventeenth head.
            alpha = torch.where(indices == self.count - 1, torch.zeros_like(alpha), alpha)
        return indices, alpha

    def forward_interpolated(self, x: torch.Tensor, ls_indices: torch.Tensor,
                             blend: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError("layer-stack input must have shape [batch, width]")
        batch_size = int(x.shape[0])
        indices0, alpha = self._normalize_selection(ls_indices, blend, batch_size, x.device)
        if self.idx_offset is None or self.idx_offset.shape[0] != batch_size or self.idx_offset.device != x.device:
            self.idx_offset = torch.arange(0, batch_size * self.count, self.count, device=x.device)
        offsets = self.idx_offset
        flat0 = indices0 + offsets
        indices1 = torch.minimum(indices0 + 1, indices0.new_tensor(self.count - 1))
        flat1 = indices1 + offsets

        l1_all = self.l1(x).reshape((-1, self.count, self.l1_outputs))
        l1_factor = self.l1_fact(x).unsqueeze(1)
        l1x_all = l1_all + l1_factor
        direct_all = None
        if self.use_jieqi_squared:
            main = l1x_all[..., :self.l2_dim]
            squared = torch.clamp(main.square() * (127.0 / 128.0), 0.0, 1.0)
            clipped = torch.clamp(main, 0.0, 1.0)
            direct_all = l1x_all[..., self.l2_dim:self.l2_dim + 1]
            l2_input = torch.cat([squared, clipped], dim=2)
        else:
            l2_input = torch.clamp(l1x_all, 0.0, 1.0)
        l2_all = self.l2(l2_input.reshape((-1, self.l2_inputs))).reshape((-1, self.count, self.l3_dim))
        l2x_all = torch.clamp(l2_all, 0.0, 1.0)
        output_all = self.output(l2x_all.reshape((-1, self.l3_dim))).reshape((-1, self.count, 1))
        result0 = output_all.reshape((-1, 1))[flat0]
        result1 = output_all.reshape((-1, 1))[flat1]
        result = result0 + (result1 - result0) * alpha.unsqueeze(1)
        if direct_all is not None:
            direct0 = direct_all.reshape((-1, 1))[flat0]
            direct1 = direct_all.reshape((-1, 1))[flat1]
            result = result + direct0 + (direct1 - direct0) * alpha.unsqueeze(1)
        return result

    def forward(self, x: torch.Tensor, ls_indices: torch.Tensor,
                blend: torch.Tensor | None = None) -> torch.Tensor:
        return self.forward_interpolated(x, ls_indices, blend)

    def get_coalesced_layer_stacks(self):
        for index in range(self.count):
            with torch.no_grad():
                l1 = nn.Linear(2 * self.l1_dim, self.l1_outputs)
                l2 = nn.Linear(self.l2_inputs, self.l3_dim)
                output = nn.Linear(self.l3_dim, 1)
                l1.weight.copy_(self.l1.weight[index * self.l1_outputs:(index + 1) * self.l1_outputs] + self.l1_fact.weight)
                l1.bias.copy_(self.l1.bias[index * self.l1_outputs:(index + 1) * self.l1_outputs])
                l2.weight.copy_(self.l2.weight[index * self.l3_dim:(index + 1) * self.l3_dim])
                l2.bias.copy_(self.l2.bias[index * self.l3_dim:(index + 1) * self.l3_dim])
                output.weight.copy_(self.output.weight[index:index + 1])
                output.bias.copy_(self.output.bias[index:index + 1])
            yield l1, l2, output


class NNUE(_LightningBase):
    """V8 training module with optional Lightning integration."""

    def __init__(self, feature_set: features_v8.FeatureSetV8, lambda_: float = 1.0,
                 lr: float = 1.5e-3, head_init_seed: int = 0) -> None:
        super().__init__()
        if feature_set.name not in features_v8.FEATURE_NAMES:
            raise ValueError("NNUE V8 accepts only V8 feature sets")
        self.num_psqt_buckets = int(feature_set.num_psqt_buckets)
        self.num_ls_buckets = int(feature_set.num_ls_buckets)
        self.l1_dim, self.l2_dim, self.l3_dim = dims_for_feature_set(feature_set)
        self.ft_dim = int(getattr(feature_set, "ft_dim", DEFAULT_L1 * 2) or DEFAULT_L1 * 2)
        if (self.ft_dim, self.l1_dim, self.l2_dim, self.l3_dim) != (2048, 1024, 15, 32):
            raise ValueError("V8 architecture is fixed at FT=2048 L1=1024 L2=15 L3=32")
        self.pairwise_ft = self.ft_dim == self.l1_dim * 2
        self.use_jieqi_squared = True
        self.feature_set = feature_set
        self.input = DoubleFeatureTransformerSlice(feature_set.num_features, self.ft_dim + self.num_psqt_buckets)
        self.layer_stacks = LayerStacks(self.num_ls_buckets, self.l1_dim, self.l2_dim, self.l3_dim, True)
        self.lambda_ = float(lambda_)
        self.lr = float(lr)
        self.ranger21_num_batches_per_epoch: int | None = None
        self.ranger21_num_epochs: int | None = None
        self.weight_clipping = [
            {"params": [self.layer_stacks.l1.weight], "min_weight": -127 / 64, "max_weight": 127 / 64, "virtual_params": self.layer_stacks.l1_fact.weight},
            {"params": [self.layer_stacks.l2.weight], "min_weight": -127 / 64, "max_weight": 127 / 64},
            {"params": [self.layer_stacks.output.weight], "min_weight": -127 * 127 / 9600, "max_weight": 127 * 127 / 9600},
        ]
        self._init_layers()
        object.__setattr__(self, "_v81_head_balance", (1.0,) * self.num_ls_buckets)
        object.__setattr__(self, "_v81_head_balance_configured", False)
        object.__setattr__(self, "_v81_head_init_seed", int(head_init_seed))
        self.reset_independent_heads(head_init_seed)

    def _zero_virtual_feature_weights(self) -> None:
        if not self.feature_set.num_virtual_features:
            return
        with torch.no_grad():
            for begin, end in self.feature_set.get_virtual_feature_ranges():
                self.input.weight[begin:end].zero_()

    def _init_layers(self) -> None:
        with torch.no_grad():
            self.input.bias[self.ft_dim:].zero_()
            self._zero_virtual_feature_weights()
            psqt = torch.as_tensor(self.feature_set.get_initial_psqt_features(), dtype=self.input.weight.dtype, device=self.input.weight.device)
            self.input.weight[:, self.ft_dim:] = psqt[:, None] / 600.0

    def reset_independent_heads(self, seed: int) -> None:
        """Deterministically initialise all phase heads independently."""

        self.layer_stacks.reset_independent_heads(seed)
        object.__setattr__(self, "_v81_head_init_seed", int(_normalized_seed(seed)))

    def set_head_balance(self, weights: Iterable[float]) -> None:
        """Install sixteen finite positive head weights, normalised to sum 16."""

        try:
            values = tuple(float(value) for value in weights)
        except (TypeError, ValueError) as exc:
            raise ValueError("head balance must contain 16 finite positive weights") from exc
        if len(values) != self.num_ls_buckets:
            raise ValueError(f"head balance must contain exactly {self.num_ls_buckets} weights")
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("head balance weights must be finite and positive")
        total = math.fsum(values)
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError("head balance weights must have positive sum")
        normalized = tuple(value * self.num_ls_buckets / total for value in values)
        object.__setattr__(self, "_v81_head_balance", normalized)
        object.__setattr__(self, "_v81_head_balance_configured", True)

    def head_balance(self) -> tuple[float, ...]:
        return tuple(getattr(self, "_v81_head_balance", (1.0,) * self.num_ls_buckets))

    def _effective_head_weights(self, layer_stack_indices: torch.Tensor,
                                layer_stack_blend: torch.Tensor | None,
                                *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        indices = torch.as_tensor(layer_stack_indices, device=device)
        if indices.ndim == 2 and indices.shape[1] == 1:
            indices = indices[:, 0]
        if indices.ndim != 1:
            raise ValueError("layer-stack bucket tensor must have shape [batch, 1] or [batch]")
        if indices.dtype not in {
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
        }:
            raise ValueError("layer-stack bucket tensor must be integral")
        indices = indices.long()
        if bool(((indices < 0) | (indices >= self.num_ls_buckets)).any().detach().item()):
            raise ValueError("layer-stack bucket index is outside [0, 16)")
        weights = torch.as_tensor(self.head_balance(), device=device, dtype=dtype)
        current = weights[indices]
        if layer_stack_blend is None:
            return current.unsqueeze(1)
        alpha = torch.as_tensor(layer_stack_blend, device=device, dtype=dtype)
        if alpha.ndim == 2 and alpha.shape[1] == 1:
            alpha = alpha[:, 0]
        if alpha.ndim != 1 or alpha.shape[0] != indices.shape[0]:
            raise ValueError("layer-stack blend tensor must have shape [batch, 1] or [batch]")
        if not bool(torch.isfinite(alpha).all().detach().item()) or bool(((alpha < 0) | (alpha > 1)).any().detach().item()):
            raise ValueError("layer-stack blend must be finite and in [0, 1]")
        next_weights = weights[torch.minimum(indices + 1, indices.new_tensor(self.num_ls_buckets - 1))]
        alpha = torch.where(indices == self.num_ls_buckets - 1, torch.zeros_like(alpha), alpha)
        return ((1.0 - alpha) * current + alpha * next_weights).unsqueeze(1)

    def _clip_weights(self) -> None:
        with torch.no_grad():
            for group in self.weight_clipping:
                for parameter in group["params"]:
                    data = parameter.data
                    minimum, maximum = group.get("min_weight"), group.get("max_weight")
                    virtual = group.get("virtual_params")
                    if virtual is not None:
                        repeats = data.shape[0] // virtual.shape[0]
                        expanded = virtual.data.repeat(repeats, 1)
                        if minimum is not None:
                            data.copy_(torch.maximum(data, data.new_full(data.shape, minimum) - expanded))
                        if maximum is not None:
                            data.copy_(torch.minimum(data, data.new_full(data.shape, maximum) - expanded))
                    else:
                        data.clamp_(minimum, maximum)

    def set_feature_set(self, new_feature_set: features_v8.FeatureSetV8) -> None:
        if new_feature_set.name == self.feature_set.name:
            return
        if self.feature_set.name == "HalfKAv2_hm_jieqi_v8" and new_feature_set.name == "HalfKAv2_hm_jieqi_v8^":
            with torch.no_grad():
                padding = self.input.weight.new_zeros((features_v8.VIRTUAL_INPUTS, self.input.weight.shape[1]))
                self.input.weight = nn.Parameter(torch.cat([self.input.weight, padding], dim=0))
            self.feature_set = new_feature_set
            return
        raise ValueError(f"cannot change V8 feature set from {self.feature_set.name} to {new_feature_set.name}")

    def forward_components(self, us, them, white_indices, white_values,
                           black_indices, black_values, psqt_indices,
                           layer_stack_indices, layer_stack_blend=None) -> ForwardComponents:
        wp, bp = self.input(white_indices, white_values, black_indices, black_values)
        w, w_psqt = torch.split(wp, self.ft_dim, dim=1)
        b, b_psqt = torch.split(bp, self.ft_dim, dim=1)
        if self.pairwise_ft:
            w0, w1 = torch.split(torch.clamp(w, 0.0, 1.0), self.l1_dim, dim=1)
            b0, b1 = torch.split(torch.clamp(b, 0.0, 1.0), self.l1_dim, dim=1)
            w, b = w0 * w1, b0 * b1
        batch_size = int(w.shape[0])
        def column(value, name):
            value = torch.as_tensor(value, device=w.device, dtype=w.dtype)
            if value.ndim == 1 and value.shape[0] == batch_size:
                value = value.unsqueeze(1)
            if tuple(value.shape) != (batch_size, 1):
                raise ValueError(f"V8 {name} must have shape [batch, 1]")
            return value
        us = column(us, "us")
        them = column(them, "them")
        accumulator = (us * torch.cat([w, b], dim=1)) + (them * torch.cat([b, w], dim=1))
        clipped = torch.clamp(accumulator, 0.0, 1.0)

        psqt_indices = torch.as_tensor(psqt_indices, device=w.device)
        if psqt_indices.ndim == 2 and tuple(psqt_indices.shape) == (batch_size, 1):
            psqt_indices = psqt_indices[:, 0]
        elif psqt_indices.ndim != 1 or psqt_indices.shape[0] != batch_size:
            raise ValueError("V8 PSQT bucket tensor must have shape [batch, 1] or [batch]")
        if psqt_indices.dtype not in {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}:
            raise ValueError("V8 PSQT bucket tensor must be integral")
        psqt_indices = psqt_indices.long()
        if bool(((psqt_indices < 0) | (psqt_indices >= self.num_psqt_buckets)).any().detach().item()):
            raise ValueError("V8 PSQT bucket index is outside [0, 16)")
        # Native V8 stores all PSQT slots in the accumulator.  With a blend,
        # use the layer-stack floor/alpha for the paired slots; batches without
        # a blend continue to use their explicit psqt_indices unchanged.
        if layer_stack_blend is not None:
            psqt_floor = torch.as_tensor(layer_stack_indices, device=w.device)
            if psqt_floor.ndim == 2 and tuple(psqt_floor.shape) == (batch_size, 1):
                psqt_floor = psqt_floor[:, 0]
            elif psqt_floor.ndim != 1 or psqt_floor.shape[0] != batch_size:
                raise ValueError("V8 layer-stack bucket tensor must have shape [batch, 1] or [batch]")
            psqt_floor = psqt_floor.long()
        else:
            psqt_floor = psqt_indices
        alpha = None
        if layer_stack_blend is not None:
            alpha = torch.as_tensor(layer_stack_blend, device=w.device, dtype=w.dtype)
            if alpha.ndim == 2 and tuple(alpha.shape) == (batch_size, 1):
                alpha = alpha[:, 0]
            elif alpha.ndim != 1 or alpha.shape[0] != batch_size:
                raise ValueError("V8 layer-stack blend tensor must have shape [batch, 1] or [batch]")
            if not bool(torch.isfinite(alpha).all().detach().item()) or bool(((alpha < 0) | (alpha > 1)).any().detach().item()):
                raise ValueError("V8 layer-stack blend must be finite and in [0, 1]")
            alpha = torch.where(psqt_floor == self.num_psqt_buckets - 1, torch.zeros_like(alpha), alpha)
        psqt0 = w_psqt.gather(1, psqt_floor.unsqueeze(1)) - b_psqt.gather(1, psqt_floor.unsqueeze(1))
        if alpha is None:
            psqt_delta = psqt0 * (us - 0.5)
        else:
            psqt_next = torch.minimum(psqt_floor + 1, psqt_floor.new_tensor(self.num_psqt_buckets - 1))
            psqt1 = w_psqt.gather(1, psqt_next.unsqueeze(1)) - b_psqt.gather(1, psqt_next.unsqueeze(1))
            psqt_delta = (psqt0 + (psqt1 - psqt0) * alpha.unsqueeze(1)) * (us - 0.5)
        head = self.layer_stacks.forward_interpolated(clipped, layer_stack_indices, layer_stack_blend)
        prediction = head + psqt_delta
        return ForwardComponents(accumulator=accumulator, psqt=psqt_delta,
                                 head=head, prediction=prediction)

    def forward(self, us, them, white_indices, white_values, black_indices,
                black_values, psqt_indices, layer_stack_indices,
                layer_stack_blend=None):
        return self.forward_components(
            us, them, white_indices, white_values, black_indices, black_values,
            psqt_indices, layer_stack_indices, layer_stack_blend).prediction

    @staticmethod
    def _unpack_batch(batch):
        if isinstance(batch, dict):
            values = [batch[key] for key in ("us", "them", "white_indices", "white_values", "black_indices", "black_values", "outcome", "score", "psqt_indices", "layer_stack_indices")]
            values.append(batch.get("eval_weight"))
            values.append(batch.get("layer_stack_blend"))
            return tuple(values)
        if len(batch) == 10:
            return tuple(batch) + (None, None)
        if len(batch) == 11:
            return tuple(batch) + (None,)
        if len(batch) == 12:
            return tuple(batch)
        raise ValueError("V8 batch must contain 10, 11, or 12 tensors")

    def step_(self, batch, loss_type: str):
        self._clip_weights()
        us, them, white_indices, white_values, black_indices, black_values, outcome, score, psqt_indices, layer_stack_indices, eval_weight, layer_stack_blend = self._unpack_batch(batch)
        prediction = self(us, them, white_indices, white_values, black_indices, black_values, psqt_indices, layer_stack_indices, layer_stack_blend)
        if prediction.ndim == 1:
            prediction = prediction.unsqueeze(1)
        if prediction.ndim != 2 or prediction.shape[1] != 1:
            raise ValueError(
                f"V8 prediction must have shape [batch, 1], got {tuple(prediction.shape)}")
        batch_size = int(prediction.shape[0])

        def target_column(value, name: str) -> torch.Tensor:
            value = torch.as_tensor(value, device=prediction.device,
                                    dtype=prediction.dtype)
            if value.ndim == 1 and value.shape[0] == batch_size:
                value = value.unsqueeze(1)
            if tuple(value.shape) != tuple(prediction.shape):
                raise ValueError(
                    f"V8 {name} must have shape [batch, 1], got {tuple(value.shape)}")
            return value

        outcome = target_column(outcome, "outcome")
        score = target_column(score, "score")
        q = (prediction * 600.0 / 361.0).sigmoid()
        p = (score / 410.0).sigmoid()
        eval_loss = (p - q).square()
        result_loss = (q - outcome).square()
        loss = self.lambda_ * eval_loss + (1.0 - self.lambda_) * result_loss
        if eval_weight is not None or layer_stack_blend is not None or getattr(self, "_v81_head_balance_configured", False):
            if eval_weight is None:
                weight = torch.ones_like(loss)
            else:
                weight = eval_weight.to(loss.device, dtype=loss.dtype)
                if weight.ndim == 1 and weight.shape[0] == loss.shape[0]:
                    weight = weight.unsqueeze(1)
                if tuple(weight.shape) != tuple(loss.shape):
                    raise ValueError("V8 eval_weight must have shape [batch, 1]")
                if not bool(torch.isfinite(weight).all().detach().item()):
                    raise ValueError("V8 eval_weight must be finite")
                weight = weight.clamp_min(0.0)
            weight = weight * self._effective_head_weights(
                layer_stack_indices, layer_stack_blend,
                dtype=loss.dtype, device=loss.device)
            denominator = weight.sum()
            if bool((denominator > 0).detach().item()):
                loss = (loss * weight).sum() / denominator
            else:
                loss = loss.sum() * 0.0
        else:
            loss = loss.mean()
        if hasattr(self, "log"):
            self.log(loss_type, loss, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step_(batch, "train_loss")

    def validation_step(self, batch, batch_idx):
        return self.step_(batch, "val_loss")

    def test_step(self, batch, batch_idx):
        return self.step_(batch, "test_loss")

    def configure_optimizers(self):
        lr = self.lr
        parameters = [
            {"params": get_parameters([self.input]), "lr": lr},
            {"params": [self.layer_stacks.l1_fact.weight], "lr": lr},
            {"params": [self.layer_stacks.l1.weight, self.layer_stacks.l1.bias], "lr": lr},
            {"params": [self.layer_stacks.l2.weight, self.layer_stacks.l2.bias], "lr": lr},
            {"params": [self.layer_stacks.output.weight, self.layer_stacks.output.bias], "lr": lr / 10.0},
        ]
        if ranger21 is not None and self.ranger21_num_batches_per_epoch and self.ranger21_num_epochs:
            optimizer = ranger21.Ranger21(parameters, lr=1.0, betas=(0.9, 0.999), eps=1e-7,
                                          num_batches_per_epoch=self.ranger21_num_batches_per_epoch,
                                          num_epochs=self.ranger21_num_epochs, use_warmup=False,
                                          warmdown_active=False, using_gc=False, using_normgc=False,
                                          use_adaptive_gradient_clipping=False, softplus=False,
                                          pnm_momentum_factor=0.0, weight_decay=0.0, logging_active=False)
        else:
            optimizer = torch.optim.AdamW(parameters, lr=1.0, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.987)
        return [optimizer], [scheduler]

    def architecture_contract(self) -> dict[str, object]:
        return {
            "ft_dim": self.ft_dim,
            "l1": self.l1_dim,
            "l2": self.l2_dim,
            "l3": self.l3_dim,
            "psqt_buckets": self.num_psqt_buckets,
            "layer_stacks": self.num_ls_buckets,
            "real_features": self.feature_set.num_real_features,
            "training_features": self.feature_set.num_features,
            "metadata_layout": "unknown-loss-threat-summary-v1",
            "layer_stack_selection": "continuous-q0.8-v1",
            "layer_stack_q_format": "Q0.8",
        }


def load_v8_checkpoint(path: str, feature_set: features_v8.FeatureSetV8, *, map_location: str = "cpu") -> NNUE:
    """Load a checkpoint only after the caller has validated its V8 sidecar."""

    if pl is not None and str(path).lower().endswith(".ckpt"):
        return NNUE.load_from_checkpoint(path, feature_set=feature_set, map_location=map_location)
    value = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(value, NNUE):
        raise TypeError("V8 checkpoint does not contain model_v8.NNUE")
    value.set_feature_set(feature_set)
    return value
