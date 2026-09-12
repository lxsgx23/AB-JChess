"""Export a trained model to the AB-JChess V8.2 runtime package.

The exporter emits the native four-chunk ABJNNUE container consumed by the
V8.2 engine. Virtual rows are coalesced once, at export time, so the runtime
always stores exactly 31,776 feature rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
from typing import Mapping

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - doctor can run without Torch
    torch = None

import features_v8
import model_v8
from weight_policy_v8 import WeightPolicyError, validate_before_torch_load


PACKAGE_VERSION = 82
PACKAGE_SCHEMA = "abjchess-v8.2-full4way-noaux-v1"
# The C++ loader compares all sixteen bytes of this package magic.
MAGIC = b"ABJCHESSV82" + b"\0" * 5
HEADER_SIZE = 24

ACCUMULATOR_WIDTH = 2048
PSQT_BUCKETS = 16
REAL_FEATURES = features_v8.REAL_INPUTS
TRANSFORMER_BIAS_SIZE = ACCUMULATOR_WIDTH * 2
FEATURE_WEIGHTS_SIZE = REAL_FEATURES * ACCUMULATOR_WIDTH * 2
PSQT_WEIGHTS_SIZE = REAL_FEATURES * PSQT_BUCKETS * 4
TRANSFORMER_PAYLOAD_SIZE = TRANSFORMER_BIAS_SIZE + FEATURE_WEIGHTS_SIZE + PSQT_WEIGHTS_SIZE
PRIMARY_SIZE = TRANSFORMER_PAYLOAD_SIZE
EVAL_HEAD_BUCKET_SIZE = 34_208
LAYER_STACKS = 16
EVAL_HEADS_SIZE = EVAL_HEAD_BUCKET_SIZE * LAYER_STACKS
PROBABILITY_SCORE_TO_MASS_SIZE = 4001 * 4
PROBABILITY_MASS_TO_SCORE_SIZE = 1901 * 4

L1_BIAS_OFFSET = 0x0000
L1_WEIGHT_OFFSET = 0x0040
L2_BIAS_OFFSET = 0x8080
L2_WEIGHT_OFFSET = 0x8100
FINAL_BIAS_OFFSET = 0x8540
FINAL_WEIGHT_OFFSET = 0x8580

CHUNK_NAMES = (
    "primary_runtime_nnue_container.bin",
    "eval_heads_runtime.bin",
    "probability_score_to_mass.i32le",
    "probability_mass_to_score.i32le",
)


class SerializeV8Error(ValueError):
    pass


def extract_runtime_tables(template: str | Path) -> dict[str, bytes]:
    """Extract only probability tables from an authenticated package template.

    Only the two probability chunks are copied; model and evaluation bytes
    from the template never enter the new package.
    """

    path = Path(template)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise SerializeV8Error(f"cannot read package template: {exc}") from exc
    if len(data) < HEADER_SIZE or data[:16] != b"ABJNNUE" + b"\0" * 9:
        raise SerializeV8Error("package template does not have the expected container magic")
    version, metadata_length = struct.unpack_from("<II", data, 16)
    if version != 3 or metadata_length == 0 or HEADER_SIZE + metadata_length > len(data):
        raise SerializeV8Error("package template header is invalid")
    try:
        metadata = json.loads(data[HEADER_SIZE:HEADER_SIZE + metadata_length].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SerializeV8Error(f"package template metadata is invalid: {exc}") from exc
    chunks = metadata.get("chunks")
    if not isinstance(chunks, list):
        raise SerializeV8Error("package template has no chunk table")
    payload_offset = HEADER_SIZE + metadata_length
    result: dict[str, bytes] = {}
    wanted = {
        CHUNK_NAMES[2]: PROBABILITY_SCORE_TO_MASS_SIZE,
        CHUNK_NAMES[3]: PROBABILITY_MASS_TO_SCORE_SIZE,
    }
    for item in chunks:
        if not isinstance(item, dict) or item.get("name") not in wanted:
            continue
        name = item["name"]
        try:
            offset = int(item["data_offset"])
            size = int(item["size"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SerializeV8Error(f"package template chunk metadata is invalid for {name}") from exc
        expected = wanted[name]
        if size != expected or offset < 0 or payload_offset + offset + size > len(data):
            raise SerializeV8Error(f"package template chunk {name} has unexpected size or bounds")
        value = data[payload_offset + offset:payload_offset + offset + size]
        expected_hash = item.get("sha256")
        if expected_hash is not None:
            if not isinstance(expected_hash, str) or hashlib.sha256(value).hexdigest() != expected_hash.lower():
                raise SerializeV8Error(f"package template chunk {name} SHA-256 mismatch")
        result[name] = value
    missing = [name for name in wanted if name not in result]
    if missing:
        raise SerializeV8Error("package template is missing: " + ", ".join(missing))
    return result


def validate_source_suffix(source: str | Path) -> None:
    suffix = Path(source).suffix.lower()
    if suffix == ".nnue":
        raise SerializeV8Error("V8 runtime packages cannot be used as PyTorch training checkpoints")
    if suffix not in {".pt", ".ckpt"}:
        raise SerializeV8Error("V8 exporter accepts only .pt or .ckpt sources")


def _require_torch() -> None:
    if torch is None:
        raise SerializeV8Error("Torch is required to export V8 weights")


def _tensor_numpy(value, dtype) -> np.ndarray:
    _require_torch()
    if not isinstance(value, torch.Tensor):
        raise SerializeV8Error("model tensor is not a Torch tensor")
    return value.detach().to(device="cpu", dtype=torch.float32).numpy().astype(dtype, copy=False)


def _quant_i16(value, scale: float) -> np.ndarray:
    raw = np.rint(np.asarray(value, dtype=np.float64) * scale)
    if np.any(raw < -32768) or np.any(raw > 32767):
        raise SerializeV8Error("int16 quantization overflow")
    return raw.astype("<i2")


def _quant_i32(value, scale: float) -> np.ndarray:
    raw = np.rint(np.asarray(value, dtype=np.float64) * scale)
    if np.any(raw < -(1 << 31)) or np.any(raw > (1 << 31) - 1):
        raise SerializeV8Error("int32 quantization overflow")
    return raw.astype("<i4")


def _quant_fc(layer, *, output: bool = False) -> tuple[np.ndarray, np.ndarray]:
    scale_bias = 9600.0 if output else 8128.0
    scale_weight = scale_bias / 127.0
    max_weight = 127.0 / scale_weight
    bias = _quant_i32(_tensor_numpy(layer.bias, np.float32), scale_bias)
    weights = np.asarray(_tensor_numpy(layer.weight, np.float32), dtype=np.float64)
    weights = np.clip(weights, -max_weight, max_weight)
    quantized = np.rint(weights * scale_weight).astype(np.int8)
    return bias, quantized


def _blocked_weights(weights: np.ndarray, input_width: int) -> bytes:
    """Pack [outputs, inputs] as [input groups][outputs][4 lanes]."""

    outputs, inputs = weights.shape
    if inputs > input_width or input_width % 4:
        raise SerializeV8Error("invalid native affine input width")
    padded = np.zeros((outputs, input_width), dtype=np.int8)
    padded[:, :inputs] = weights
    return padded.reshape(outputs, input_width // 4, 4).transpose(1, 0, 2).tobytes()


def pack_eval_head(l1, l2, output) -> bytes:
    """Quantize and pack one native 2048 -> 16 -> 32 -> 1 head."""

    _require_torch()
    if (l1.in_features, l1.out_features) != (2048, 16):
        raise SerializeV8Error("V8 L1 head must be Linear(2048, 16)")
    if (l2.in_features, l2.out_features) != (30, 32):
        raise SerializeV8Error("V8 L2 head must be Linear(30, 32)")
    if (output.in_features, output.out_features) != (32, 1):
        raise SerializeV8Error("V8 output head must be Linear(32, 1)")
    l1_bias, l1_weights = _quant_fc(l1)
    l2_bias, l2_weights = _quant_fc(l2)
    final_bias, final_weights = _quant_fc(output, output=True)
    payload = bytearray(EVAL_HEAD_BUCKET_SIZE)
    payload[L1_BIAS_OFFSET:L1_BIAS_OFFSET + l1_bias.nbytes] = l1_bias.tobytes()
    blocked_l1 = _blocked_weights(l1_weights, 2048)
    payload[L1_WEIGHT_OFFSET:L1_WEIGHT_OFFSET + len(blocked_l1)] = blocked_l1
    payload[L2_BIAS_OFFSET:L2_BIAS_OFFSET + l2_bias.nbytes] = l2_bias.tobytes()
    blocked_l2 = _blocked_weights(l2_weights, 32)
    payload[L2_WEIGHT_OFFSET:L2_WEIGHT_OFFSET + len(blocked_l2)] = blocked_l2
    payload[FINAL_BIAS_OFFSET:FINAL_BIAS_OFFSET + final_bias.nbytes] = final_bias.tobytes()
    payload[FINAL_WEIGHT_OFFSET:FINAL_WEIGHT_OFFSET + final_weights.nbytes] = final_weights.reshape(-1).tobytes()
    return bytes(payload)


def pack_primary(model: model_v8.NNUE) -> bytes:
    _require_torch()
    feature_set = model.feature_set
    if feature_set.num_real_features != REAL_FEATURES:
        raise SerializeV8Error(f"V8 runtime expects {REAL_FEATURES} real features, got {feature_set.num_real_features}")
    if (model.ft_dim, model.num_psqt_buckets) != (ACCUMULATOR_WIDTH, PSQT_BUCKETS):
        raise SerializeV8Error("V8 transformer dimensions are not 2048 + 16")
    bias = _quant_i16(_tensor_numpy(model.input.bias[:ACCUMULATOR_WIDTH], np.float32), 127.0)
    coalesced = model_v8.coalesce_ft_weights(model, model.input)
    if tuple(coalesced.shape) != (REAL_FEATURES, ACCUMULATOR_WIDTH + PSQT_BUCKETS):
        raise SerializeV8Error(f"coalesced transformer shape is {tuple(coalesced.shape)}")
    ft = _quant_i16(_tensor_numpy(coalesced[:, :ACCUMULATOR_WIDTH], np.float32), 127.0)
    psqt = _quant_i32(_tensor_numpy(coalesced[:, ACCUMULATOR_WIDTH:ACCUMULATOR_WIDTH + PSQT_BUCKETS], np.float32), 9600.0)
    payload = bytearray(PRIMARY_SIZE)
    cursor = 0
    payload[cursor:cursor + bias.nbytes] = bias.tobytes(); cursor += bias.nbytes
    payload[cursor:cursor + ft.nbytes] = ft.tobytes(); cursor += ft.nbytes
    payload[cursor:cursor + psqt.nbytes] = psqt.tobytes(); cursor += psqt.nbytes
    if cursor != PRIMARY_SIZE:
        raise AssertionError("V8 primary payload size calculation is inconsistent")
    return bytes(payload)


def pack_eval_heads(model: model_v8.NNUE) -> bytes:
    payload = bytearray()
    for l1, l2, output in model.layer_stacks.get_coalesced_layer_stacks():
        payload.extend(pack_eval_head(l1, l2, output))
    if len(payload) != EVAL_HEADS_SIZE:
        raise SerializeV8Error("V8 eval-head payload has the wrong size")
    return bytes(payload)


def _optional_payload(value: bytes | bytearray | str | Path | None, size: int, name: str) -> bytes:
    if value is None:
        return bytes(size)
    if isinstance(value, (str, Path)):
        try:
            value = Path(value).read_bytes()
        except OSError as exc:
            raise SerializeV8Error(f"cannot read {name}: {exc}") from exc
    value = bytes(value)
    if len(value) != size:
        raise SerializeV8Error(f"{name} must contain exactly {size} bytes")
    return value


def _runtime_table_payloads(*, probability_score_to_mass=None, probability_mass_to_score=None,
                            template_package=None,
                            smoke_only: bool = False) -> dict[str, bytes]:
    """Resolve probability chunks without silently manufacturing tables.

    A normal runtime package needs both probability lookup tables because the
    engine uses them for hidden-position aggregation.
    """

    values = {
        CHUNK_NAMES[2]: probability_score_to_mass,
        CHUNK_NAMES[3]: probability_mass_to_score,
    }
    sizes = {
        CHUNK_NAMES[2]: PROBABILITY_SCORE_TO_MASS_SIZE,
        CHUNK_NAMES[3]: PROBABILITY_MASS_TO_SCORE_SIZE,
    }
    if template_package is not None:
        tables = extract_runtime_tables(template_package)
        for name in values:
            if values[name] is None:
                values[name] = tables[name]

    probability_names = (CHUNK_NAMES[2], CHUNK_NAMES[3])
    missing = [name for name in probability_names if values[name] is None]
    if missing:
        if not smoke_only:
            names = ", ".join(missing)
            raise SerializeV8Error(
                "V8 export requires runtime tables (missing: " + names
                + "); provide --template-package or both probability table paths, or use "
                  "--smoke-only for a structural zero-table package"
            )
        if len(missing) != len(probability_names):
            raise SerializeV8Error(
                "--smoke-only cannot mix supplied and missing probability tables"
            )
        for name in probability_names:
            values[name] = bytes(sizes[name])

    return {
        name: _optional_payload(values[name], sizes[name], name)
        for name in values
    }


def build_chunks(model: model_v8.NNUE, *, probability_score_to_mass=None,
                 probability_mass_to_score=None, template_package=None,
                 smoke_only: bool = False) -> dict[str, bytes]:
    tables = _runtime_table_payloads(
        probability_score_to_mass=probability_score_to_mass,
        probability_mass_to_score=probability_mass_to_score,
        template_package=template_package,
        smoke_only=smoke_only,
    )
    return {
        CHUNK_NAMES[0]: pack_primary(model),
        CHUNK_NAMES[1]: pack_eval_heads(model),
        CHUNK_NAMES[2]: tables[CHUNK_NAMES[2]],
        CHUNK_NAMES[3]: tables[CHUNK_NAMES[3]],
    }


def _chunk_metadata(chunks: Mapping[str, bytes]) -> list[dict[str, object]]:
    offset = 0
    output = []
    for name in CHUNK_NAMES:
        payload = chunks[name]
        output.append({
            "name": name,
            "role": "runtime",
            "dtype": "u8" if name.endswith(".bin") else "i32le",
            "data_offset": offset,
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })
        offset += len(payload)
    return output


def build_package_bytes(model: model_v8.NNUE, *, description: str | None = None,
                        probability_score_to_mass=None, probability_mass_to_score=None,
                        template_package=None, smoke_only: bool = False) -> bytes:
    chunks = build_chunks(model, probability_score_to_mass=probability_score_to_mass,
                          probability_mass_to_score=probability_mass_to_score,
                          template_package=template_package, smoke_only=smoke_only)
    feature_name = model.feature_set.name
    metadata = {
        "format_version": PACKAGE_VERSION,
        "schema": PACKAGE_SCHEMA,
        # Runtime packages have one closed feature ABI.  The training side
        # may use either the plain or factorized name, but both are coalesced
        # to this exact native identity before export.
        "feature_identity": "HalfKAv2_hm_jieqi_v8.1_full4way_meta_midmirror_threat_interp",
        "feature_identity_sha256": features_v8.feature_identity_sha256(feature_name),
        "feature_dimensions": REAL_FEATURES,
        "accumulator_width": ACCUMULATOR_WIDTH,
        "psqt_buckets": PSQT_BUCKETS,
        "layer_stacks": LAYER_STACKS,
        "interpolation": {"format": "q0.8", "rounding": "+127/255", "version": "v1"},
        "interpolation_format": "Q0.8",
        "interpolation_formula": "continuous-q0.8-v1",
        "description": description or f"AB-JChess V8 NNUE ({feature_name})",
        "smoke_only": bool(smoke_only),
        "architecture": model.architecture_contract(),
        "payload_bytes": sum(len(chunks[name]) for name in CHUNK_NAMES),
        "chunks": _chunk_metadata(chunks),
    }
    metadata_bytes = json.dumps(metadata, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload = b"".join(chunks[name] for name in CHUNK_NAMES)
    return MAGIC + struct.pack("<II", PACKAGE_VERSION, len(metadata_bytes)) + metadata_bytes + payload


def export_model(model: model_v8.NNUE, target: str | Path, **kwargs) -> Path:
    target = Path(target).resolve()
    if target.suffix.lower() != ".nnue":
        raise SerializeV8Error("V8 runtime output must use the .nnue extension")
    data = build_package_bytes(model, **kwargs)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(target)
    return target


def load_source(source: str | Path, feature_set: features_v8.FeatureSetV8) -> model_v8.NNUE:
    validate_source_suffix(source)
    try:
        provenance = validate_before_torch_load(source, expected_feature_name=feature_set.name)
    except WeightPolicyError as exc:
        raise SerializeV8Error(f"V8 weight policy rejected source: {exc}") from exc
    return model_v8.load_v8_checkpoint(str(source), feature_set)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Export authenticated PyTorch V8 weights to an ABJNNUE V8 package")
    parser.add_argument("source", help="V8 .pt or .ckpt checkpoint with .abjv8.json sidecar")
    parser.add_argument("target", help="V8 runtime package (.nnue)")
    features_v8.add_argparse_args(parser)
    parser.add_argument("--description", default=None)
    parser.add_argument("--template-package", default=None,
                        help="authenticated package used only for probability tables")
    parser.add_argument("--probability-score-to-mass", default=None)
    parser.add_argument("--probability-mass-to-score", default=None)
    parser.add_argument(
        "--smoke-only", action="store_true",
        help="allow all-zero probability tables for structural tests; never use for play",
    )
    args = parser.parse_args(argv)
    feature_set = features_v8.get_feature_set_from_name(args.features)
    model = load_source(args.source, feature_set)
    model.eval()
    export_model(model, args.target, description=args.description, template_package=args.template_package,
                 probability_score_to_mass=args.probability_score_to_mass,
                 probability_mass_to_score=args.probability_mass_to_score,
                 smoke_only=args.smoke_only)
    print(f"wrote V8 package {Path(args.target).resolve()}")


if __name__ == "__main__":
    main()
