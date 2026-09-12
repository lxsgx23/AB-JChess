"""Provenance and format policy for AB-JChess V8 training weights.

V8 intentionally has a separate sidecar schema. Runtime ``.nnue`` files and
unsigned pickles are never accepted at a training-weight loading boundary.
Validation hashes the payload before the caller invokes ``torch.load``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import features_v8


SCHEMA = "abjchess-v8.1-training-weight-v1"
ENGINE = "abjchessV8.1"
SIDECAR_SUFFIX = ".abjv8.json"
HEX64_ZERO = "0" * 64
INTERPOLATION_SCHEMA = "continuous-q0.8-v1"


def expected_architecture(feature_name: str) -> dict[str, int | str]:
    """Return the closed tensor contract for a V8 training feature name."""

    if feature_name not in features_v8.FEATURE_NAMES:
        raise ValueError(f"unsupported V8 feature {feature_name!r}")
    return {
        "ft_dim": 2048,
        "l1": 1024,
        "l2": 15,
        "l3": 32,
        "psqt_buckets": 16,
        "layer_stacks": 16,
        "real_features": features_v8.REAL_INPUTS,
        "training_features": features_v8.TOTAL_INPUTS if feature_name.endswith("^") else features_v8.REAL_INPUTS,
        "metadata_layout": "unknown-loss-threat-summary-v1",
        "layer_stack_selection": INTERPOLATION_SCHEMA,
        "layer_stack_q_format": "Q0.8",
    }


class WeightPolicyError(ValueError):
    pass


@dataclass(frozen=True)
class WeightProvenance:
    schema: str
    engine: str
    weight_format: str
    feature_name: str
    feature_identity_sha256: str
    architecture: Mapping[str, Any]
    payload_sha256: str
    teacher_dataset_sha256: str
    corpus_sha256: str | None
    source: Path
    head_balance: Mapping[str, Any] | None = None
    independent_init: Mapping[str, Any] | None = None
    source_manifest_sha256: str | None = None
    selection: Mapping[str, Any] | None = None


def sidecar_path(payload: str | Path) -> Path:
    path = Path(payload)
    return path.with_name(path.name + SIDECAR_SUFFIX)


def _is_hash(value: object, *, allow_zero: bool = False) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value) and (allow_zero or value != HEX64_ZERO)


def _fail(path: Path, message: str) -> WeightPolicyError:
    return WeightPolicyError(f"{path}: {message}")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
    except OSError as exc:
        raise _fail(path, f"cannot hash payload: {exc}") from exc
    return digest.hexdigest()


def _read_sidecar(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise _fail(path, f"cannot read sidecar: {exc}") from exc

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise _fail(path, f"duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=reject_duplicates)
    except WeightPolicyError:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise _fail(path, f"invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise _fail(path, "sidecar root must be an object")
    return value


def feature_identity_sha256(feature_set: object | str) -> str:
    return features_v8.feature_identity_sha256(feature_set)


def validate_sha256_text(value: str, *, field: str = "SHA-256") -> str:
    if not _is_hash(value):
        raise ValueError(f"{field} must be a non-zero lowercase SHA-256")
    return value


def _required_string(obj: Mapping[str, Any], key: str, path: Path) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value:
        raise _fail(path, f"{key} must be a non-empty string")
    return value


def validate_v8_weight(
    payload: str | Path,
    *,
    expected_feature_name: str | None = None,
    expected_format: str | None = None,
    expected_source_manifest_sha256: str | None = None,
    expected_head_balance_cap: float | None = None,
    expected_coverage_report_sha256: str | None = None,
    expected_head_balance_weights: Sequence[float] | None = None,
    expected_head_init_seed: int | None = None,
) -> WeightProvenance:
    path = Path(payload).resolve()
    if not path.is_file():
        raise _fail(path, "weight payload is not a regular file")
    if path.suffix.lower() not in {".pt", ".ckpt"}:
        raise _fail(path, "V8 accepts only .pt/.ckpt training weights; runtime .nnue is not a checkpoint")
    sidecar = sidecar_path(path)
    if not sidecar.is_file():
        raise _fail(path, f"missing V8 provenance sidecar {sidecar.name}")
    obj = _read_sidecar(sidecar)
    allowed = {
        "schema", "engine", "weight_format", "feature_set", "architecture",
        "payload_sha256", "teacher_dataset_sha256", "corpus_sha256", "source_manifest_sha256",
        "head_balance", "independent_init", "selection",
    }
    unknown = sorted(set(obj) - allowed)
    if unknown:
        raise _fail(sidecar, "unknown keys: " + ", ".join(unknown))
    if _required_string(obj, "schema", sidecar) != SCHEMA:
        raise _fail(sidecar, f"schema must be {SCHEMA!r}")
    if _required_string(obj, "engine", sidecar) != ENGINE:
        raise _fail(sidecar, f"engine must be {ENGINE!r}")
    weight_format = _required_string(obj, "weight_format", sidecar)
    if weight_format not in {"pytorch-module-v1", "pytorch-lightning-v1"}:
        raise _fail(sidecar, "unsupported weight_format")
    if expected_format is not None and weight_format != expected_format:
        raise _fail(sidecar, f"weight_format does not match {expected_format!r}")
    feature = obj.get("feature_set")
    if not isinstance(feature, dict) or set(feature) != {"name", "identity_sha256"}:
        raise _fail(sidecar, "feature_set must contain exactly name and identity_sha256")
    feature_name = _required_string(feature, "name", sidecar)
    if feature_name not in features_v8.FEATURE_NAMES:
        raise _fail(sidecar, f"unsupported V8 feature {feature_name!r}")
    feature_identity = feature.get("identity_sha256")
    if not _is_hash(feature_identity):
        raise _fail(sidecar, "feature_set.identity_sha256 must be lowercase SHA-256")
    if feature_identity != features_v8.feature_identity_sha256(feature_name):
        raise _fail(sidecar, "feature identity does not match the V8 contract")
    if expected_feature_name is not None and feature_name != expected_feature_name:
        raise _fail(sidecar, f"feature_set.name does not match {expected_feature_name!r}")
    architecture = obj.get("architecture")
    if not isinstance(architecture, dict) or not architecture:
        raise _fail(sidecar, "architecture must be a non-empty object")
    for key, value in architecture.items():
        if not isinstance(key, str) or not key or isinstance(value, bool) or not isinstance(value, (int, str)):
            raise _fail(sidecar, "architecture values must be integer or string")
        if isinstance(value, int) and value < 0:
            raise _fail(sidecar, "architecture values cannot be negative")
    expected = expected_architecture(feature_name)
    for key, value in expected.items():
        if architecture.get(key) != value:
            raise _fail(sidecar, f"architecture.{key} must be {value!r} for the V8 contract")
    selection = obj.get("selection")
    if (not isinstance(selection, dict)
            or set(selection) != {"formula", "q_format"}
            or selection.get("formula") != INTERPOLATION_SCHEMA
            or selection.get("q_format") != "Q0.8"):
        raise _fail(sidecar, "selection must declare continuous-q0.8-v1")
    source_manifest_hash = obj.get("source_manifest_sha256")
    if source_manifest_hash is not None and not _is_hash(source_manifest_hash):
        raise _fail(sidecar, "source_manifest_sha256 must be lowercase SHA-256")
    if (expected_source_manifest_sha256 is not None
            and source_manifest_hash != validate_sha256_text(
                expected_source_manifest_sha256,
                field="expected_source_manifest_sha256")):
        raise _fail(sidecar, "source_manifest_sha256 does not match the current run")
    balance = obj.get("head_balance")
    if not isinstance(balance, dict):
        raise _fail(sidecar, "head_balance provenance is required for the V8 contract")
    if set(balance) - {"cap", "weights", "coverage_report_sha256"}:
        raise _fail(sidecar, "head_balance contains unknown fields")
    try:
        balance_cap = float(balance.get("cap"))
    except (TypeError, ValueError) as exc:
        raise _fail(sidecar, "head_balance.cap is invalid") from exc
    if isinstance(balance.get("cap"), bool) or not math.isfinite(balance_cap) or balance_cap <= 0.0:
        raise _fail(sidecar, "head_balance.cap is invalid")
    if expected_head_balance_cap is not None:
        try:
            expected_cap = float(expected_head_balance_cap)
        except (TypeError, ValueError) as exc:
            raise _fail(sidecar, "expected_head_balance_cap is invalid") from exc
        if not math.isfinite(expected_cap) or expected_cap <= 0.0 or balance_cap != expected_cap:
            raise _fail(sidecar, "head_balance.cap does not match the current run")
    if not isinstance(balance.get("weights"), list) or len(balance["weights"]) != 16:
        raise _fail(sidecar, "head_balance.weights must contain 16 values")
    coverage_hash = balance.get("coverage_report_sha256")
    if coverage_hash is not None and not _is_hash(coverage_hash):
        raise _fail(sidecar, "head_balance.coverage_report_sha256 must be lowercase SHA-256")
    if expected_coverage_report_sha256 is not None:
        expected_coverage_hash = validate_sha256_text(
            expected_coverage_report_sha256,
            field="expected_coverage_report_sha256")
        if coverage_hash != expected_coverage_hash:
            raise _fail(sidecar, "head_balance coverage report does not match the current run")
    if expected_coverage_report_sha256 is not None and coverage_hash is None:
        raise _fail(sidecar, "head_balance coverage report hash is required")
    try:
        values = [float(value) for value in balance["weights"]]
    except (TypeError, ValueError) as exc:
        raise _fail(sidecar, "head_balance.weights are invalid") from exc
    if (any(not math.isfinite(value) or value <= 0 for value in values)
            or abs(math.fsum(values) - 16.0) > 1e-6):
        raise _fail(sidecar, "head_balance.weights must be finite positive and sum to 16")
    if expected_head_balance_weights is not None:
        try:
            expected_weights = [float(value) for value in expected_head_balance_weights]
        except (TypeError, ValueError) as exc:
            raise _fail(sidecar, "expected_head_balance_weights are invalid") from exc
        if (len(expected_weights) != 16
                or any(not math.isfinite(value) or value <= 0.0 for value in expected_weights)
                or abs(math.fsum(expected_weights) - 16.0) > 1e-6):
            raise _fail(sidecar, "expected_head_balance_weights are invalid")
        if any(abs(a - b) > 1e-9 for a, b in zip(values, expected_weights)):
            raise _fail(sidecar, "head balance weights do not match the current run")
    independent = obj.get("independent_init")
    if (not isinstance(independent, dict)
            or set(independent) != {"seed", "formula"}
            or not isinstance(independent.get("seed"), int)
            or isinstance(independent.get("seed"), bool)
            or independent.get("formula") != "weyl-kaiming-orthogonal-v1"):
        raise _fail(sidecar, "independent_init must declare weyl-kaiming-orthogonal-v1 and an integer seed")
    if expected_head_init_seed is not None:
        if isinstance(expected_head_init_seed, bool) or not isinstance(expected_head_init_seed, int):
            raise _fail(sidecar, "expected_head_init_seed is invalid")
        if independent["seed"] != expected_head_init_seed:
            raise _fail(sidecar, "independent init seed does not match the current run")
    payload_hash = obj.get("payload_sha256")
    if not _is_hash(payload_hash):
        raise _fail(sidecar, "payload_sha256 must be lowercase SHA-256")
    observed = _hash_file(path)
    if observed != payload_hash:
        raise _fail(path, f"payload SHA-256 mismatch: expected {payload_hash}, got {observed}")
    teacher_hash = obj.get("teacher_dataset_sha256")
    if not _is_hash(teacher_hash):
        raise _fail(sidecar, "teacher_dataset_sha256 must identify the jqv4 training set")
    corpus_hash = obj.get("corpus_sha256")
    if corpus_hash is not None and not _is_hash(corpus_hash):
        raise _fail(sidecar, "corpus_sha256 must be a non-zero lowercase SHA-256")
    return WeightProvenance(
        schema=SCHEMA,
        engine=ENGINE,
        weight_format=weight_format,
        feature_name=feature_name,
        feature_identity_sha256=feature_identity,
        architecture=dict(architecture),
        payload_sha256=payload_hash,
        teacher_dataset_sha256=teacher_hash,
        corpus_sha256=corpus_hash,
        source=sidecar,
        head_balance=dict(balance),
        independent_init=dict(independent),
        source_manifest_sha256=source_manifest_hash,
        selection=dict(selection),
    )


def write_v8_sidecar(
    payload: str | Path,
    *,
    feature_name: str,
    feature_identity_sha256: str,
    architecture: Mapping[str, Any],
    teacher_dataset_sha256: str,
    corpus_sha256: str | None = None,
    weight_format: str = "pytorch-module-v1",
    head_balance_weights: list[float] | tuple[float, ...] | None = None,
    head_balance_cap: float = 8.0,
    head_init_seed: int = 0,
    coverage_report_sha256: str | None = None,
    source_manifest_sha256: str | None = None,
) -> Path:
    path = Path(payload).resolve()
    if not path.is_file():
        raise _fail(path, "cannot write sidecar for a missing payload")
    if feature_name not in features_v8.FEATURE_NAMES:
        raise _fail(path, f"unsupported V8 feature {feature_name!r}")
    expected_identity = features_v8.feature_identity_sha256(feature_name)
    if feature_identity_sha256 != expected_identity:
        raise _fail(path, "feature_identity_sha256 does not match V8 feature contract")
    if not _is_hash(teacher_dataset_sha256):
        raise _fail(path, "teacher_dataset_sha256 must be a non-zero SHA-256")
    if corpus_sha256 is not None and not _is_hash(corpus_sha256):
        raise _fail(path, "corpus_sha256 must be a non-zero SHA-256")
    if weight_format not in {"pytorch-module-v1", "pytorch-lightning-v1"}:
        raise _fail(path, "unsupported weight_format")
    expected = expected_architecture(feature_name)
    for key, value in expected.items():
        if architecture.get(key) != value:
            raise _fail(path, f"architecture.{key} must be {value!r} for the V8 contract")
    if isinstance(head_init_seed, bool) or not isinstance(head_init_seed, int):
        raise _fail(path, "head_init_seed must be an integer")
    try:
        cap_value = float(head_balance_cap)
    except (TypeError, ValueError) as exc:
        raise _fail(path, "head_balance_cap must be finite and positive") from exc
    if not math.isfinite(cap_value) or cap_value <= 0:
        raise _fail(path, "head_balance_cap must be finite and positive")
    if head_balance_weights is None:
        head_balance_weights = [1.0] * 16
    try:
        balance_values = [float(value) for value in head_balance_weights]
    except (TypeError, ValueError) as exc:
        raise _fail(path, "head_balance_weights are invalid") from exc
    if len(balance_values) != 16 or any(not math.isfinite(value) or value <= 0 for value in balance_values):
        raise _fail(path, "head_balance_weights must contain 16 finite positive values")
    if abs(sum(balance_values) - 16.0) > 1e-6:
        raise _fail(path, "head_balance_weights must sum to 16")
    if coverage_report_sha256 is not None and not _is_hash(coverage_report_sha256):
        raise _fail(path, "coverage_report_sha256 must be a lowercase SHA-256")
    if source_manifest_sha256 is not None and not _is_hash(source_manifest_sha256):
        raise _fail(path, "source_manifest_sha256 must be a lowercase SHA-256")
    document = {
        "schema": SCHEMA,
        "engine": ENGINE,
        "weight_format": weight_format,
        "feature_set": {"name": feature_name, "identity_sha256": feature_identity_sha256},
        "architecture": dict(architecture),
        "payload_sha256": _hash_file(path),
        "teacher_dataset_sha256": teacher_dataset_sha256,
        **({"corpus_sha256": corpus_sha256} if corpus_sha256 is not None else {}),
        **({"source_manifest_sha256": source_manifest_sha256}
           if source_manifest_sha256 is not None else {}),
        "selection": {"formula": INTERPOLATION_SCHEMA, "q_format": "Q0.8"},
        "head_balance": {
            "cap": cap_value,
            "weights": balance_values,
            **({"coverage_report_sha256": coverage_report_sha256} if coverage_report_sha256 is not None else {}),
        },
        "independent_init": {"seed": head_init_seed, "formula": "weyl-kaiming-orthogonal-v1"},
    }
    target = sidecar_path(path)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(target)
    return target


def validate_before_torch_load(
    payload: str | Path, *, expected_feature_name: str | None = None,
    expected_format: str | None = None, **kwargs) -> WeightProvenance:
    return validate_v8_weight(
        payload, expected_feature_name=expected_feature_name,
        expected_format=expected_format, **kwargs)
