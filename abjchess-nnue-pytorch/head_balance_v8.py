"""Head coverage and balancing policy for the V8 training contract.

The native stream reports how many samples select each of the sixteen phase
heads.  This module keeps the policy deliberately small and serialisable so
the exact values used by a run can be written to its provenance sidecar.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
from pathlib import Path
from typing import Any, Mapping, Sequence


HEAD_COUNT = 16
DEFAULT_CAP = 8.0
REPORT_SCHEMA = "abjchess-v8.1-head-coverage-v1"


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Encode report JSON deterministically for authentication."""

    return json.dumps(
        dict(value), ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _counts(values: Sequence[int]) -> list[int]:
    try:
        result = list(values)
    except TypeError as exc:
        raise ValueError("head coverage counts must contain exactly 16 values") from exc
    if len(result) != HEAD_COUNT:
        raise ValueError("head coverage counts must contain exactly 16 values")
    if any(isinstance(value, bool) or not isinstance(value, numbers.Integral) or int(value) < 0 for value in result):
        raise ValueError("head coverage counts must be non-negative integers")
    if not any(result):
        raise ValueError("head coverage counts must contain at least one positive count")
    return [int(value) for value in result]


def _cap_value(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("head balance cap must be finite and positive")
    try:
        cap_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("head balance cap must be finite and positive") from exc
    if not math.isfinite(cap_value) or cap_value <= 0.0:
        raise ValueError("head balance cap must be finite and positive")
    return cap_value


def compute_head_weights(counts: Sequence[int], cap: float = DEFAULT_CAP) -> tuple[float, ...]:
    """Return capped inverse-frequency weights normalised to sum to 16.

    ``count == 0`` is treated as one observation for the inverse-frequency
    calculation, as specified by the runtime contract.  The cap is applied
    before normalisation; this means an extreme/empty head cannot dominate
    the loss while still receiving a positive finite weight.
    """

    values = _counts(counts)
    cap_value = _cap_value(cap)
    total = float(sum(values))
    raw = [min(cap_value, total / max(float(value), 1.0)) for value in values]
    raw_sum = math.fsum(raw)
    if not math.isfinite(raw_sum) or raw_sum <= 0.0:
        raise ValueError("head balance normalization has no positive mass")
    result = tuple(16.0 * value / raw_sum for value in raw)
    if any(not math.isfinite(value) or value <= 0.0 for value in result):
        raise ValueError("head balance weights must be finite and positive")
    # A tolerance is intentional here: JSON round trips can move the final
    # ulp, while consumers need the semantic sum rather than bit identity.
    if abs(math.fsum(result) - 16.0) > 1e-9:
        raise ValueError("head balance weights do not normalize to 16")
    return result


def _hash_manifest(manifest_sha256: str) -> str:
    if not isinstance(manifest_sha256, str) or len(manifest_sha256) != 64:
        raise ValueError("manifest_sha256 must be a 64-character SHA-256")
    if any(ch not in "0123456789abcdef" for ch in manifest_sha256.lower()):
        raise ValueError("manifest_sha256 must be lowercase hexadecimal SHA-256")
    return manifest_sha256.lower()


def make_coverage_report(
    counts: Sequence[int],
    *,
    manifest_sha256: str,
    cap: float = DEFAULT_CAP,
) -> dict[str, Any]:
    """Create an authenticated, deterministic coverage report document."""

    values = _counts(counts)
    manifest = _hash_manifest(manifest_sha256)
    cap_value = _cap_value(cap)
    weights = compute_head_weights(values, cap_value)
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "manifest_sha256": manifest,
        "counts": values,
        "cap": cap_value,
        "weights": list(weights),
    }
    report["content_sha256"] = hashlib.sha256(_canonical_json_bytes(report)).hexdigest()
    return report


def validate_coverage_report(
    report: Mapping[str, Any],
    *,
    manifest_sha256: str | None = None,
    cap: float | None = None,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Validate a report and return its immutable counts and weights."""

    if not isinstance(report, Mapping) or report.get("schema") != REPORT_SCHEMA:
        raise ValueError(f"coverage report schema must be {REPORT_SCHEMA!r}")
    required = {"schema", "manifest_sha256", "counts", "cap", "weights", "content_sha256"}
    if set(report) != required:
        raise ValueError("coverage report contains missing or unknown fields")
    observed_manifest = _hash_manifest(report["manifest_sha256"])
    if report["manifest_sha256"] != observed_manifest:
        raise ValueError("coverage report manifest_sha256 must be lowercase hexadecimal")
    if manifest_sha256 is not None and observed_manifest != _hash_manifest(manifest_sha256):
        raise ValueError("coverage report manifest_sha256 does not match the dataset")
    values = _counts(report["counts"])
    try:
        report_cap = _cap_value(report["cap"])
    except ValueError as exc:
        raise ValueError("coverage report cap is invalid") from exc
    if cap is not None and report_cap != _cap_value(cap):
        raise ValueError("coverage report cap does not match the requested cap")
    expected = compute_head_weights(values, report_cap)
    try:
        supplied = tuple(float(value) for value in report["weights"])
    except (TypeError, ValueError) as exc:
        raise ValueError("coverage report weights are invalid") from exc
    if len(supplied) != HEAD_COUNT or any(not math.isfinite(value) for value in supplied):
        raise ValueError("coverage report weights must contain 16 finite values")
    if any(abs(a - b) > 1e-9 for a, b in zip(supplied, expected)):
        raise ValueError("coverage report weights do not match counts and cap")
    unsigned = dict(report)
    unsigned.pop("content_sha256", None)
    encoded = _canonical_json_bytes(unsigned)
    if report.get("content_sha256") != hashlib.sha256(encoded).hexdigest():
        raise ValueError("coverage report content_sha256 is invalid")
    return tuple(values), supplied


def load_coverage_report(
    path: str | Path,
    *,
    manifest_sha256: str | None = None,
    cap: float | None = None,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    source = Path(path)
    try:
        report = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read head coverage report {source}: {exc}") from exc
    return validate_coverage_report(report, manifest_sha256=manifest_sha256, cap=cap)


def write_coverage_report(path: str | Path, report: Mapping[str, Any]) -> Path:
    """Validate and atomically write a report document."""

    validate_coverage_report(report)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_bytes(_canonical_json_bytes(report) + b"\n")
    temporary.replace(target)
    return target


def coverage_report_sha256(path: str | Path) -> str:
    """Return the digest of the exact canonical report file bytes."""

    source = Path(path)
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read head coverage report {source}: {exc}") from exc
    try:
        report = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"coverage report is not valid JSON: {exc}") from exc
    validate_coverage_report(report)
    if raw != _canonical_json_bytes(report) + b"\n":
        raise ValueError("coverage report is not canonical JSON")
    return hashlib.sha256(raw).hexdigest()


__all__ = [
    "HEAD_COUNT", "DEFAULT_CAP", "REPORT_SCHEMA", "compute_head_weights",
    "make_coverage_report", "validate_coverage_report", "load_coverage_report",
    "write_coverage_report", "coverage_report_sha256",
]
