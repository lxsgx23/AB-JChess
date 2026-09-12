"""Train the closed AB-JChess V8 NNUE on jqv4 binary data.

This entry point uses only the V8 model, feature registry, and provenance
writer in this source tree. The C++ loader accepts only the two
V8 feature names and emits the 31,776-row ABI consumed by ``model_v8.NNUE``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import uuid
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torch import set_num_threads as t_set_num_threads
from torch.utils.data import DataLoader

import features_v8
import model_v8
import nnue_dataset
from head_balance_v8 import (
    DEFAULT_CAP,
    coverage_report_sha256,
    load_coverage_report,
)
from ddp_utils import (
    batches_per_rank,
    inspect_v3_shard,
    requested_gpu_count,
    resolve_distributed_context,
)
from weight_policy_v8 import (
    WeightPolicyError,
    validate_before_torch_load,
    validate_sha256_text,
    write_v8_sidecar,
)


def validate_lambda_value(value: float) -> float:
    """Validate the eval/result loss interpolation coefficient."""

    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("lambda must be between 0 and 1")
    return result


def checkpoint_architecture(model: model_v8.NNUE) -> dict[str, object]:
    return {
        **model.architecture_contract(),
        "feature_identity_sha256": features_v8.feature_identity_sha256(model.feature_set.name),
        "metadata_layout": "unknown-loss-threat-summary-v1",
        "layer_stack_selection": "continuous-q0.8-v1",
        "layer_stack_q_format": "Q0.8",
    }


class ABJChessV8ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def __init__(self, *args, provenance_feature_set=None,
                 teacher_dataset_sha256: str | None = None,
                 corpus_sha256: str | None = None,
                 source_manifest_sha256: str | None = None,
                 coverage_report_sha256_value: str | None = None,
                 head_balance_cap: float | None = None,
                 keep_last_only: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.saved_checkpoint_paths: list[Path] = []
        self.provenance_feature_set = provenance_feature_set
        self.teacher_dataset_sha256 = (
            validate_sha256_text(teacher_dataset_sha256,
                                 field="teacher_dataset_sha256")
            if teacher_dataset_sha256 is not None else None)
        self.corpus_sha256 = (
            validate_sha256_text(corpus_sha256, field="corpus_sha256")
            if corpus_sha256 is not None else None)
        self.source_manifest_sha256 = source_manifest_sha256
        self.coverage_report_sha256 = coverage_report_sha256_value
        self.head_balance_cap = head_balance_cap
        self.keep_last_only = bool(keep_last_only)

    def _save_checkpoint(self, trainer, filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)
        path = Path(filepath)
        if path.is_file():
            self.saved_checkpoint_paths.append(path)
            # Authenticate each payload immediately after Lightning writes it.
            # This keeps an interrupted run resumable even when on_train_end
            # is never delivered (for example after a worker or host failure).
            if self.provenance_feature_set is not None:
                model = getattr(trainer, "lightning_module", None)
                if model is None:
                    raise RuntimeError(
                        "cannot write ABJChess V8 checkpoint provenance without the Lightning module")
                if self.teacher_dataset_sha256 is None:
                    raise RuntimeError(
                        "teacher_dataset_sha256 is required for ABJChess V8 checkpoint provenance")
                write_checkpoint_sidecars(
                    [path], model=model,
                    feature_set=self.provenance_feature_set,
                    teacher_dataset_sha256=self.teacher_dataset_sha256,
                    corpus_sha256=self.corpus_sha256,
                    source_manifest_sha256=self.source_manifest_sha256,
                    coverage_report_sha256_value=self.coverage_report_sha256,
                    head_balance_cap=self.head_balance_cap)
            if self.keep_last_only:
                self._prune_checkpoint_dir(path)

    @staticmethod
    def _prune_checkpoint_dir(current: Path) -> None:
        """Keep only Lightning's resumable ``last.ckpt`` plus its sidecar.

        Lightning can call this hook once for an epoch checkpoint and once for
        ``last.ckpt``.  Preserve the file currently being written in either
        case, then remove older payloads and matching provenance sidecars.
        """
        directory = current.parent
        last = directory / "last.ckpt"
        keep = {current.resolve()}
        if last.is_file():
            keep.add(last.resolve())
        for candidate in directory.glob("*.ckpt"):
            if candidate.resolve() not in keep:
                candidate.unlink(missing_ok=True)
                candidate.with_name(candidate.name + ".abjv8.json").unlink(
                    missing_ok=True)
        for sidecar in directory.glob("*.ckpt.abjv8.json"):
            payload = sidecar.name[:-len(".abjv8.json")]
            payload_path = directory / payload
            if payload_path.resolve() not in keep:
                sidecar.unlink(missing_ok=True)


def _checkpoint_paths(checkpoint_callback) -> list[Path]:
    return list(getattr(checkpoint_callback, "saved_checkpoint_paths", ()))


def write_checkpoint_sidecars(checkpoint_paths, *, model, feature_set,
                              teacher_dataset_sha256: str,
                              corpus_sha256: str | None = None,
                              source_manifest_sha256: str | None = None,
                              coverage_report_sha256_value: str | None = None,
                              head_balance_cap: float | None = None) -> list[Path]:
    teacher_hash = validate_sha256_text(
        teacher_dataset_sha256, field="teacher_dataset_sha256")
    corpus_hash = (
        validate_sha256_text(corpus_sha256, field="corpus_sha256")
        if corpus_sha256 is not None else None)
    architecture = checkpoint_architecture(model)
    written: list[Path] = []
    seen: set[Path] = set()
    for raw_path in checkpoint_paths:
        path = Path(raw_path).resolve()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        balance_cap = (
            float(head_balance_cap)
            if head_balance_cap is not None
            else float(getattr(model, "_v81_head_balance_cap", DEFAULT_CAP))
        )
        write_v8_sidecar(
            path,
            feature_name=feature_set.name,
            feature_identity_sha256=features_v8.feature_identity_sha256(feature_set.name),
            architecture=architecture,
            teacher_dataset_sha256=teacher_hash,
            corpus_sha256=corpus_hash,
            weight_format="pytorch-lightning-v1",
            head_balance_weights=list(model.head_balance()) if hasattr(model, "head_balance") else None,
            head_balance_cap=balance_cap,
            head_init_seed=int(getattr(model, "_v81_head_init_seed", 0)),
            source_manifest_sha256=source_manifest_sha256,
            coverage_report_sha256=coverage_report_sha256_value,
        )
        written.append(path.with_name(path.name + ".abjv8.json"))
    return written


class ABJChessV8SidecarCallback(pl.callbacks.Callback):
    def __init__(self, checkpoint_callback, feature_set, teacher_dataset_sha256,
                 corpus_sha256: str | None = None,
                 source_manifest_sha256: str | None = None,
                 coverage_report_sha256_value: str | None = None,
                 head_balance_cap: float | None = None):
        super().__init__()
        self.checkpoint_callback = checkpoint_callback
        self.feature_set = feature_set
        self.teacher_dataset_sha256 = teacher_dataset_sha256
        self.corpus_sha256 = corpus_sha256
        self.source_manifest_sha256 = source_manifest_sha256
        self.coverage_report_sha256 = coverage_report_sha256_value
        self.head_balance_cap = head_balance_cap

    def on_train_end(self, trainer, pl_module):
        if not getattr(trainer, "is_global_zero", True):
            return
        model = getattr(trainer, "lightning_module", None) or pl_module
        write_checkpoint_sidecars(
            _checkpoint_paths(self.checkpoint_callback),
            model=model,
            feature_set=self.feature_set,
            teacher_dataset_sha256=self.teacher_dataset_sha256,
            corpus_sha256=self.corpus_sha256,
            source_manifest_sha256=self.source_manifest_sha256,
            coverage_report_sha256_value=self.coverage_report_sha256,
            head_balance_cap=self.head_balance_cap,
        )


class ABJChessV8EpochCallback(pl.callbacks.Callback):
    """Emit a stable, unbuffered epoch marker for PowerShell/log consumers."""

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        if getattr(trainer, "is_global_zero", True):
            total = int(getattr(trainer, "max_epochs", 0) or 0)
            current = int(getattr(trainer, "current_epoch", 0)) + 1
            print(f"[V8][epoch {current}/{total}] START", flush=True)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if getattr(trainer, "is_global_zero", True):
            total = int(getattr(trainer, "max_epochs", 0) or 0)
            current = int(getattr(trainer, "current_epoch", 0)) + 1
            print(f"[V8][epoch {current}/{total}] END", flush=True)


JQV4_MANIFEST_SCHEMA = "abjchess-v8-jqv4-manifest-v1"
JQV4_SHUFFLE_STRATEGY = "hierarchical-block-window-v1"
JQV4_U64_MAX = (1 << 64) - 1
JQV4_U32_MAX = (1 << 32) - 1


def _canonical_teacher_dataset_sha256(payload: dict) -> str:
    """Recompute the generator's provenance digest from manifest content."""
    files = payload.get("files")
    sources = []
    for source in files:
        sources.append({
            "shard": source["shard"],
            "relative_path": source["relative_path"],
            "record_start": source["record_start"],
            "record_end": source["record_end"],
            "bytes": source["bytes"],
            "sha256": source["sha256"],
            "dataset_uuid": source["dataset_uuid"],
            "file_uuid": source["file_uuid"],
            "metadata_sha256": source["metadata_sha256"],
            "feature_set": source["feature_set"],
        })
    sources.sort(key=lambda source: (
        source["record_start"], source["record_end"], source["file_uuid"]))
    split = payload["split"]
    shuffle = payload["shuffle"]
    canonical = {
        "sources": sources,
        "split": {
            "strategy": "whole-game-h64-v1",
            "seed": split["seed"],
            "numerator": split["numerator"],
            "denominator": split["denominator"],
        },
        "shuffle": {
            "strategy": JQV4_SHUFFLE_STRATEGY,
            "seed": shuffle["seed"],
            "buffer_blocks": shuffle["buffer_blocks"],
        },
    }
    encoded = json.dumps(
        canonical, ensure_ascii=True, sort_keys=True,
        separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_corpus_sha256(payload: dict) -> str:
    """Compute the sampling-independent identity of a complete JQv4 corpus."""

    files = payload.get("files")
    sources = []
    for source in files:
        sources.append({
            "shard": source["shard"],
            "relative_path": source["relative_path"],
            "record_start": source["record_start"],
            "record_end": source["record_end"],
            "bytes": source["bytes"],
            "sha256": source["sha256"],
            "dataset_uuid": source["dataset_uuid"],
            "file_uuid": source["file_uuid"],
            "metadata_sha256": source["metadata_sha256"],
            "feature_set": source["feature_set"],
        })
    sources.sort(key=lambda source: (
        source["record_start"], source["record_end"], source["file_uuid"]))
    encoded = json.dumps(
        {"sources": sources}, ensure_ascii=True, sort_keys=True,
        separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_jqv4_manifest(path: str | os.PathLike[str]) -> dict | None:
    """Read and structurally validate a JQv4 manifest.

    The native loader performs the expensive source SHA-256 verification when
    it opens the stream.  Python validates the cheap, deterministic contract
    first so malformed, mixed-partition, or missing-source manifests fail
    before Lightning starts constructing a model/checkpoint.
    """

    candidate = Path(path)
    lowered = candidate.name.lower()
    if candidate.suffix.lower() != ".json" and not lowered.endswith(
        (".jqv8.json", ".jqv4.json")
    ):
        return None
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        # A JSON-looking input is intended to be a manifest; don't silently
        # reinterpret it as a V3 binary path after a parse error.
        raise SystemExit(f"cannot read JQv4 manifest {candidate}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"JQv4 manifest root must be an object: {candidate}")
    if payload.get("schema") != JQV4_MANIFEST_SCHEMA:
        raise SystemExit(f"unsupported JQv4 manifest schema in {candidate}")
    if payload.get("input_kind") != "jqv4":
        raise SystemExit(f"unsupported JQv4 manifest input_kind in {candidate}")
    split = payload.get("split")
    if not isinstance(split, dict):
        raise SystemExit(f"JQv4 manifest split must be an object: {candidate}")
    strategy = split.get("strategy")
    partition = split.get("partition", payload.get("partition"))
    if strategy != "whole-game-h64-v1" or partition not in {"train", "validation"}:
        raise SystemExit(f"invalid JQv4 split metadata in {candidate}")
    seed = split.get("seed")
    numerator = split.get("numerator")
    denominator = split.get("denominator")
    # Keep the Python preflight exactly aligned with the native JQv4 ABI:
    # ``seed`` is u64, fractions are u32, and JSON booleans are not integers.
    if (not isinstance(seed, int) or isinstance(seed, bool)
            or not 0 <= seed <= JQV4_U64_MAX
            or not isinstance(numerator, int) or isinstance(numerator, bool)
            or not 0 <= numerator <= JQV4_U32_MAX
            or not isinstance(denominator, int) or isinstance(denominator, bool)
            or not 1 <= denominator <= JQV4_U32_MAX
            or numerator >= denominator):
        raise SystemExit(f"invalid JQv4 split seed/fraction in {candidate}")
    if payload.get("partition") not in {None, partition}:
        raise SystemExit(f"JQv4 manifest partition fields disagree in {candidate}")
    shuffle = payload.get("shuffle")
    if not isinstance(shuffle, dict):
        raise SystemExit(f"JQv4 manifest shuffle must be an object: {candidate}")
    if shuffle.get("strategy") != JQV4_SHUFFLE_STRATEGY:
        raise SystemExit(f"invalid JQv4 shuffle strategy in {candidate}")
    shuffle_seed = shuffle.get("seed")
    shuffle_buffer = shuffle.get("buffer_blocks")
    if (not isinstance(shuffle_seed, int) or isinstance(shuffle_seed, bool)
            or not 0 <= shuffle_seed <= JQV4_U64_MAX):
        raise SystemExit(f"invalid JQv4 shuffle seed in {candidate}")
    if (not isinstance(shuffle_buffer, int) or isinstance(shuffle_buffer, bool)
            or not 1 <= shuffle_buffer <= JQV4_U32_MAX):
        raise SystemExit(f"invalid JQv4 shuffle buffer_blocks in {candidate}")
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise SystemExit(f"JQv4 manifest files must be a non-empty array: {candidate}")
    input_root_value = payload.get("input_root")
    input_root = (
        Path(input_root_value)
        if isinstance(input_root_value, str) and input_root_value
        else candidate.parent
    )
    if not input_root.is_absolute():
        input_root = candidate.parent / input_root
    seen_paths: set[str] = set()
    seen_uuids: set[str] = set()
    shards: set[str] = set()
    ranges: list[tuple[int, int]] = []
    # The native bounded-shuffle stream consumes files in manifest order.
    # Require the generator's canonical identity order so reordering JSON
    # cannot preserve the teacher digest while changing training samples.
    file_order: list[tuple[int, int, str]] = []
    manifest_dataset_uuid = payload.get("dataset_uuid")
    if not isinstance(manifest_dataset_uuid, str):
        raise SystemExit(f"JQv4 manifest dataset_uuid must be a string: {candidate}")
    try:
        manifest_dataset_uuid = str(uuid.UUID(manifest_dataset_uuid))
    except (ValueError, AttributeError) as exc:
        raise SystemExit(f"JQv4 manifest dataset_uuid is invalid: {candidate}") from exc
    manifest_feature_set = payload.get("feature_set")
    if not isinstance(manifest_feature_set, str) or not manifest_feature_set:
        raise SystemExit(f"JQv4 manifest feature_set is missing: {candidate}")
    for index, entry in enumerate(files):
        if not isinstance(entry, dict):
            raise SystemExit(f"JQv4 manifest file #{index} must be an object: {candidate}")
        raw_path = entry.get("path") or entry.get("relative_path")
        if not isinstance(raw_path, str) or not raw_path:
            raise SystemExit(f"JQv4 manifest file #{index} has no path: {candidate}")
        source = Path(raw_path)
        if not source.is_absolute():
            source = input_root / source
        source = source.resolve()
        if source.suffix.lower() != ".jqv4":
            raise SystemExit(f"JQv4 manifest source is not .jqv4: {source}")
        if not source.is_file():
            raise SystemExit(f"JQv4 manifest source is missing: {source}")
        key = os.path.normcase(str(source))
        if key in seen_paths:
            raise SystemExit(f"JQv4 manifest contains a duplicate source: {source}")
        seen_paths.add(key)
        file_uuid = entry.get("file_uuid")
        if not isinstance(file_uuid, str) or not file_uuid:
            raise SystemExit(f"JQv4 manifest file #{index} has no file_uuid")
        try:
            uuid_key = str(uuid.UUID(file_uuid))
        except (ValueError, AttributeError) as exc:
            raise SystemExit(
                f"JQv4 manifest file_uuid is invalid: {file_uuid!r}"
            ) from exc
        if uuid_key in seen_uuids:
            raise SystemExit(f"JQv4 manifest contains duplicate file_uuid: {file_uuid}")
        seen_uuids.add(uuid_key)
        source_dataset_uuid = entry.get("dataset_uuid")
        if not isinstance(source_dataset_uuid, str) or not source_dataset_uuid:
            raise SystemExit(f"JQv4 manifest source dataset_uuid is missing: {source}")
        try:
            if str(uuid.UUID(source_dataset_uuid)) != manifest_dataset_uuid:
                raise SystemExit(
                    f"JQv4 manifest source dataset_uuid disagrees with root: {source}"
                )
        except (ValueError, AttributeError) as exc:
            raise SystemExit(
                f"JQv4 manifest source dataset_uuid is invalid: {source}"
            ) from exc
        if entry.get("feature_set") != manifest_feature_set:
            raise SystemExit(
                f"JQv4 manifest source feature_set disagrees with root: {source}"
            )
        start = entry.get("record_start")
        end = entry.get("record_end")
        if (not isinstance(start, int) or start < 0 or
                not isinstance(end, int) or end <= start):
            raise SystemExit(
                f"JQv4 manifest source record range is invalid: {source}"
            )
        ranges.append((start, end))
        file_order.append((start, end, uuid_key))
        shard = str(entry.get("shard", ""))
        if shard not in {"1", "2", "3", "4"}:
            raise SystemExit(f"JQv4 manifest file has an invalid shard: {shard!r}")
        shards.add(shard)
        try:
            validate_sha256_text(entry.get("sha256"), field=f"sha256 ({source.name})")
        except (TypeError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
    try:
        declared_teacher_hash = validate_sha256_text(
            payload["teacher_dataset_sha256"],
            field=f"teacher_dataset_sha256 ({candidate.name})")
        declared_corpus_hash = validate_sha256_text(
            payload["corpus_sha256"], field=f"corpus_sha256 ({candidate.name})")
    except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"invalid JQv4 manifest provenance in {candidate}: {exc}") from exc
    if file_order != sorted(file_order):
        raise SystemExit(
            f"JQv4 manifest files are not in canonical order: {candidate}")
    # The training manifest lists the complete corpus and can authenticate its
    # own digest.  Validation intentionally lists only shard 4, so its digest
    # is compared to the already-authenticated training digest in main().
    if partition == "train":
        try:
            computed_teacher_hash = _canonical_teacher_dataset_sha256(payload)
            computed_corpus_hash = _canonical_corpus_sha256(payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(
                f"invalid JQv4 manifest provenance fields in {candidate}: {exc}") from exc
        if declared_teacher_hash != computed_teacher_hash:
            raise SystemExit(
                f"JQv4 manifest teacher_dataset_sha256 is stale or tampered: {candidate}")
        if declared_corpus_hash != computed_corpus_hash:
            raise SystemExit(
                f"JQv4 manifest corpus_sha256 is stale or tampered: {candidate}")
    payload["teacher_dataset_sha256"] = declared_teacher_hash
    payload["corpus_sha256"] = declared_corpus_hash
    if partition == "validation" and shards != {"4"}:
        raise SystemExit(
            f"validation JQv4 manifest must contain only shard 4: {candidate}"
        )
    if partition == "train" and shards != {"1", "2", "3", "4"}:
        raise SystemExit(
            f"training JQv4 manifest must contain shards 1,2,3,4: {candidate}"
        )
    ranges.sort()
    for previous, current in zip(ranges, ranges[1:]):
        if current[0] != previous[1]:
            raise SystemExit(
                f"JQv4 manifest record ranges are not contiguous: {candidate}"
            )
    return payload


def manifest_teacher_hash(path: str | os.PathLike[str]) -> str | None:
    """Return the provenance hash for a JQv4 manifest, if ``path`` is one."""

    payload = read_jqv4_manifest(path)
    return None if payload is None else payload["teacher_dataset_sha256"]


def _manifest_source_identity(source: dict) -> tuple:
    return (
        source["shard"], source["relative_path"], source["record_start"],
        source["record_end"], source["bytes"], source["sha256"],
        source["dataset_uuid"], source["file_uuid"],
        source["metadata_sha256"], source["feature_set"],
    )


def validate_jqv4_manifest_pair(train: dict, validation: dict) -> None:
    """Require validation to be the authenticated shard-4 view of train."""
    if train.get("corpus_sha256") != validation.get("corpus_sha256"):
        raise SystemExit("train and validation manifests have different corpus_sha256")
    if train.get("split", {}).get("strategy") != validation.get("split", {}).get("strategy"):
        raise SystemExit("train and validation manifests have different split strategy")
    for field in ("seed", "numerator", "denominator"):
        if train.get("split", {}).get(field) != validation.get("split", {}).get(field):
            raise SystemExit(
                f"train and validation manifests have different split {field}")
    train_sources = {
        _manifest_source_identity(source): source
        for source in train.get("files", [])
        if source.get("shard") == "4"
    }
    validation_sources = {
        _manifest_source_identity(source): source
        for source in validation.get("files", [])
    }
    if train_sources != validation_sources:
        raise SystemExit(
            "validation manifest files are not the shard-4 subset of the train manifest")


def validate_resume_provenance(path, *, feature_name: str, expected_teacher_dataset_sha256: str,
                               expected_format: str,
                               expected_source_manifest_sha256: str | None = None,
                               expected_head_balance_cap: float | None = None,
                               expected_coverage_report_sha256: str | None = None,
                               expected_head_balance_weights=None,
                               expected_head_init_seed: int | None = None):
    """Validate a resume payload and bind it to this run's exact teacher data."""

    provenance = validate_before_torch_load(
        path, expected_feature_name=feature_name, expected_format=expected_format,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
        expected_head_balance_cap=expected_head_balance_cap,
        expected_coverage_report_sha256=expected_coverage_report_sha256,
        expected_head_balance_weights=expected_head_balance_weights,
        expected_head_init_seed=expected_head_init_seed)
    if provenance.teacher_dataset_sha256 != expected_teacher_dataset_sha256:
        raise WeightPolicyError(
            "teacher_dataset_sha256 does not match the current training dataset "
            f"(checkpoint={provenance.teacher_dataset_sha256}, "
            f"current={expected_teacher_dataset_sha256})")
    return provenance


def validate_dataset_transition_resume_provenance(
        path, *, feature_name: str, current_teacher_dataset_sha256: str,
        current_corpus_sha256: str | None,
        expected_format: str, expected_head_init_seed: int | None = None):
    """Authenticate a prior checkpoint while allowing an intentional dataset change."""

    provenance = validate_before_torch_load(
        path, expected_feature_name=feature_name, expected_format=expected_format,
        expected_head_init_seed=expected_head_init_seed)
    current_hash = validate_sha256_text(
        current_teacher_dataset_sha256, field="current_teacher_dataset_sha256")
    if provenance.corpus_sha256 is None:
        raise WeightPolicyError(
            "dataset transition requires prior checkpoint corpus_sha256")
    if current_corpus_sha256 is None:
        raise WeightPolicyError(
            "dataset transition requires current corpus_sha256")
    current_corpus_hash = validate_sha256_text(
        current_corpus_sha256, field="current_corpus_sha256")
    if provenance.teacher_dataset_sha256 == current_hash:
        raise WeightPolicyError(
            "dataset transition requires a different teacher dataset "
            f"(checkpoint={provenance.teacher_dataset_sha256}, current={current_hash})")
    if provenance.corpus_sha256 == current_corpus_hash:
        raise WeightPolicyError(
            "dataset transition requires a different corpus "
            f"(checkpoint={provenance.corpus_sha256}, current={current_corpus_hash})")
    return provenance


def _file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_dataset_transition_lineage(
        output_dir: str | os.PathLike[str], checkpoint, provenance,
        *, current_teacher_dataset_sha256: str,
        current_corpus_sha256: str | None,
        current_source_manifest_sha256: str | None,
        current_coverage_report_sha256: str | None,
        current_lambda: float, max_epochs: int | None) -> Path:
    """Write an atomic, immutable record linking a transition to its prior run."""

    checkpoint_path = Path(checkpoint).resolve()
    prior_sidecar = Path(provenance.source).resolve()
    current_teacher_hash = validate_sha256_text(
        current_teacher_dataset_sha256, field="current_teacher_dataset_sha256")
    if provenance.corpus_sha256 is None:
        raise WeightPolicyError("dataset transition lineage requires prior corpus_sha256")
    if current_corpus_sha256 is None:
        raise WeightPolicyError("dataset transition lineage requires current corpus_sha256")
    current_corpus_hash = validate_sha256_text(
        current_corpus_sha256, field="current_corpus_sha256")
    target_dir = Path(output_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    document = {
        "schema": "abjchess-v8-dataset-transition-lineage-v1",
        "prior_checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _file_sha256(checkpoint_path),
        },
        "prior_sidecar": {
            "path": str(prior_sidecar),
            "sha256": _file_sha256(prior_sidecar),
        },
        "teacher_dataset_sha256": {
            "prior": provenance.teacher_dataset_sha256,
            "current": current_teacher_hash,
        },
        "corpus_sha256": {
            "prior": provenance.corpus_sha256,
            "current": current_corpus_hash,
        },
        "current_manifest_sha256": current_source_manifest_sha256,
        "current_source_manifest_sha256": current_source_manifest_sha256,
        "current_coverage_report_sha256": current_coverage_report_sha256,
        "current_lambda": float(current_lambda),
        "inherited_epoch_intent": {"max_epochs": max_epochs},
    }
    target = target_dir / "dataset-transition-lineage.json"
    encoded = json.dumps(document, ensure_ascii=True, sort_keys=True,
                         separators=(",", ":")) + "\n"
    temporary = target.with_name(f"{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # Hard-linking a new name is an atomic no-replace publication on
            # both NTFS and POSIX filesystems.
            os.link(temporary, target)
        except FileExistsError:
            if target.read_text(encoding="utf-8") == encoded:
                return target
            raise WeightPolicyError(
                f"dataset transition lineage already exists with different content: {target}")
    finally:
        temporary.unlink(missing_ok=True)
    return target


def validate_resume_options(args) -> None:
    if not getattr(args, "allow_dataset_transition", False):
        return
    if not getattr(args, "resume_from_checkpoint", None):
        raise SystemExit("--allow-dataset-transition requires --resume_from_checkpoint")
    if getattr(args, "resume_from_model", None):
        raise SystemExit("--allow-dataset-transition cannot be used with --resume-from-model")
    if Path(args.resume_from_checkpoint).suffix.lower() != ".ckpt":
        raise SystemExit("--allow-dataset-transition requires a Lightning .ckpt checkpoint")


def make_data_loaders(train_filename, val_filename, feature_set, num_workers,
                      batch_size, filtered, random_fen_skipping, main_device,
                      epoch_size, val_size, context):
    train_batches = batches_per_rank(epoch_size, batch_size, context.world_size)
    val_batches = batches_per_rank(val_size, batch_size, context.world_size)
    train_infinite = nnue_dataset.SparseBatchDataset(
        feature_set.name, train_filename, batch_size, num_workers=num_workers,
        filtered=filtered, random_fen_skipping=random_fen_skipping,
        device=main_device, rank=context.global_rank, world_size=context.world_size)
    val_infinite = nnue_dataset.SparseBatchDataset(
        feature_set.name, val_filename, batch_size, num_workers=num_workers,
        filtered=filtered, random_fen_skipping=random_fen_skipping,
        device=main_device, rank=context.global_rank, world_size=context.world_size)
    train = DataLoader(nnue_dataset.FixedNumBatchesDataset(train_infinite, train_batches),
                       batch_size=None, num_workers=0)
    val = DataLoader(nnue_dataset.FixedNumBatchesDataset(val_infinite, val_batches),
                     batch_size=None, num_workers=0)
    return train, val, train_batches, val_batches


def build_parser():
    parser = argparse.ArgumentParser(description="Train AB-JChess V8 NNUE")
    parser.add_argument("train", help="jqv4 training data (.bin)")
    parser.add_argument("val", help="jqv4 validation data (.bin)")
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--lambda", default=1.0, type=float, dest="lambda_",
                        help="evaluation/result loss interpolation")
    parser.add_argument("--lr", default=1.5e-3, type=float)
    parser.add_argument("--num-workers", default=1, type=int)
    parser.add_argument("--batch-size", default=-1, type=int)
    parser.add_argument("--threads", default=-1, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--head-coverage-report", default=None,
                        help="authenticated head coverage report JSON")
    parser.add_argument("--head-balance-cap", default=DEFAULT_CAP, type=float)
    parser.add_argument("--teacher-dataset-sha256", "--teacher-package-sha256",
                        dest="teacher_dataset_sha256", default=None,
                        help="SHA-256 of the exact teacher dataset (read from a JQv4 manifest when omitted)")
    parser.add_argument("--smart-fen-skipping", action="store_true",
                        dest="smart_fen_skipping_deprecated")
    parser.add_argument("--no-smart-fen-skipping", action="store_true")
    parser.add_argument("--random-fen-skipping", default=3, type=int)
    parser.add_argument("--resume-from-model")
    parser.add_argument("--allow-dataset-transition", action="store_true",
                        help="allow an authenticated Lightning checkpoint to resume on a different teacher dataset")
    parser.add_argument("--epoch-size", default=20_000_000, type=int)
    parser.add_argument("--validation-size", default=1_000_000, type=int)
    parser.add_argument("--keep-last-only", action="store_true",
                        help="retain only last.ckpt and its provenance sidecar")
    features_v8.add_argparse_args(parser)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate_resume_options(args)
    try:
        args.lambda_ = validate_lambda_value(args.lambda_)
    except (TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    train_manifest = read_jqv4_manifest(args.train)
    val_manifest = read_jqv4_manifest(args.val)
    train_manifest_hash = (
        None if train_manifest is None else train_manifest["teacher_dataset_sha256"]
    )
    corpus_sha256 = (
        None if train_manifest is None else train_manifest.get("corpus_sha256")
    )
    # This is intentionally distinct from the canonical teacher-data digest:
    # the manifest bytes also bind file order, paths, and partition metadata.
    source_manifest_sha256 = (
        None if train_manifest is None
        else hashlib.sha256(Path(args.train).resolve().read_bytes()).hexdigest()
    )
    val_manifest_hash = (
        None if val_manifest is None else val_manifest["teacher_dataset_sha256"]
    )
    if train_manifest is not None or val_manifest is not None:
        if train_manifest is None or val_manifest is None:
            raise SystemExit("JQv4 training requires both train and validation manifests")
        if train_manifest_hash != val_manifest_hash:
            raise SystemExit("train and validation manifests have different teacher_dataset_sha256")
        if train_manifest.get("corpus_sha256") != val_manifest.get("corpus_sha256"):
            raise SystemExit("train and validation manifests have different corpus_sha256")
        if train_manifest.get("dataset_uuid") != val_manifest.get("dataset_uuid"):
            raise SystemExit("train and validation manifests have different dataset_uuid")
        if train_manifest.get("feature_set") != val_manifest.get("feature_set"):
            raise SystemExit("train and validation manifests have different feature_set")
        if train_manifest.get("shuffle") != val_manifest.get("shuffle"):
            raise SystemExit("train and validation manifests have different shuffle configuration")
        validate_jqv4_manifest_pair(train_manifest, val_manifest)
        if train_manifest.get("split", {}).get("partition") != "train":
            raise SystemExit("training input manifest must use partition=train")
        if val_manifest.get("split", {}).get("partition") != "validation":
            raise SystemExit("validation input manifest must use partition=validation")
        if args.features not in nnue_dataset.JIEQI_V8_FEATURE_SETS:
            raise SystemExit("JQv4 manifests require a V8 Jieqi feature set")
        if args.teacher_dataset_sha256 is None:
            args.teacher_dataset_sha256 = train_manifest_hash
        elif args.teacher_dataset_sha256.lower() != train_manifest_hash.lower():
            raise SystemExit("--teacher-dataset-sha256 does not match the JQv4 manifest")
    try:
        args.teacher_dataset_sha256 = validate_sha256_text(
            args.teacher_dataset_sha256, field="teacher_dataset_sha256")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not os.path.isfile(args.train) or not os.path.isfile(args.val):
        raise SystemExit("both train and validation files must exist")

    if args.gpus is None:
        args.gpus = 1
    try:
        gpu_count = requested_gpu_count(args.gpus, torch.cuda.device_count())
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    # The JQv4 native stream owns one decoder and one rank.  Reject an
    # accidental multi-GPU invocation before Lightning can initialize DDP;
    # otherwise each child process fails later and leaves a misleading
    # rendezvous/DLL error behind. Binary V3 inputs retain their existing
    # multi-GPU behavior.
    if train_manifest is not None and gpu_count > 1:
        raise SystemExit(
            "JQv4 V8 training requires exactly one GPU/rank; "
            "multi-GPU DDP is not supported by the native stream"
        )
    if gpu_count > 1 and args.strategy is None:
        args.strategy = "ddp"
    context = resolve_distributed_context(gpu_count)
    pl.seed_everything(args.seed)

    feature_set = features_v8.get_feature_set_from_name(args.features)
    head_weights = None
    coverage_hash = None
    if args.head_coverage_report:
        try:
            _, head_weights = load_coverage_report(
                args.head_coverage_report,
                manifest_sha256=source_manifest_sha256,
                cap=args.head_balance_cap)
            coverage_hash = coverage_report_sha256(args.head_coverage_report)
        except ValueError as exc:
            raise SystemExit(f"invalid head coverage report: {exc}") from exc
    trainer_checkpoint = getattr(args, "resume_from_checkpoint", None)
    resume_provenance = None
    dataset_transition = bool(getattr(args, "allow_dataset_transition", False))
    if trainer_checkpoint:
        try:
            # Lightning unpickles the checkpoint during ``fit``; authenticate
            # its V8 sidecar before handing it to the framework.
            if dataset_transition:
                resume_provenance = validate_dataset_transition_resume_provenance(
                    trainer_checkpoint, feature_name=feature_set.name,
                    current_teacher_dataset_sha256=args.teacher_dataset_sha256,
                    current_corpus_sha256=corpus_sha256,
                    expected_format="pytorch-lightning-v1",
                    expected_head_init_seed=args.seed)
            else:
                resume_provenance = validate_resume_provenance(
                    trainer_checkpoint, feature_name=feature_set.name,
                    expected_teacher_dataset_sha256=args.teacher_dataset_sha256,
                    expected_format="pytorch-lightning-v1",
                    expected_source_manifest_sha256=source_manifest_sha256,
                    expected_head_balance_cap=args.head_balance_cap,
                    expected_coverage_report_sha256=coverage_hash,
                    expected_head_balance_weights=head_weights,
                    expected_head_init_seed=args.seed)
        except (WeightPolicyError, ValueError, TypeError, RuntimeError) as exc:
            raise SystemExit(f"AB-JChess V8 rejected resume checkpoint: {exc}") from exc
    if args.resume_from_model:
        if Path(args.resume_from_model).suffix.lower() != ".pt":
            raise SystemExit("--resume-from-model accepts only a V8 .pt module; use --resume_from_checkpoint for .ckpt")
        try:
            resume_provenance = validate_resume_provenance(
                args.resume_from_model, feature_name=feature_set.name,
                expected_teacher_dataset_sha256=args.teacher_dataset_sha256,
                expected_format="pytorch-module-v1",
                expected_source_manifest_sha256=source_manifest_sha256,
                expected_head_balance_cap=args.head_balance_cap,
                expected_coverage_report_sha256=coverage_hash,
                expected_head_balance_weights=head_weights,
                expected_head_init_seed=args.seed)
            nnue = model_v8.load_v8_checkpoint(args.resume_from_model, feature_set)
        except (WeightPolicyError, ValueError, TypeError, RuntimeError) as exc:
            raise SystemExit(f"AB-JChess V8 rejected resume model: {exc}") from exc
        nnue.lambda_, nnue.lr = args.lambda_, args.lr
    else:
        nnue = model_v8.NNUE(feature_set=feature_set, lambda_=args.lambda_, lr=args.lr,
                              head_init_seed=args.seed)
    if head_weights is not None:
        nnue.set_head_balance(head_weights)
        object.__setattr__(nnue, "_v81_head_balance_cap", float(args.head_balance_cap))
    elif resume_provenance is not None and not dataset_transition:
        nnue.set_head_balance(resume_provenance.head_balance["weights"])
        object.__setattr__(
            nnue, "_v81_head_balance_cap",
            float(resume_provenance.head_balance["cap"]))
        object.__setattr__(
            nnue, "_v81_head_init_seed",
            int(resume_provenance.independent_init["seed"]))
    else:
        object.__setattr__(
            nnue, "_v81_head_balance_cap",
            float(args.head_balance_cap if dataset_transition else DEFAULT_CAP))

    if dataset_transition:
        write_dataset_transition_lineage(
            args.default_root_dir or "logs-v8/", trainer_checkpoint,
            resume_provenance,
            current_teacher_dataset_sha256=args.teacher_dataset_sha256,
            current_corpus_sha256=corpus_sha256,
            current_source_manifest_sha256=source_manifest_sha256,
            current_coverage_report_sha256=coverage_hash,
            current_lambda=args.lambda_, max_epochs=args.max_epochs)

    batch_size = args.batch_size if args.batch_size > 0 else 16_384
    if args.threads > 0:
        t_set_num_threads(args.threads)
    logdir = args.default_root_dir or "logs-v8/"
    logger = pl_loggers.TensorBoardLogger(logdir)
    checkpoint_callback = ABJChessV8ModelCheckpoint(
        save_last=True, every_n_epochs=1, save_top_k=-1,
        provenance_feature_set=feature_set,
        teacher_dataset_sha256=args.teacher_dataset_sha256,
        corpus_sha256=corpus_sha256,
        source_manifest_sha256=source_manifest_sha256,
        coverage_report_sha256_value=coverage_hash,
        head_balance_cap=args.head_balance_cap,
        keep_last_only=args.keep_last_only)
    sidecar_callback = ABJChessV8SidecarCallback(
        checkpoint_callback, feature_set, args.teacher_dataset_sha256,
        corpus_sha256=corpus_sha256,
        source_manifest_sha256=source_manifest_sha256,
        coverage_report_sha256_value=coverage_hash,
        head_balance_cap=args.head_balance_cap)
    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=[checkpoint_callback, sidecar_callback,
                         ABJChessV8EpochCallback()], logger=logger,
        replace_sampler_ddp=False)
    root_device = trainer.strategy.root_device
    if root_device.type == "cuda":
        device_index = root_device.index if root_device.index is not None else context.local_rank
        main_device = f"cuda:{device_index}"
    else:
        main_device = str(root_device)
    train, val, train_batches, val_batches = make_data_loaders(
        args.train, args.val, feature_set, args.num_workers, batch_size,
        not args.no_smart_fen_skipping, args.random_fen_skipping, main_device,
        args.epoch_size, args.validation_size, context)
    nnue.ranger21_num_batches_per_epoch = train_batches
    nnue.ranger21_num_epochs = args.max_epochs if args.max_epochs and args.max_epochs > 0 else 1

    # A JQv4 manifest is not a V3 fixed-record stream.  Keep the diagnostic
    # range inspection only for binary .bin inputs (V3 or V8-on-V3).
    if feature_set.name in nnue_dataset.JIEQI_FEATURE_SETS and not nnue_dataset.is_jqv4_manifest(args.train):
        train_shard = inspect_v3_shard(args.train, context.global_rank, context.world_size)
        val_shard = inspect_v3_shard(args.val, context.global_rank, context.world_size)
        print(f"V8 train shard {train_shard.start_record}:{train_shard.end_record} "
              f"val {val_shard.start_record}:{val_shard.end_record}")
    trainer.fit(nnue, train, val)


if __name__ == "__main__":
    main()
