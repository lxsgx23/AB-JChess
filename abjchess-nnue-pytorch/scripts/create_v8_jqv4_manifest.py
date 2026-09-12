"""Validate JQv4 shards and emit deterministic V8 train/validation manifests.

The manifest is deliberately small: it contains source identities and paths,
not decoded records.  The native loader reopens these files and keeps one
decoded block in memory at a time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


MAGIC = b"JQTDV4\r\n"
HEADER_BYTES = 128
FOOTER_BYTES = 96
SCHEMA = "abjchess-v8-jqv4-manifest-v1"
SHUFFLE_STRATEGY = "hierarchical-block-window-v1"
DEFAULT_FEATURE_SET = "HalfKAv2_hm_jieqi_v8^"
U64_MAX = (1 << 64) - 1
U32_MAX = (1 << 32) - 1
# The V8 JQv4 training tree is a fixed teacher corpus.  Treat a missing tail
# shard/file as corruption rather than silently authenticating a shorter
# contiguous prefix.
EXPECTED_FILE_COUNT = 224
EXPECTED_RECORD_COUNT = 1_000_000_000
SHARD_GROUPS = {
    "1-4": ("1", "2", "3", "4"),
    "5-8": ("5", "6", "7", "8"),
}
FILENAME_RE = re.compile(
    r"^train-(?P<begin>\d{12})-(?P<end>\d{12})-"
    r"(?P<uuid>[0-9a-fA-F-]{36})\.jqv4$"
)


class ManifestError(ValueError):
    """Raised when a source tree cannot be trusted as a JQv4 dataset."""


@dataclass(frozen=True)
class Source:
    shard: str
    path: str
    relative_path: str
    record_start: int
    record_end: int
    bytes: int
    sha256: str
    dataset_uuid: str
    file_uuid: str
    metadata_sha256: str
    feature_set: str


def _u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def _u64(data: bytes, offset: int) -> int:
    return struct.unpack_from("<Q", data, offset)[0]


def _uuid(data: bytes) -> str:
    if len(data) != 16:
        raise ManifestError("UUID field must contain 16 bytes")
    return str(uuid.UUID(bytes=data))


def _read_source(path: Path, root: Path, shard: str, *, hash_file: bool = True) -> Source:
    match = FILENAME_RE.match(path.name)
    if match is None:
        raise ManifestError(f"invalid JQv4 filename: {path.name}")
    begin = int(match.group("begin"))
    end = int(match.group("end"))
    if begin >= end:
        raise ManifestError(f"empty record range in {path.name}")

    size = path.stat().st_size
    if size < HEADER_BYTES + FOOTER_BYTES:
        raise ManifestError(f"JQv4 file is too small: {path}")
    with path.open("rb") as stream:
        header = stream.read(HEADER_BYTES)
        metadata_bytes = _u32(header, 44)
        stream.seek(HEADER_BYTES)
        metadata = stream.read(metadata_bytes)
        stream.seek(size - FOOTER_BYTES)
        footer = stream.read(FOOTER_BYTES)
    if header[:8] != MAGIC:
        raise ManifestError(f"bad JQv4 magic: {path}")
    if struct.unpack_from("<HHH", header, 8) != (4, 0, HEADER_BYTES):
        raise ManifestError(f"unsupported JQv4 header: {path}")
    if _u32(header, 40) != 40 or metadata_bytes > 16 * 1024 * 1024:
        raise ManifestError(f"invalid JQv4 metadata header: {path}")
    first_block = _u64(header, 48)
    if first_block < HEADER_BYTES + metadata_bytes or first_block >= size:
        raise ManifestError(f"invalid first block offset: {path}")
    if footer[:8] != b"JQEND4\r\n":
        raise ManifestError(f"bad JQv4 footer magic: {path}")
    footer_size, footer_version = struct.unpack_from("<HH", footer, 8)
    if footer_size != FOOTER_BYTES or footer_version != 1:
        raise ManifestError(f"unsupported JQv4 footer: {path}")
    footer_records = _u64(footer, 56)
    committed_bytes = _u64(footer, 72)
    if footer_records != end - begin or committed_bytes != size:
        raise ManifestError(
            f"record/footer range mismatch in {path.name}: "
            f"filename={end - begin}, footer={footer_records}, bytes={size}/{committed_bytes}"
        )
    try:
        metadata_obj = json.loads(metadata.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"invalid JQv4 metadata JSON: {path}") from exc
    dataset_uuid = str(metadata_obj.get("dataset_uuid", ""))
    file_uuid = str(metadata_obj.get("file_uuid", ""))
    try:
        dataset_uuid = str(uuid.UUID(dataset_uuid))
        file_uuid = str(uuid.UUID(file_uuid))
    except (ValueError, AttributeError) as exc:
        raise ManifestError(f"metadata UUID is invalid: {path}") from exc
    header_dataset = _uuid(header[56:72])
    header_file = _uuid(header[72:88])
    if dataset_uuid != header_dataset or file_uuid != header_file:
        raise ManifestError(f"header/metadata UUID mismatch: {path}")
    if _uuid(footer[16:32]) != file_uuid:
        raise ManifestError(f"header/footer file UUID mismatch: {path}")
    feature_set = str(metadata_obj.get("feature_set", {}).get("name", ""))
    if not feature_set:
        raise ManifestError(f"metadata feature_set.name is missing: {path}")
    return Source(
        shard=shard,
        path=str(path.resolve()),
        relative_path=path.resolve().relative_to(root.resolve()).as_posix(),
        record_start=begin,
        record_end=end,
        bytes=size,
        # Header/metadata probing should stay cheap.  Manifest generation
        # keeps the default full hash, while the standalone probe can opt out
        # and reserve hashing for an explicit verification pass.
        sha256=hashlib.sha256(path.read_bytes()).hexdigest() if hash_file else "",
        dataset_uuid=dataset_uuid,
        file_uuid=file_uuid,
        metadata_sha256=hashlib.sha256(metadata).hexdigest(),
        feature_set=feature_set,
    )


def enumerate_jqv4_sources(
    root: Path | str, *, hash_files: bool = True,
    allow_extra_shards: bool = False, shard_group: str = "1-4",
) -> list[Source]:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ManifestError(f"input root does not exist: {root}")
    if shard_group not in SHARD_GROUPS:
        raise ManifestError(
            "shard_group must be one of: " + ", ".join(sorted(SHARD_GROUPS)))
    directories = {item.name for item in root.iterdir() if item.is_dir()}
    physical_shards = SHARD_GROUPS[shard_group]
    expected = set(physical_shards)
    if allow_extra_shards:
        missing = sorted(expected - directories)
        if missing:
            raise ManifestError(
                "input root is missing required shard directory: "
                + ", ".join(missing))
        selected_directories = expected
    else:
        selected_directories = expected
    if directories != expected and not allow_extra_shards:
        unexpected = sorted(directories - expected)
        if unexpected:
            raise ManifestError(f"unexpected shard directory: {', '.join(unexpected)}")
        raise ManifestError(
            "input root must contain exactly directories "
            + ", ".join(physical_shards))
    sources: list[Source] = []
    for logical_index, physical_shard in enumerate(sorted(selected_directories, key=int), 1):
        shard_root = root / physical_shard
        logical_shard = str(logical_index)
        children = sorted(shard_root.iterdir(), key=lambda item: item.name)
        if any(not item.is_file() for item in children):
            raise ManifestError(f"shard {physical_shard} contains a non-file entry")
        if any(item.suffix.lower() != ".jqv4" for item in children):
            raise ManifestError(f"shard {physical_shard} contains a non-.jqv4 file")
        if not children:
            raise ManifestError(f"shard {physical_shard} is empty")
        sources.extend(
            _read_source(item, root, logical_shard, hash_file=hash_files)
            for item in children
        )
    sources.sort(key=lambda source: (source.record_start, source.record_end, source.file_uuid))
    previous_end = 0
    file_uuids: set[str] = set()
    dataset_uuids: set[str] = set()
    ranges: set[tuple[int, int]] = set()
    for source in sources:
        if source.file_uuid in file_uuids:
            raise ManifestError(f"duplicate file UUID: {source.file_uuid}")
        if (source.record_start, source.record_end) in ranges:
            raise ManifestError(f"duplicate record range: {source.record_start}:{source.record_end}")
        if source.record_start != previous_end:
            raise ManifestError(
                f"record ranges are not contiguous at {source.relative_path}: "
                f"expected {previous_end}, got {source.record_start}"
            )
        previous_end = source.record_end
        file_uuids.add(source.file_uuid)
        dataset_uuids.add(source.dataset_uuid)
        ranges.add((source.record_start, source.record_end))
    if len(dataset_uuids) != 1:
        raise ManifestError("all JQv4 files must use one dataset UUID")
    feature_sets = {source.feature_set for source in sources}
    if len(feature_sets) != 1:
        raise ManifestError("all JQv4 files must use one feature set")
    return sources


def validate_expected_inventory(sources: list[Source]) -> None:
    """Require the complete production JQv4 corpus, including its tail."""

    if len(sources) != EXPECTED_FILE_COUNT:
        raise ManifestError(
            f"JQv4 corpus must contain exactly {EXPECTED_FILE_COUNT} .jqv4 files; "
            f"found {len(sources)}")
    if not sources or sources[-1].record_end != EXPECTED_RECORD_COUNT:
        actual_end = 0 if not sources else sources[-1].record_end
        raise ManifestError(
            "JQv4 corpus record range must end at "
            f"{EXPECTED_RECORD_COUNT:,}; found {actual_end:,}")


def _canonical_sources(sources: Iterable[Source]) -> list[dict]:
    return [
        {
            "shard": source.shard,
            "relative_path": source.relative_path,
            "record_start": source.record_start,
            "record_end": source.record_end,
            "bytes": source.bytes,
            "sha256": source.sha256,
            "dataset_uuid": source.dataset_uuid,
            "file_uuid": source.file_uuid,
            "metadata_sha256": source.metadata_sha256,
            "feature_set": source.feature_set,
        }
        for source in sources
    ]


def _shuffle_config(shuffle_seed: int, shuffle_buffer_blocks: int) -> dict:
    return {
        "strategy": SHUFFLE_STRATEGY,
        "seed": shuffle_seed,
        "buffer_blocks": shuffle_buffer_blocks,
    }


def _teacher_dataset_sha256(sources: list[Source], seed: int, numerator: int,
                            denominator: int, shuffle_seed: int,
                            shuffle_buffer_blocks: int) -> str:
    payload = {
        "sources": _canonical_sources(sources),
        "split": {
            "strategy": "whole-game-h64-v1",
            "seed": seed,
            "numerator": numerator,
            "denominator": denominator,
        },
        "shuffle": _shuffle_config(shuffle_seed, shuffle_buffer_blocks),
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _corpus_sha256(sources: list[Source]) -> str:
    """Hash physical source identities without split or shuffle configuration."""

    encoded = json.dumps(
        {"sources": _canonical_sources(sources)}, ensure_ascii=True,
        sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _manifest(sources: list[Source], output_root: Path, partition: str,
              seed: int, numerator: int, denominator: int, shuffle_seed: int,
              shuffle_buffer_blocks: int) -> dict:
    # Validation is physically read from shard 4.  The game hash still makes
    # the logical split whole-game and causes train to exclude the same IDs.
    selected = [source for source in sources if partition == "train" or source.shard == "4"]
    return {
        "schema": SCHEMA,
        "input_kind": "jqv4",
        "input_root": str(Path(sources[0].path).parents[1].resolve()),
        "partition": partition,
        "dataset_uuid": sources[0].dataset_uuid,
        "feature_set": sources[0].feature_set,
        # Keep this partition-local.  The provenance hash below intentionally
        # covers the complete teacher corpus, but consumers use this field
        # for sizing and progress estimates of the files listed here.
        "total_records": sum(source.record_end - source.record_start for source in selected),
        "teacher_dataset_sha256": _teacher_dataset_sha256(
            sources, seed, numerator, denominator,
            shuffle_seed, shuffle_buffer_blocks),
        "corpus_sha256": _corpus_sha256(sources),
        "split": {
            "strategy": "whole-game-h64-v1",
            "seed": seed,
            "numerator": numerator,
            "denominator": denominator,
            "partition": partition,
        },
        "shuffle": _shuffle_config(shuffle_seed, shuffle_buffer_blocks),
        "metadata_validation": "deferred-to-native-loader",
        "files": [asdict(source) for source in selected],
    }


def _validate_manifest_parameters(seed: int, numerator: int, denominator: int,
                                  shuffle_seed: int,
                                  shuffle_buffer_blocks: int) -> None:
    if (not isinstance(seed, int) or isinstance(seed, bool)
            or not 0 <= seed <= U64_MAX):
        raise ManifestError("seed must be an unsigned 64-bit integer")
    if (not isinstance(numerator, int) or isinstance(numerator, bool)
            or not 0 <= numerator <= U32_MAX):
        raise ManifestError("numerator must be an unsigned 32-bit integer")
    if (not isinstance(denominator, int) or isinstance(denominator, bool)
            or not 1 <= denominator <= U32_MAX):
        raise ManifestError("denominator must be a nonzero unsigned 32-bit integer")
    if numerator >= denominator:
        raise ManifestError("split seed/fraction is invalid")
    if (not isinstance(shuffle_seed, int) or isinstance(shuffle_seed, bool)
            or not 0 <= shuffle_seed <= U64_MAX):
        raise ManifestError("shuffle_seed must be an unsigned 64-bit integer")
    if (not isinstance(shuffle_buffer_blocks, int) or isinstance(shuffle_buffer_blocks, bool)
            or not 1 <= shuffle_buffer_blocks <= U32_MAX):
        raise ManifestError("shuffle_buffer_blocks must be a nonzero unsigned 32-bit integer")


def _write_manifests(sources: list[Source], output_root: Path | str, *,
                     seed: int = 20260825, numerator: int = 1,
                     denominator: int = 20, shuffle_seed: int = 42,
                     shuffle_buffer_blocks: int = 4096,
                     feature_set: str | None = None) -> tuple[Path, Path]:
    """Serialize manifests from already-probed sources.

    This deliberately does not enforce the production inventory.  It is an
    internal serialization seam used by small unit fixtures; the public
    ``build_manifests`` and CLI call ``validate_expected_inventory`` before
    reaching this writer.
    """

    _validate_manifest_parameters(
        seed, numerator, denominator, shuffle_seed, shuffle_buffer_blocks)
    if feature_set is not None:
        if not isinstance(feature_set, str) or not feature_set:
            raise ManifestError("feature_set must be a non-empty string")
        sources = [
            Source(**{**asdict(source), "feature_set": feature_set})
            for source in sources
        ]
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    train_path = output_root / "train.jqv8.json"
    val_path = output_root / "validation.jqv8.json"
    for path, partition in ((train_path, "train"), (val_path, "validation")):
        payload = _manifest(
            sources, output_root, partition, seed, numerator, denominator,
            shuffle_seed, shuffle_buffer_blocks)
        text = json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n"
        path.write_text(text, encoding="utf-8", newline="\n")
    return train_path, val_path


def build_manifests(root: Path | str, output_root: Path | str, *, seed: int = 20260825,
                    numerator: int = 1, denominator: int = 20,
                    shuffle_seed: int = 42,
                    shuffle_buffer_blocks: int = 4096,
                    feature_set: str = DEFAULT_FEATURE_SET,
                    allow_extra_shards: bool = False,
                    shard_group: str = "1-4") -> tuple[Path, Path]:
    """Probe and serialize the complete production JQv4 corpus."""

    _validate_manifest_parameters(
        seed, numerator, denominator, shuffle_seed, shuffle_buffer_blocks)
    root = Path(root).expanduser().resolve()
    sources = enumerate_jqv4_sources(
        root, allow_extra_shards=allow_extra_shards,
        shard_group=shard_group)
    validate_expected_inventory(sources)
    return _write_manifests(
        sources, output_root, seed=seed, numerator=numerator,
        denominator=denominator, shuffle_seed=shuffle_seed,
        shuffle_buffer_blocks=shuffle_buffer_blocks, feature_set=feature_set)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", default=20260825, type=int)
    parser.add_argument("--numerator", default=1, type=int)
    parser.add_argument("--denominator", default=20, type=int)
    parser.add_argument("--shuffle-seed", default=42, type=int)
    parser.add_argument("--shuffle-buffer-blocks", default=4096, type=int)
    parser.add_argument("--feature-set", default=DEFAULT_FEATURE_SET,
                        help="logical production feature identity for the manifest")
    parser.add_argument(
        "--shard-group", default="1-4", choices=sorted(SHARD_GROUPS),
        help="physical shard directory group to read; emitted shard IDs stay logical 1..4",
    )
    parser.add_argument(
        "--allow-extra-shards", action="store_true",
        help="explicitly select shards 1..4 when the input root also contains other shard directories",
    )
    args = parser.parse_args(argv)
    try:
        train, validation = build_manifests(
            args.input_root, args.output_root, seed=args.seed,
            numerator=args.numerator, denominator=args.denominator,
            shuffle_seed=args.shuffle_seed,
            shuffle_buffer_blocks=args.shuffle_buffer_blocks,
            feature_set=args.feature_set,
            allow_extra_shards=args.allow_extra_shards,
            shard_group=args.shard_group)
    except ManifestError as exc:
        parser.error(str(exc))
    print(f"train_manifest={train}")
    print(f"validation_manifest={validation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
