from dataclasses import dataclass
from pathlib import Path
import os
import struct


V3_MAGIC = 0x514A4B50
V3_VERSION = 3
V3_RECORD_SIZE = 270
V3_FEN_BYTES = 256
V3_HEADER = struct.Struct("<4I")


@dataclass(frozen=True)
class DistributedContext:
    global_rank: int
    local_rank: int
    world_size: int


@dataclass(frozen=True)
class V3ShardInfo:
    path: Path
    total_records: int
    start_record: int
    end_record: int

    @property
    def shard_records(self) -> int:
        return self.end_record - self.start_record


def compute_shard_range(
    total_records: int, rank: int, world_size: int
) -> tuple[int, int]:
    if total_records <= 0:
        raise ValueError("v3 file has no records")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank {rank} is outside [0, {world_size})")
    if world_size > total_records:
        raise ValueError(
            f"world_size {world_size} has fewer records ({total_records}) than ranks"
        )
    start = total_records * rank // world_size
    end = total_records * (rank + 1) // world_size
    if start >= end:
        raise ValueError(
            f"rank {rank}/{world_size} received an empty v3 record range"
        )
    return start, end


def inspect_v3_shard(path, rank: int, world_size: int) -> V3ShardInfo:
    path = Path(path).resolve()
    size = path.stat().st_size
    if size < V3_HEADER.size:
        raise ValueError(f"{path}: file is smaller than the 16-byte v3 header")
    with path.open("rb") as handle:
        magic, version, record_size, fen_bytes = V3_HEADER.unpack(
            handle.read(V3_HEADER.size)
        )
    if magic != V3_MAGIC:
        raise ValueError(f"{path}: invalid v3 magic 0x{magic:08X}")
    if version != V3_VERSION:
        raise ValueError(f"{path}: unsupported v3 version {version}")
    if record_size != V3_RECORD_SIZE or fen_bytes != V3_FEN_BYTES:
        raise ValueError(
            f"{path}: expected record_size={V3_RECORD_SIZE}, "
            f"fen_bytes={V3_FEN_BYTES}; got {record_size}, {fen_bytes}"
        )
    data_bytes = size - V3_HEADER.size
    if data_bytes % record_size:
        raise ValueError(
            f"{path}: payload is not an integral number of "
            f"{record_size}-byte records"
        )
    total_records = data_bytes // record_size
    start, end = compute_shard_range(total_records, rank, world_size)
    return V3ShardInfo(path, total_records, start, end)


def _gpu_device_count(device_ids, available_device_count: int) -> int:
    if not device_ids:
        raise ValueError("GPU device selector must not be empty")
    seen = set()
    for device_id in device_ids:
        if type(device_id) is not int:
            raise ValueError(
                f"GPU device IDs must be integers, got {device_id!r}"
            )
        if device_id in seen:
            raise ValueError(f"GPU device ID {device_id} is duplicated")
        if device_id < 0 or device_id >= available_device_count:
            raise ValueError(
                f"GPU device ID {device_id} is outside [0, "
                f"{available_device_count})"
            )
        seen.add(device_id)
    return len(device_ids)


def requested_gpu_count(gpus, available_device_count: int) -> int:
    if gpus is None:
        count = 1
    elif isinstance(gpus, (list, tuple)):
        count = _gpu_device_count(gpus, available_device_count)
    elif isinstance(gpus, int):
        count = available_device_count if gpus == -1 else gpus
    else:
        value = str(gpus).strip()
        if value == "-1":
            count = available_device_count
        elif "," in value:
            items = value.split(",")
            if any(not item.strip() for item in items):
                raise ValueError(
                    f"GPU device selector {value!r} contains an empty item"
                )
            try:
                device_ids = [int(item.strip()) for item in items]
            except ValueError as error:
                raise ValueError(
                    f"GPU device selector {value!r} must contain integer IDs"
                ) from error
            count = _gpu_device_count(device_ids, available_device_count)
        else:
            count = int(value)
    if count <= 0:
        raise ValueError(f"requested GPU count must be positive, got {count}")
    if count > available_device_count:
        raise ValueError(
            f"requested {count} GPUs, but torch sees {available_device_count}"
        )
    return count


def _environment_int(environ, name: str, default: int) -> int:
    value = environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer, got {value!r}") from error


def resolve_distributed_context(
    requested_world_size: int, environ=None
) -> DistributedContext:
    environ = os.environ if environ is None else environ
    env_world = environ.get("WORLD_SIZE")
    world_size = _environment_int(environ, "WORLD_SIZE", requested_world_size)
    if env_world is not None and world_size != requested_world_size:
        raise ValueError(
            f"WORLD_SIZE={world_size} conflicts with requested GPU count "
            f"{requested_world_size}"
        )
    local_rank = _environment_int(environ, "LOCAL_RANK", 0)
    global_rank = _environment_int(environ, "RANK", local_rank)
    if world_size <= 0:
        raise ValueError("WORLD_SIZE must be positive")
    if not 0 <= global_rank < world_size:
        raise ValueError(f"RANK={global_rank} is outside WORLD_SIZE={world_size}")
    if not 0 <= local_rank < world_size:
        raise ValueError(
            f"LOCAL_RANK={local_rank} is outside single-node "
            f"WORLD_SIZE={world_size}"
        )
    if global_rank != local_rank:
        raise ValueError(
            "multi-node rank layout is not supported; "
            "global rank must equal local rank"
        )
    return DistributedContext(global_rank, local_rank, world_size)


def batches_per_rank(
    global_positions: int, batch_size: int, world_size: int
) -> int:
    if global_positions <= 0 or batch_size <= 0 or world_size <= 0:
        raise ValueError("positions, batch_size, and world_size must be positive")
    denominator = batch_size * world_size
    return (global_positions + denominator - 1) // denominator
