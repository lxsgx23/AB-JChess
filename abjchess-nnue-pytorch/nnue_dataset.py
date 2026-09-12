import numpy as np
import ctypes
import torch
import os
import sys
import glob
import operator
import json
from dataclasses import dataclass
from typing import Optional

from ddp_utils import inspect_v3_shard


JIEQI_V3_FEATURE_SETS = {
    "HalfKAv2_hm_jieqi_v3_fullthreats",
    "HalfKAv2_hm_jieqi_v3_fullthreats^",
}

# V8 uses the jqv4 binary observation layout with its own loader dispatch and
# model ABI. Keep this set separate from the V3 feature names so a V8 training
# invocation cannot silently select the wrong feature encoder.
JIEQI_V8_FEATURE_SETS = {
    "HalfKAv2_hm_jieqi_v8",
    "HalfKAv2_hm_jieqi_v8^",
}
JIEQI_FEATURE_SETS = JIEQI_V3_FEATURE_SETS | JIEQI_V8_FEATURE_SETS

JQV4_MANIFEST_SCHEMA = "abjchess-v8-jqv4-manifest-v1"

C_INT_MIN = -(1 << 31)
C_INT_MAX = (1 << 31) - 1


def _checked_c_int(name, value):
    try:
        value = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer, got {value!r}") from error
    if value < C_INT_MIN or value > C_INT_MAX:
        raise ValueError(
            f"{name}={value} is outside the signed 32-bit C int range"
        )
    return int(value)


def _checked_sparse_stream_arguments(
    num_workers, batch_size, random_fen_skipping, rank, world_size
):
    num_workers = _checked_c_int("num_workers", num_workers)
    batch_size = _checked_c_int("batch_size", batch_size)
    random_fen_skipping = _checked_c_int(
        "random_fen_skipping", random_fen_skipping
    )
    rank = _checked_c_int("rank", rank)
    world_size = _checked_c_int("world_size", world_size)
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if random_fen_skipping < 0:
        raise ValueError("random_fen_skipping must be non-negative")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank {rank} is outside [0, {world_size})")
    return num_workers, batch_size, random_fen_skipping, rank, world_size


def find_training_data_loader(directory=None, platform=sys.platform):
    """Find the native stream library used by this training process.

    ``V8_TRAINING_DATA_LOADER`` may point at an explicit DLL. A runner can use
    this to load a freshly-built V8 library without copying it over another
    library in the source tree. Directory scanning is retained as the fallback
    for existing callers.
    """
    if platform.startswith("win"):
        suffix = ".dll"
    elif platform == "darwin":
        suffix = ".dylib"
    else:
        suffix = ".so"
    if directory is None:
        directory = os.environ.get("V8_TRAINING_DATA_LOADER")
    if directory is not None:
        explicit = os.path.abspath(os.fspath(directory))
        if os.path.isfile(explicit):
            if not explicit.lower().endswith(suffix):
                raise RuntimeError(
                    f"training data loader override must end in {suffix}: {explicit}"
                )
            return explicit
        directory = explicit
    else:
        directory = os.getcwd()
    directory = os.path.abspath(os.fspath(directory))
    pattern = os.path.join(directory, f"*training_data_loader{suffix}")
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"cannot find a {suffix} training data loader in {directory}"
        )
    if len(candidates) != 1:
        raise RuntimeError(
            f"multiple {suffix} training data loaders found in {directory}: "
            + ", ".join(os.path.abspath(path) for path in candidates)
        )
    return os.path.abspath(candidates[0])


try:
    dllpath = find_training_data_loader()
except FileNotFoundError as error:
    print(error)
    sys.exit(1)

if sys.platform == 'win32' and hasattr(os, 'add_dll_directory'):
    # The packaged V8 loader is statically linked to the MinGW C++ and
    # pthread runtimes.  Do not preload same-named DLLs from an MSYS2 install:
    # Windows resolves those by basename, so another copy can satisfy the
    # import and fail with an entry-point error in an otherwise valid process.
    os.add_dll_directory(os.path.dirname(dllpath))

try:
    dll = ctypes.cdll.LoadLibrary(dllpath)
except OSError as error:
    raise RuntimeError(
        "cannot load the V8 training_data_loader DLL at "
        f"{dllpath}. Rebuild it with -DV8_STATIC_MINGW_RUNTIME=ON "
        "or replace it with the packaged V8 DLL; the loader must not depend "
        "on libstdc++-6.dll, libgcc_s_seh-1.dll, or libwinpthread-1.dll."
    ) from error

class SparseBatch(ctypes.Structure):
    _fields_ = [
        ('num_inputs', ctypes.c_int),
        ('size', ctypes.c_int),
        ('is_white', ctypes.POINTER(ctypes.c_float)),
        ('outcome', ctypes.POINTER(ctypes.c_float)),
        ('score', ctypes.POINTER(ctypes.c_float)),
        ('num_active_white_features', ctypes.c_int),
        ('num_active_black_features', ctypes.c_int),
        ('max_active_features', ctypes.c_int),
        ('white', ctypes.POINTER(ctypes.c_int)),
        ('black', ctypes.POINTER(ctypes.c_int)),
        ('white_values', ctypes.POINTER(ctypes.c_float)),
        ('black_values', ctypes.POINTER(ctypes.c_float)),
        ('psqt_indices', ctypes.POINTER(ctypes.c_int)),
        ('layer_stack_indices', ctypes.POINTER(ctypes.c_int)),
        # Optional tail field added for the JQv4 V8 stream. The C++ ABI leaves
        # this null when the field is unavailable, so callers receive ``None``.
        ('eval_weight', ctypes.POINTER(ctypes.c_float)),
        # Append-only layer-stack blend tail; consumers that do not request it
        # ignore this field.
        ('layer_stack_blend', ctypes.POINTER(ctypes.c_float)),
    ]

    def get_tensors(self, device, include_eval_weight=False, include_layer_stack_blend=False):
        white_values = torch.from_numpy(np.ctypeslib.as_array(self.white_values, shape=(self.size, self.max_active_features))).pin_memory().to(device=device, non_blocking=True)
        black_values = torch.from_numpy(np.ctypeslib.as_array(self.black_values, shape=(self.size, self.max_active_features))).pin_memory().to(device=device, non_blocking=True)
        white_indices = torch.from_numpy(np.ctypeslib.as_array(self.white, shape=(self.size, self.max_active_features))).pin_memory().to(device=device, non_blocking=True)
        black_indices = torch.from_numpy(np.ctypeslib.as_array(self.black, shape=(self.size, self.max_active_features))).pin_memory().to(device=device, non_blocking=True)
        us = torch.from_numpy(np.ctypeslib.as_array(self.is_white, shape=(self.size, 1))).pin_memory().to(device=device, non_blocking=True)
        them = 1.0 - us
        outcome = torch.from_numpy(np.ctypeslib.as_array(self.outcome, shape=(self.size, 1))).pin_memory().to(device=device, non_blocking=True)
        score = torch.from_numpy(np.ctypeslib.as_array(self.score, shape=(self.size, 1))).pin_memory().to(device=device, non_blocking=True)
        psqt_indices = torch.from_numpy(np.ctypeslib.as_array(self.psqt_indices, shape=(self.size,))).long().pin_memory().to(device=device, non_blocking=True)
        layer_stack_indices = torch.from_numpy(np.ctypeslib.as_array(self.layer_stack_indices, shape=(self.size,))).long().pin_memory().to(device=device, non_blocking=True)
        eval_weight = None
        if bool(self.eval_weight):
            eval_weight = torch.from_numpy(np.ctypeslib.as_array(self.eval_weight, shape=(self.size, 1))).pin_memory().to(device=device, non_blocking=True)
        layer_stack_blend = None
        if include_layer_stack_blend and bool(self.layer_stack_blend):
            layer_stack_blend = torch.from_numpy(np.ctypeslib.as_array(self.layer_stack_blend, shape=(self.size, 1))).pin_memory().to(device=device, non_blocking=True)
        base = (us, them, white_indices, white_values, black_indices,
                black_values, outcome, score, psqt_indices,
                layer_stack_indices)
        # Preserve the Python ABI. Only the manifest-backed JQv4 V8 stream opts
        # into the optional tail field.
        if include_layer_stack_blend:
            return base + (eval_weight, layer_stack_blend)
        return base + (eval_weight,) if include_eval_weight else base

SparseBatchPtr = ctypes.POINTER(SparseBatch)


def is_jqv4_manifest(filename) -> bool:
    """Return whether ``filename`` names a V8 JQv4 manifest.

    A conventional ``*.jqv8.json``/``*.jqv4.json`` path is treated as a
    manifest even when malformed, so the native loader can report the actual
    JSON error instead of a misleading V3 header error.  Other JSON paths are
    identified by their manifest schema.
    """

    path = os.fspath(filename)
    lowered = path.lower()
    if not lowered.endswith(".json"):
        return False
    conventional = lowered.endswith(".jqv8.json") or lowered.endswith(
        ".jqv4.json"
    )
    try:
        with open(path, "r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return conventional
    return isinstance(payload, dict) and (
        payload.get("input_kind") == "jqv4"
        or payload.get("schema") == JQV4_MANIFEST_SCHEMA
    )


@dataclass(frozen=True)
class SparseStreamBindings:
    create_legacy: object
    create_v2: Optional[object]
    destroy: object
    fetch_next: object
    destroy_batch: object
    last_error: Optional[object]

    def select_creator(self, rank, world_size):
        if self.create_v2 is not None:
            return self.create_v2
        if rank == 0 and world_size == 1:
            return self.create_legacy
        raise RuntimeError(
            "multi-GPU v3 loading requires create_sparse_batch_stream_v2; "
            "rebuild training_data_loader from the current source"
        )

    def get_last_error(self):
        if self.last_error is None:
            return ""
        value = self.last_error()
        return value.decode("utf-8", errors="replace") if value else ""


def bind_sparse_stream_functions(loader):
    legacy = loader.create_sparse_batch_stream
    legacy.restype = ctypes.c_void_p
    legacy.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_int,
    ]
    create_v2 = getattr(loader, "create_sparse_batch_stream_v2", None)
    if create_v2 is not None:
        create_v2.restype = ctypes.c_void_p
        create_v2.argtypes = legacy.argtypes + [ctypes.c_int, ctypes.c_int]
    destroy = loader.destroy_sparse_batch_stream
    destroy.restype = None
    destroy.argtypes = [ctypes.c_void_p]
    fetch_next = loader.fetch_next_sparse_batch
    fetch_next.restype = SparseBatchPtr
    fetch_next.argtypes = [ctypes.c_void_p]
    destroy_batch = loader.destroy_sparse_batch
    destroy_batch.restype = None
    destroy_batch.argtypes = [SparseBatchPtr]
    last_error = getattr(loader, "get_training_data_loader_last_error", None)
    if last_error is not None:
        last_error.restype = ctypes.c_char_p
        last_error.argtypes = []
    return SparseStreamBindings(
        legacy, create_v2, destroy, fetch_next, destroy_batch, last_error
    )


SPARSE_STREAM_BINDINGS = bind_sparse_stream_functions(dll)

# Public aliases for callers. All ctypes metadata is configured by
# bind_sparse_stream_functions above.
create_sparse_batch_stream = SPARSE_STREAM_BINDINGS.create_legacy
destroy_sparse_batch_stream = SPARSE_STREAM_BINDINGS.destroy
fetch_next_sparse_batch = SPARSE_STREAM_BINDINGS.fetch_next
destroy_sparse_batch = SPARSE_STREAM_BINDINGS.destroy_batch


def _reject_unsupported_multi_rank(feature_set, rank, world_size):
    if (rank != 0 or world_size != 1) and feature_set not in JIEQI_FEATURE_SETS:
        raise RuntimeError(
            f"multi-rank sparse loading is only supported for Jieqi v3/V8 "
            f"feature sets, got {feature_set!r} at rank {rank}/{world_size}"
        )


class TrainingDataProvider:
    def __init__(
        self,
        feature_set,
        bindings,
        filename,
        cyclic,
        num_workers,
        batch_size=None,
        filtered=False,
        random_fen_skipping=0,
        device='cpu',
        rank=0,
        world_size=1):

        (
            num_workers,
            batch_size,
            random_fen_skipping,
            rank,
            world_size,
        ) = _checked_sparse_stream_arguments(
            num_workers,
            batch_size,
            random_fen_skipping,
            rank,
            world_size,
        )

        self.feature_set = feature_set.encode('utf-8')
        self.bindings = bindings
        self.destroy_stream = bindings.destroy
        self.fetch_next = bindings.fetch_next
        self.destroy_part = bindings.destroy_batch
        self.filename = filename
        self.encoded_filename = os.fsencode(filename)
        self.cyclic = cyclic
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.filtered = filtered
        self.random_fen_skipping = random_fen_skipping
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.include_eval_weight = is_jqv4_manifest(filename)
        self.include_layer_stack_blend = (
            self.include_eval_weight
            and self.feature_set.decode("utf-8").startswith("HalfKAv2_hm_jieqi_v8")
        )
        self.stream = None
        self._owner_pid = os.getpid()

        creator = bindings.select_creator(rank, world_size)
        create_args = (
            self.feature_set,
            self.num_workers,
            self.encoded_filename,
            self.batch_size,
            cyclic,
            filtered,
            random_fen_skipping,
        )
        if creator is bindings.create_v2:
            self.stream = creator(*create_args, rank, world_size)
        else:
            self.stream = creator(*create_args)
        if not self.stream:
            detail = bindings.get_last_error()
            raise RuntimeError(
                f"failed to create sparse stream for {filename} "
                f"at rank {rank}/{world_size}: "
                f"{detail or 'unknown loader error'}"
            )

    def __iter__(self):
        return self

    def __next__(self):
        v = self.fetch_next(self.stream)

        if v:
            try:
                return v.contents.get_tensors(
                    self.device, include_eval_weight=self.include_eval_weight,
                    include_layer_stack_blend=self.include_layer_stack_blend)
            finally:
                self.destroy_part(v)
        detail = self.bindings.get_last_error()
        if detail:
            raise RuntimeError(
                f"sparse stream failed for {self.filename!r} at rank "
                f"{self.rank}/{self.world_size}: {detail}"
            )
        raise StopIteration

    def close(self, _os=os):
        stream = getattr(self, "stream", None)
        if not stream:
            return
        self.stream = None
        if _os.getpid() == getattr(self, "_owner_pid", None):
            self.destroy_stream(stream)

    def __del__(self):
        self.close()


class SparseBatchProvider(TrainingDataProvider):
    def __init__(self, feature_set, filename, batch_size, cyclic=True, num_workers=1, filtered=False, random_fen_skipping=0, device='cpu', rank=0, world_size=1):
        _reject_unsupported_multi_rank(feature_set, rank, world_size)
        super(SparseBatchProvider, self).__init__(
            feature_set,
            SPARSE_STREAM_BINDINGS,
            filename,
            cyclic,
            num_workers,
            batch_size,
            filtered,
            random_fen_skipping,
            device,
            rank,
            world_size)

class SparseBatchDataset(torch.utils.data.IterableDataset):
  def __init__(self, feature_set, filename, batch_size, cyclic=True, num_workers=1, filtered=False, random_fen_skipping=0, device='cpu', rank=0, world_size=1):
    super().__init__()
    (
      num_workers,
      batch_size,
      random_fen_skipping,
      rank,
      world_size,
    ) = _checked_sparse_stream_arguments(
      num_workers,
      batch_size,
      random_fen_skipping,
      rank,
      world_size,
    )
    _reject_unsupported_multi_rank(feature_set, rank, world_size)
    self.feature_set = feature_set
    self.filename = filename
    self.batch_size = batch_size
    self.cyclic = cyclic
    self.num_workers = num_workers
    self.filtered = filtered
    self.random_fen_skipping = random_fen_skipping
    self.device = device
    self.rank = rank
    self.world_size = world_size
    # JQv4 manifests are JSON descriptors, not V3 binary shards.  Keep the
    # existing binary-shard inspection (used by scheduling/debug output) while avoiding
    # an attempted 32-bit magic read from the manifest itself. V8 may still
    # consume a binary shard, so inspect that path as well.
    jqv4_manifest = is_jqv4_manifest(filename)
    self.shard_info = (
        inspect_v3_shard(filename, rank, world_size)
        if feature_set in JIEQI_FEATURE_SETS and not jqv4_manifest
        else None
    )

  def __iter__(self):
    return SparseBatchProvider(self.feature_set, self.filename, self.batch_size, cyclic=self.cyclic, num_workers=self.num_workers, filtered=self.filtered, random_fen_skipping=self.random_fen_skipping, device=self.device, rank=self.rank, world_size=self.world_size)

class FixedNumBatchesDataset(torch.utils.data.IterableDataset):
  def __init__(self, dataset, num_batches):
    super().__init__()
    self.dataset = dataset
    self.num_batches = num_batches
    self._iterator = None
    self._iterator_pid = None

  def __len__(self):
    return self.num_batches

  def __iter__(self):
    if torch.utils.data.get_worker_info() is not None:
      raise RuntimeError(
          "FixedNumBatchesDataset requires DataLoader num_workers=0"
      )
    current_pid = os.getpid()
    if self._iterator is None or self._iterator_pid != current_pid:
      self._iterator = iter(self.dataset)
      self._iterator_pid = current_pid
    for _ in range(self.num_batches):
      yield next(self._iterator)

  def __getstate__(self):
    state = self.__dict__.copy()
    state["_iterator"] = None
    state["_iterator_pid"] = None
    return state
