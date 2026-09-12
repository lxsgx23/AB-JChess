"""Fit and validate independent V8.2 chance-search probability tables.

The runtime ABI is intentionally fixed here: scores cover ``[-2000, 2000]``
and probability mass covers ``[-950, 950]``.  This tool only writes the two
raw lookup-table files requested by the caller; it never edits an NNUE
package.  A trace is a CSV/JSONL record stream with ``score`` and ``outcome``
fields, or an NPZ containing the corresponding arrays.  NPY arrays can be
supplied with ``--scores`` and ``--results``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


SCORE_MIN, SCORE_MAX = -2000, 2000
MASS_MIN, MASS_MAX = -950, 950
SCORE_COUNT = SCORE_MAX - SCORE_MIN + 1
MASS_COUNT = MASS_MAX - MASS_MIN + 1

CALIBRATION_REPORT_SCHEMA = "abjchess-v8.2-probability-calibration-v1"
TABLE_FORMAT = "v8.2-asymmetric-mass-i32le-v1"
CALIBRATION_THRESHOLDS = {
    "max_roundtrip_error_cp": 12.0,
    "mae_roundtrip_error_cp": 1.5,
    "p99_roundtrip_error_cp": 8.0,
    "ece": 0.01,
}


class CalibrationError(ValueError):
    """Raised when calibration input or a table violates the V8 contract."""


def round_half_away(value: float | np.ndarray) -> int | np.ndarray:
    """Round ties away from zero independently of the host C locale."""

    if isinstance(value, np.ndarray):
        return np.where(value >= 0, np.floor(value + 0.5),
                        np.ceil(value - 0.5)).astype(np.int64)
    return int(math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5))


@dataclass(frozen=True)
class Trace:
    scores: np.ndarray
    outcomes: np.ndarray
    weights: np.ndarray | None = None
    game_id: np.ndarray | None = None
    eval_weight: np.ndarray | None = None


@dataclass(frozen=True)
class PreparedData:
    scores: np.ndarray
    outcomes: np.ndarray
    weights: np.ndarray
    game_ids: np.ndarray | None
    records: int
    games: int


@dataclass(frozen=True)
class LogisticFit:
    tau: float
    used_weight: float
    objective: float
    records: int
    games: int


@dataclass(frozen=True)
class ProbabilityTables:
    tau: float
    score_to_mass: np.ndarray
    mass_to_score: np.ndarray

    def __post_init__(self) -> None:
        if self.score_to_mass.shape != (SCORE_COUNT,):
            raise CalibrationError("score-to-mass table length differs from V8.2 ABI")
        if self.mass_to_score.shape != (MASS_COUNT,):
            raise CalibrationError("mass-to-score table length differs from V8.2 ABI")


def _as_vector(name: str, values: Any, *, dtype: Any = np.float64) -> np.ndarray:
    try:
        value = np.asarray(values, dtype=dtype).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise CalibrationError(f"{name} must be a one-dimensional array") from exc
    if value.size == 0:
        raise CalibrationError(f"{name} must be non-empty")
    return value


def _canonical_game_keys(values: np.ndarray, size: int) -> list[tuple[str, str]]:
    if values.size != size:
        raise CalibrationError("game_id length does not match score")
    result: list[tuple[str, str]] = []
    for value in values.tolist():
        if value is None:
            raise CalibrationError("game_id values must not be null")
        if isinstance(value, float) and not math.isfinite(value):
            raise CalibrationError("game_id values must be finite")
        result.append((type(value).__name__, str(value)))
    return result


def prepare_calibration_data(trace: Trace) -> PreparedData:
    """Filter eligible records and assign flat or equal-per-game weights.

    ``eval_weight`` is a gate: only values exactly equal to ``1`` are used.
    Optional per-record ``weights`` are retained without game IDs and are
    normalized within each game when game IDs are present, so every game has
    one unit of total calibration mass.
    """

    scores = _as_vector("score", trace.scores)
    outcomes = _as_vector("outcome", trace.outcomes)
    if scores.size != outcomes.size:
        raise CalibrationError("score and outcome must have equal lengths")
    if not np.isfinite(scores).all() or not np.isfinite(outcomes).all():
        raise CalibrationError("score and outcome values must be finite")
    if np.any((outcomes < 0.0) | (outcomes > 1.0)):
        raise CalibrationError("outcome values must be in [0,1]")

    if trace.eval_weight is None:
        eligible = np.ones(scores.size, dtype=bool)
    else:
        eval_weight = _as_vector("eval_weight", trace.eval_weight)
        if eval_weight.size != scores.size:
            raise CalibrationError("eval_weight length does not match score")
        if not np.isfinite(eval_weight).all():
            raise CalibrationError("eval_weight values must be finite")
        eligible = eval_weight == 1.0
    if not np.any(eligible):
        raise CalibrationError("at least one eval_weight value must equal exactly 1")

    if trace.weights is None:
        raw_weights = np.ones(scores.size, dtype=np.float64)
    else:
        raw_weights = _as_vector("weight", trace.weights)
        if raw_weights.size != scores.size:
            raise CalibrationError("weight length does not match score")
        if not np.isfinite(raw_weights).all() or np.any(raw_weights < 0.0):
            raise CalibrationError("weight values must be finite and non-negative")

    game_ids: np.ndarray | None = None
    keys: list[tuple[str, str]] | None = None
    if trace.game_id is not None:
        game_ids = np.asarray(trace.game_id, dtype=object).reshape(-1)
        keys = _canonical_game_keys(game_ids, scores.size)

    selected = np.flatnonzero(eligible)
    scores = scores[selected]
    outcomes = outcomes[selected]
    weights = raw_weights[selected].astype(np.float64, copy=True)
    if game_ids is not None:
        game_ids = game_ids[selected]
        selected_keys = [keys[int(index)] for index in selected]
        unique_keys = sorted(set(selected_keys))
        for key in unique_keys:
            local = np.asarray([item == key for item in selected_keys], dtype=bool)
            denominator = float(weights[local].sum())
            if denominator <= 0.0:
                raise CalibrationError("each eligible game must have positive weight")
            weights[local] /= denominator
        games = len(unique_keys)
    else:
        games = 1
    if float(weights.sum()) <= 0.0:
        raise CalibrationError("eligible calibration weights must have positive sum")
    return PreparedData(scores, outcomes, weights, game_ids,
                        int(scores.size), int(games))


def normalized_eval_weights(eval_weight: Sequence[float] | None,
                            game_id: Sequence[Any] | None = None,
                            *, size: int | None = None) -> np.ndarray:
    """Return equal-per-game weights for rows with ``eval_weight == 1``.

    This small array-oriented API is useful to callers that prepare arrays
    directly instead of constructing :class:`Trace`.  The returned vector has
    the original length, with ineligible rows set to zero.  When game IDs are
    supplied, every eligible game contributes exactly one unit of total
    weight; otherwise every eligible record contributes one unit.
    """

    if eval_weight is None:
        if size is None:
            if game_id is None:
                raise CalibrationError("size is required when eval_weight is absent")
            size = int(np.asarray(game_id).reshape(-1).size)
        raw = np.ones(int(size), dtype=np.float64)
    else:
        try:
            raw = np.asarray(eval_weight, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise CalibrationError("eval_weight must be a one-dimensional array") from exc
        if raw.size == 0:
            raise CalibrationError("eval_weight must be non-empty")
        if size is not None and raw.size != int(size):
            raise CalibrationError("eval_weight length does not match score")
    if not np.isfinite(raw).all():
        raise CalibrationError("eval_weight values must be finite")
    eligible = raw == 1.0
    if not np.any(eligible):
        raise CalibrationError("at least one eval_weight value must equal exactly 1")

    keys = _canonical_game_keys(np.asarray(game_id, dtype=object).reshape(-1), raw.size) \
        if game_id is not None else None
    result = np.zeros(raw.size, dtype=np.float64)
    if keys is None:
        result[eligible] = 1.0
        return result

    counts: dict[tuple[str, str], int] = {}
    for index, key in enumerate(keys):
        if eligible[index]:
            counts[key] = counts.get(key, 0) + 1
    for index, key in enumerate(keys):
        if eligible[index]:
            result[index] = 1.0 / counts[key]
    return result


def _field_value(row: Mapping[str, Any], names: Sequence[str]) -> Any:
    lowered = {str(key).strip().lower(): value for key, value in row.items()}
    for name in names:
        if name in lowered:
            return lowered[name]
    return None


def _parse_float(value: Any, field: str, line: int) -> float:
    if value is None or str(value).strip() == "":
        raise CalibrationError(f"missing {field} at input row {line}")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise CalibrationError(f"invalid {field} at input row {line}: {value!r}") from exc


def _trace_from_rows(rows: Sequence[Mapping[str, Any]]) -> Trace:
    if not rows:
        raise CalibrationError("trace contains no records")
    scores: list[float] = []
    outcomes: list[float] = []
    weights: list[float] = []
    eval_weights: list[float] = []
    games: list[Any] = []
    have_weight = have_eval = have_game = False
    for line, row in enumerate(rows, 1):
        score = _field_value(row, ("score", "scores"))
        outcome = _field_value(row, ("outcome", "result", "results"))
        scores.append(_parse_float(score, "score", line))
        outcomes.append(_parse_float(outcome, "outcome", line))
        weight = _field_value(row, ("weight", "weights"))
        eval_weight = _field_value(row, ("eval_weight", "eval-weight", "evalweight"))
        game = _field_value(row, ("game_id", "game-id", "gameid", "game"))
        if weight is not None and str(weight).strip() != "":
            have_weight = True
            weights.append(_parse_float(weight, "weight", line))
        else:
            weights.append(1.0)
        if eval_weight is not None and str(eval_weight).strip() != "":
            have_eval = True
            eval_weights.append(_parse_float(eval_weight, "eval_weight", line))
        else:
            eval_weights.append(1.0)
        if game is not None and str(game).strip() != "":
            have_game = True
            games.append(game)
        else:
            games.append("")
    return Trace(
        np.asarray(scores, dtype=np.float64),
        np.asarray(outcomes, dtype=np.float64),
        np.asarray(weights, dtype=np.float64) if have_weight else None,
        np.asarray(games, dtype=object) if have_game else None,
        np.asarray(eval_weights, dtype=np.float64) if have_eval else None,
    )


def load_trace(path: str | Path) -> Trace:
    """Load CSV, JSONL/JSON, or NPZ trace data."""

    source = Path(path)
    if not source.exists():
        raise CalibrationError(f"trace does not exist: {source}")
    suffix = source.suffix.lower()
    try:
        if suffix == ".csv":
            with source.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None:
                    raise CalibrationError("CSV trace must have a header")
                return _trace_from_rows(list(reader))
        if suffix in {".jsonl", ".ndjson", ".json"}:
            text = source.read_text(encoding="utf-8")
            if suffix == ".json":
                value = json.loads(text)
                rows = value if isinstance(value, list) else value.get("records")
                if not isinstance(rows, list):
                    raise CalibrationError("JSON trace must be an array or records object")
            else:
                rows = []
                for line, raw in enumerate(text.splitlines(), 1):
                    if not raw.strip():
                        continue
                    try:
                        value = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        raise CalibrationError(f"invalid JSONL row {line}: {exc}") from exc
                    if not isinstance(value, dict):
                        raise CalibrationError(f"JSONL row {line} must be an object")
                    rows.append(value)
            if not all(isinstance(row, dict) for row in rows):
                raise CalibrationError("JSON trace records must be objects")
            return _trace_from_rows(rows)
        if suffix == ".npz":
            with np.load(source, allow_pickle=True) as archive:
                def take(names: Sequence[str]) -> np.ndarray | None:
                    for name in names:
                        if name in archive.files:
                            return np.asarray(archive[name])
                    return None
                scores = take(("scores", "score"))
                outcomes = take(("outcomes", "results", "result", "outcome"))
                if scores is None or outcomes is None:
                    raise CalibrationError("NPZ trace requires scores and outcomes arrays")
                return Trace(
                    scores, outcomes,
                    take(("weights", "weight")),
                    take(("game_id", "game_ids", "game")),
                    take(("eval_weight", "eval_weights", "evalweight")),
                )
    except OSError as exc:
        raise CalibrationError(f"cannot read trace: {source}") from exc
    raise CalibrationError(
        f"unsupported trace format {source.suffix!r}; use CSV, JSONL, JSON, or NPZ"
    )


def load_array_trace(scores_path: str | Path, results_path: str | Path,
                     weights_path: str | Path | None = None,
                     game_id_path: str | Path | None = None,
                     eval_weight_path: str | Path | None = None) -> Trace:
    """Load the V9-style separate NPY arrays used by calibration streams."""

    def read(path: str | Path, label: str) -> np.ndarray:
        try:
            return np.load(Path(path), allow_pickle=True)
        except (OSError, ValueError) as exc:
            raise CalibrationError(f"cannot read {label} array: {path}") from exc

    return Trace(
        read(scores_path, "score"),
        read(results_path, "outcome"),
        None if weights_path is None else read(weights_path, "weight"),
        None if game_id_path is None else read(game_id_path, "game_id"),
        None if eval_weight_path is None else read(eval_weight_path, "eval_weight"),
    )


def _objective(log_tau: float, data: PreparedData) -> float:
    tau = math.exp(float(log_tau))
    z = np.clip(data.scores / tau, -60.0, 60.0)
    loss = np.maximum(z, 0.0) - data.outcomes * z + np.log1p(np.exp(-np.abs(z)))
    return float(np.sum(data.weights * loss) / data.weights.sum())


def fit_temperature(trace: Trace | PreparedData, *, tau_bounds: tuple[float, float] = (25.0, 10_000.0),
                    iterations: int = 96) -> LogisticFit:
    """Fit ``sigmoid(score / tau)`` by deterministic golden-section search."""

    data = trace if isinstance(trace, PreparedData) else prepare_calibration_data(trace)
    low, high = map(float, tau_bounds)
    if not (0.0 < low < high) or iterations < 8:
        raise CalibrationError("invalid tau bounds or iteration count")
    a, b = math.log(low), math.log(high)
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    c = b - (b - a) / phi
    d = a + (b - a) / phi
    fc, fd = _objective(c, data), _objective(d, data)
    for _ in range(int(iterations)):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - (b - a) / phi
            fc = _objective(c, data)
        else:
            a, c, fc = c, d, fd
            d = a + (b - a) / phi
            fd = _objective(d, data)
    tau = math.exp((a + b) / 2.0)
    return LogisticFit(tau, float(data.weights.sum()), _objective(math.log(tau), data),
                       data.records, data.games)


def fit_symmetric_logistic(score: Sequence[float], result: Sequence[float],
                           eval_weight: Sequence[float] | None = None, *,
                           game_id: Sequence[Any] | None = None,
                           weight: Sequence[float] | None = None,
                           tau_bounds: tuple[float, float] = (25.0, 10_000.0),
                           iterations: int = 96) -> LogisticFit:
    """Fit ``sigmoid(score / tau)`` from separate arrays.

    ``eval_weight`` remains a strict eligibility gate.  Optional ``weight``
    values are calibration weights and are normalized per game through the
    same path used by the trace loader.
    """

    return fit_temperature(
        Trace(np.asarray(score), np.asarray(result),
              None if weight is None else np.asarray(weight),
              None if game_id is None else np.asarray(game_id, dtype=object),
              None if eval_weight is None else np.asarray(eval_weight)),
        tau_bounds=tau_bounds, iterations=iterations)


def build_probability_tables(tau: float) -> ProbabilityTables:
    """Build monotone V8.2 tables while preserving the asymmetric mass ABI."""

    tau = float(tau)
    if not math.isfinite(tau) or tau <= 0.0:
        raise CalibrationError("tau must be finite and positive")
    endpoint = math.tanh(SCORE_MAX / (2.0 * tau))
    if not 0.0 < endpoint <= 1.0:
        raise CalibrationError("tau gives an unusable logistic domain")

    scores = np.arange(SCORE_MIN, SCORE_MAX + 1, dtype=np.float64)
    normalized = np.tanh(scores / (2.0 * tau)) / endpoint
    positive = np.asarray(round_half_away(
        MASS_MAX * np.maximum(normalized, 0.0)), dtype=np.int64)
    negative = np.asarray(round_half_away(
        (-MASS_MIN) * np.minimum(normalized, 0.0)), dtype=np.int64)
    masses = np.where(scores >= 0.0, positive, negative).astype(np.int32)
    masses[0], masses[SCORE_COUNT // 2], masses[-1] = MASS_MIN, 0, MASS_MAX
    masses = np.maximum.accumulate(masses).astype(np.int32)

    mass_values = np.arange(MASS_MIN, MASS_MAX + 1, dtype=np.float64)
    inverse_arg = np.where(
        mass_values < 0.0,
        (mass_values / float(-MASS_MIN)) * endpoint,
        (mass_values / float(MASS_MAX)) * endpoint,
    )
    inverse_arg = np.clip(inverse_arg, -1.0, 1.0)
    inverse = np.empty(MASS_COUNT, dtype=np.int64)
    interior = np.abs(inverse_arg) < 1.0
    inverse[interior] = round_half_away(
        2.0 * tau * np.arctanh(inverse_arg[interior]))
    inverse[~interior] = np.where(inverse_arg[~interior] < 0.0,
                                  SCORE_MIN, SCORE_MAX)
    inverse = np.clip(inverse, SCORE_MIN, SCORE_MAX)
    inverse[0], inverse[-MASS_MIN], inverse[-1] = SCORE_MIN, 0, SCORE_MAX
    inverse = np.maximum.accumulate(inverse).astype(np.int32)
    return ProbabilityTables(tau, masses, inverse)


def validate_probability_tables(tables: ProbabilityTables) -> dict[str, float | bool]:
    """Check V8.2 dimensions, ordering, closed endpoints, and round trips."""

    s2m = np.asarray(tables.score_to_mass)
    m2s = np.asarray(tables.mass_to_score)
    if s2m.shape != (SCORE_COUNT,) or m2s.shape != (MASS_COUNT,):
        raise CalibrationError("probability tables have invalid V8.2 lengths")
    if s2m.dtype.kind not in "iu" or m2s.dtype.kind not in "iu":
        raise CalibrationError("probability tables must use integer arrays")
    if np.any((s2m < MASS_MIN) | (s2m > MASS_MAX)):
        raise CalibrationError("score-to-mass values are outside V8.2 mass range")
    if np.any((m2s < SCORE_MIN) | (m2s > SCORE_MAX)):
        raise CalibrationError("mass-to-score values are outside V8.2 score range")
    if np.any(np.diff(s2m) < 0) or np.any(np.diff(m2s) < 0):
        raise CalibrationError("probability tables are not monotone")
    if int(s2m[0]) != MASS_MIN or int(s2m[-1]) != MASS_MAX or int(s2m[SCORE_COUNT // 2]) != 0:
        raise CalibrationError("score-to-mass endpoints or zero point are invalid")
    if int(m2s[0]) != SCORE_MIN or int(m2s[-1]) != SCORE_MAX or int(m2s[-MASS_MIN]) != 0:
        raise CalibrationError("mass-to-score endpoints or zero point are invalid")
    scores = np.arange(SCORE_MIN, SCORE_MAX + 1, dtype=np.int64)
    roundtrip = m2s[np.clip(s2m.astype(np.int64) - MASS_MIN, 0, MASS_COUNT - 1)]
    errors = np.abs(roundtrip.astype(np.int64) - scores)
    mass_rt = s2m[np.clip(m2s.astype(np.int64) - SCORE_MIN, 0, SCORE_COUNT - 1)]
    mass_errors = np.abs(mass_rt.astype(np.int64) - np.arange(MASS_MIN, MASS_MAX + 1))
    return {
        "score_to_mass_count": SCORE_COUNT,
        "mass_to_score_count": MASS_COUNT,
        "score_to_mass_min": int(s2m.min()),
        "score_to_mass_max": int(s2m.max()),
        "mass_to_score_min": int(m2s.min()),
        "mass_to_score_max": int(m2s.max()),
        "max_roundtrip_error_cp": int(errors.max()),
        "mae_roundtrip_error_cp": float(errors.mean()),
        "p99_roundtrip_error_cp": float(np.quantile(errors, 0.99)),
        "max_mass_roundtrip_error": int(mass_errors.max()),
        "monotone": True,
        "endpoint_closed": True,
        "zero_point": True,
        # The V8.2 ABI uses symmetric mass range ([-950, 950]),
        # so strict odd symmetry is impossible without changing the ABI.
        "odd_symmetric": bool(np.array_equal(s2m, -s2m[::-1]) and
                               np.array_equal(m2s, -m2s[::-1])),
    }


def _sigmoid(scores: np.ndarray, tau: float) -> np.ndarray:
    z = np.clip(np.asarray(scores, dtype=np.float64) / float(tau), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def _calibration_metrics_arrays(probability: np.ndarray, outcomes: np.ndarray,
                                weights: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    probability = np.asarray(probability, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if not (probability.size == outcomes.size == weights.size == scores.size):
        raise CalibrationError("calibration metric arrays must have equal lengths")
    total = float(weights.sum())
    if total <= 0.0:
        raise CalibrationError("calibration weights must have positive sum")
    edges = np.linspace(0.0, 1.0, 21)
    bins = np.clip(np.searchsorted(edges, probability, side="right") - 1, 0, 19)
    ece = 0.0
    center_max = 0.0
    center = np.abs(scores) <= 400.0
    for mask in (np.ones(probability.size, dtype=bool), center):
        grouped_error = 0.0
        for index in range(20):
            selected = mask & (bins == index)
            if not np.any(selected):
                continue
            mass = float(weights[selected].sum())
            error = abs(float(np.sum(weights[selected] * probability[selected]) / mass)
                        - float(np.sum(weights[selected] * outcomes[selected]) / mass))
            if mask is center:
                center_max = max(center_max, error)
            else:
                grouped_error += (mass / total) * error
        if mask is not center:
            ece = grouped_error
    brier = float(np.sum(weights * (probability - outcomes) ** 2) / total)
    return {"ece": float(ece), "center_max_error": float(center_max), "brier": brier}


def calibration_metrics(data: PreparedData, tau: float) -> dict[str, float]:
    return _calibration_metrics_arrays(
        _sigmoid(data.scores, tau), data.outcomes, data.weights, data.scores)


def calibration_report(score: Sequence[float], result: Sequence[float],
                       eval_weight: Sequence[float] | None = None, *,
                       game_id: Sequence[Any] | None = None, tau: float,
                       weight: Sequence[float] | None = None,
                       reference_score: Sequence[float] | None = None,
                       bootstrap_samples: int = 2000,
                       bootstrap_seed: int = 20260828) -> dict[str, Any]:
    """Report calibration and optional whole-game Brier non-inferiority.

    Eligible records follow the V8.2 policy: only ``eval_weight == 1`` rows
    participate, and game IDs make every game contribute equal total weight.
    When a reference score stream is supplied, the bootstrap resamples whole
    games rather than individual positions, so long games cannot dominate the
    comparison.
    """

    tau = float(tau)
    if not math.isfinite(tau) or tau <= 0.0:
        raise CalibrationError("tau must be finite and positive")
    if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int):
        raise CalibrationError("bootstrap_samples must be a non-negative integer")
    if bootstrap_samples < 0:
        raise CalibrationError("bootstrap_samples must be a non-negative integer")

    raw_scores = _as_vector("score", score)
    raw_weight = None if weight is None else _as_vector("weight", weight)
    if raw_weight is not None and raw_weight.size != raw_scores.size:
        raise CalibrationError("weight length does not match score")
    prepared = prepare_calibration_data(
        Trace(raw_scores, np.asarray(result),
              weights=raw_weight,
              game_id=None if game_id is None else np.asarray(game_id, dtype=object),
              eval_weight=None if eval_weight is None else np.asarray(eval_weight)))
    eligible_weights = normalized_eval_weights(
        eval_weight, game_id, size=raw_scores.size)
    keep = eligible_weights > 0.0
    if raw_weight is not None:
        keep &= raw_weight > 0.0
        positive = prepared.weights > 0.0
        if not bool(np.all(positive)):
            prepared = PreparedData(
                prepared.scores[positive], prepared.outcomes[positive],
                prepared.weights[positive],
                None if prepared.game_ids is None else prepared.game_ids[positive],
                int(positive.sum()), prepared.games)

    reference = None
    if reference_score is not None:
        reference = _as_vector("reference_score", reference_score)
        if reference.size != raw_scores.size:
            raise CalibrationError("reference_score length does not match score")
        if not np.isfinite(reference).all():
            raise CalibrationError("reference_score values must be finite")
        reference = reference[keep]

    metrics = calibration_metrics(prepared, tau)
    table_validation = validate_probability_tables(build_probability_tables(tau))

    if prepared.game_ids is None:
        groups = [np.asarray([index], dtype=np.int64)
                  for index in range(prepared.records)]
    else:
        keys = _canonical_game_keys(prepared.game_ids, prepared.records)
        grouped: dict[tuple[str, str], list[int]] = {}
        for index, key in enumerate(keys):
            grouped.setdefault(key, []).append(index)
        groups = [np.asarray(grouped[key], dtype=np.int64)
                  for key in sorted(grouped)]

    reference_brier = None
    brier_delta = None
    ci: list[float | None] = [None, None]
    if reference is not None:
        reference_metrics = _calibration_metrics_arrays(
            _sigmoid(reference, tau), prepared.outcomes, prepared.weights,
            prepared.scores)
        reference_brier = reference_metrics["brier"]
        brier_delta = float(metrics["brier"] - reference_brier)
        if bootstrap_samples:
            rng = np.random.default_rng(int(bootstrap_seed))
            deltas: list[float] = []
            group_count = len(groups)
            for _ in range(bootstrap_samples):
                picked = rng.integers(0, group_count, size=group_count)
                indices = np.concatenate([groups[int(index)] for index in picked])
                sample_scores = prepared.scores[indices]
                sample_outcomes = prepared.outcomes[indices]
                sample_reference = reference[indices]
                if prepared.game_ids is None:
                    sample_weights = np.ones(indices.size, dtype=np.float64)
                else:
                    sample_weights = np.asarray(
                        [1.0 / groups[int(group)].size
                         for group in picked
                         for _ in groups[int(group)]],
                        dtype=np.float64,
                    )
                candidate_brier = _calibration_metrics_arrays(
                    _sigmoid(sample_scores, tau), sample_outcomes,
                    sample_weights, sample_scores)["brier"]
                baseline_brier = _calibration_metrics_arrays(
                    _sigmoid(sample_reference, tau), sample_outcomes,
                    sample_weights, sample_scores)["brier"]
                deltas.append(candidate_brier - baseline_brier)
            ci = [float(np.quantile(deltas, 0.025)),
                  float(np.quantile(deltas, 0.975))]
        else:
            ci = [brier_delta, brier_delta]

    report = {
        "schema": CALIBRATION_REPORT_SCHEMA,
        "tau": tau,
        "records": int(prepared.records),
        "games": int(len(groups)),
        **metrics,
        "max_roundtrip_error_cp": int(table_validation["max_roundtrip_error_cp"]),
        "mae_roundtrip_error_cp": float(table_validation["mae_roundtrip_error_cp"]),
        "p99_roundtrip_error_cp": float(table_validation["p99_roundtrip_error_cp"]),
        "reference_brier": reference_brier,
        "brier_delta": brier_delta,
        "brier_delta_ci95": ci,
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(bootstrap_seed),
        "bootstrap_noninferior": (
            None if brier_delta is None else bool(ci[1] is not None and ci[1] <= 0.0)
        ),
        "validation": table_validation,
    }
    report["publication_gate"] = _publication_gate(table_validation, report)
    return report


def _publication_gate(validation: Mapping[str, Any], metrics: Mapping[str, Any] | None) -> dict[str, Any]:
    reasons: list[str] = ["no playing-strength A/B evidence"]
    for key in ("max_roundtrip_error_cp", "mae_roundtrip_error_cp", "p99_roundtrip_error_cp"):
        if float(validation[key]) > CALIBRATION_THRESHOLDS[key]:
            reasons.append(f"{key} exceeds threshold")
    if metrics is None:
        status = "candidate_only"
    else:
        if float(metrics["ece"]) > CALIBRATION_THRESHOLDS["ece"]:
            reasons.append("ece exceeds threshold")
        status = "failed" if len(reasons) > 1 else "candidate_only"
    return {"status": status, "passed": False, "reasons": reasons,
            "thresholds": dict(CALIBRATION_THRESHOLDS)}


def write_probability_tables(tables: ProbabilityTables, score_to_mass: str | Path,
                             mass_to_score: str | Path, *, force: bool = False) -> tuple[Path, Path]:
    validate_probability_tables(tables)
    outputs = (Path(score_to_mass), Path(mass_to_score))
    for path in outputs:
        if path.suffix.lower() == ".nnue":
            raise CalibrationError("probability tool writes raw tables, not NNUE packages")
        if path.exists() and not force:
            raise CalibrationError(f"refusing to overwrite existing output: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    outputs[0].write_bytes(np.asarray(tables.score_to_mass, dtype="<i4").tobytes())
    outputs[1].write_bytes(np.asarray(tables.mass_to_score, dtype="<i4").tobytes())
    return outputs


def read_probability_tables(score_to_mass: str | Path, mass_to_score: str | Path,
                            *, tau: float = 1.0) -> ProbabilityTables:
    def read(path: str | Path, count: int, label: str) -> np.ndarray:
        path = Path(path)
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise CalibrationError(f"cannot read {label} table: {path}") from exc
        expected = count * 4
        if len(raw) != expected:
            raise CalibrationError(f"{label} table must be exactly {expected} bytes; got {len(raw)}")
        return np.frombuffer(raw, dtype="<i4").copy()
    tables = ProbabilityTables(float(tau), read(score_to_mass, SCORE_COUNT, "score-to-mass"),
                               read(mass_to_score, MASS_COUNT, "mass-to-score"))
    validate_probability_tables(tables)
    return tables


def _resolve_trace(trace: str | Path | None, scores_path: str | Path | None,
                   results_path: str | Path | None, weights_path: str | Path | None,
                   game_id_path: str | Path | None, eval_weight_path: str | Path | None) -> Trace | None:
    if trace is not None:
        if scores_path is not None or results_path is not None:
            raise CalibrationError("provide --trace or --scores/--results, not both")
        return load_trace(trace)
    if (scores_path is None) != (results_path is None):
        raise CalibrationError("--scores and --results must be supplied together")
    if scores_path is None:
        return None
    return load_array_trace(scores_path, results_path, weights_path, game_id_path, eval_weight_path)


def fit_from_inputs(*, trace: str | Path | None,
                    scores_path: str | Path | None,
                    results_path: str | Path | None,
                    weights_path: str | Path | None = None,
                    game_id_path: str | Path | None = None,
                    eval_weight_path: str | Path | None = None,
                    tau: float | None,
                    tau_bounds: tuple[float, float] = (25.0, 10_000.0),
                    iterations: int = 96,
                    score_to_mass: str | Path,
                    mass_to_score: str | Path,
                    report: str | Path,
                    dry_run: bool = False,
                    force: bool = False) -> dict[str, Any]:
    source = _resolve_trace(trace, scores_path, results_path, weights_path,
                            game_id_path, eval_weight_path)
    prepared = None if source is None else prepare_calibration_data(source)
    if tau is None and prepared is None:
        raise CalibrationError("a score/outcome trace is required when --tau is omitted")
    fit = None if tau is not None else fit_temperature(
        prepared, tau_bounds=tau_bounds, iterations=iterations)
    selected_tau = float(tau if tau is not None else fit.tau)
    tables = build_probability_tables(selected_tau)
    validation = validate_probability_tables(tables)
    metrics = None if prepared is None else calibration_metrics(prepared, selected_tau)
    result: dict[str, Any] = {
        "schema": CALIBRATION_REPORT_SCHEMA,
        "table_format": TABLE_FORMAT,
        "abi": {"score_min": SCORE_MIN, "score_max": SCORE_MAX,
                "score_count": SCORE_COUNT, "mass_min": MASS_MIN,
                "mass_max": MASS_MAX, "mass_count": MASS_COUNT},
        "tau": selected_tau,
        "fit": None if fit is None else asdict(fit),
        "records": None if prepared is None else prepared.records,
        "games": None if prepared is None else prepared.games,
        "tables": {"score_to_mass_count": SCORE_COUNT,
                   "mass_to_score_count": MASS_COUNT},
        "validation": validation,
        "metrics": metrics,
        "publication_gate": _publication_gate(validation, metrics),
        "dry_run": bool(dry_run),
        "outputs": {"score_to_mass": str(Path(score_to_mass)),
                    "mass_to_score": str(Path(mass_to_score)),
                    "report": str(Path(report))},
    }
    if metrics is not None:
        result.update(metrics)
    if not dry_run:
        report_path = Path(report)
        if report_path.exists() and not force:
            raise CalibrationError(f"refusing to overwrite existing report: {report_path}")
        write_probability_tables(tables, score_to_mass, mass_to_score, force=force)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n",
                               encoding="ascii", newline="\n")
    return result


def verify_from_inputs(*, score_to_mass: str | Path, mass_to_score: str | Path,
                       tau: float | None, trace: str | Path | None = None,
                       scores_path: str | Path | None = None,
                       results_path: str | Path | None = None,
                       weights_path: str | Path | None = None,
                       game_id_path: str | Path | None = None,
                       eval_weight_path: str | Path | None = None) -> dict[str, Any]:
    tables = read_probability_tables(score_to_mass, mass_to_score,
                                     tau=1.0 if tau is None else tau)
    source = _resolve_trace(trace, scores_path, results_path, weights_path,
                            game_id_path, eval_weight_path)
    prepared = None if source is None else prepare_calibration_data(source)
    metrics = None
    if prepared is not None:
        if tau is None:
            raise CalibrationError("--tau is required when verifying metrics against a trace")
        metrics = calibration_metrics(prepared, tau)
    validation = validate_probability_tables(tables)
    return {"schema": CALIBRATION_REPORT_SCHEMA, "table_format": TABLE_FORMAT,
            "tau": None if tau is None else float(tau), "validation": validation,
            "metrics": metrics, "records": None if prepared is None else prepared.records,
            "games": None if prepared is None else prepared.games,
            "publication_gate": _publication_gate(validation, metrics)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_input_args(command: argparse.ArgumentParser) -> None:
        command.add_argument("--trace", type=Path,
                             help="CSV/JSONL/JSON/NPZ score-outcome trace")
        command.add_argument("--scores", type=Path, help="separate .npy scores array")
        command.add_argument("--results", type=Path, help="separate .npy outcomes array")
        command.add_argument("--weights", "--weight", dest="weights", type=Path)
        command.add_argument("--game-id", "--game-ids", dest="game_id", type=Path)
        command.add_argument("--eval-weight", dest="eval_weight", type=Path)

    fit = sub.add_parser("fit", help="fit tau and emit candidate raw tables")
    add_input_args(fit)
    fit.add_argument("--tau", type=float, help="skip fitting and use this fixed temperature")
    fit.add_argument("--tau-min", type=float, default=25.0)
    fit.add_argument("--tau-max", type=float, default=10_000.0)
    fit.add_argument("--iterations", type=int, default=96)
    fit.add_argument("--score-to-mass", type=Path, required=True)
    fit.add_argument("--mass-to-score", type=Path, required=True)
    fit.add_argument("--report", type=Path, required=True)
    fit.add_argument("--dry-run", action="store_true",
                     help="fit and print the report without writing any files")
    fit.add_argument("--force", action="store_true", help="allow replacing explicit candidate outputs")

    verify = sub.add_parser("verify", help="validate existing raw tables and optional trace metrics")
    add_input_args(verify)
    verify.add_argument("--tau", type=float)
    verify.add_argument("--score-to-mass", type=Path, required=True)
    verify.add_argument("--mass-to-score", type=Path, required=True)
    verify.add_argument("--report", type=Path)
    verify.add_argument("--force", action="store_true")

    report = sub.add_parser(
        "report", help="fit or evaluate tau with optional whole-game reference comparison"
    )
    add_input_args(report)
    report.add_argument("--tau", type=float,
                        help="fixed temperature; omit to fit from the input trace")
    report.add_argument("--tau-min", type=float, default=25.0)
    report.add_argument("--tau-max", type=float, default=10_000.0)
    report.add_argument("--iterations", type=int, default=96)
    report.add_argument("--reference-scores", type=Path,
                        help="optional .npy reference score stream")
    report.add_argument("--bootstrap-samples", type=int, default=2000)
    report.add_argument("--bootstrap-seed", type=int, default=20260828)
    report.add_argument("--score-to-mass", type=Path, required=True)
    report.add_argument("--mass-to-score", type=Path, required=True)
    report.add_argument("--report", type=Path, required=True)
    report.add_argument("--force", action="store_true",
                        help="allow replacing explicit candidate outputs")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "fit":
        result = fit_from_inputs(
            trace=args.trace, scores_path=args.scores, results_path=args.results,
            weights_path=args.weights, game_id_path=args.game_id,
            eval_weight_path=args.eval_weight, tau=args.tau,
            tau_bounds=(args.tau_min, args.tau_max), iterations=args.iterations,
            score_to_mass=args.score_to_mass, mass_to_score=args.mass_to_score,
            report=args.report, dry_run=args.dry_run, force=args.force)
    elif args.command == "verify":
        result = verify_from_inputs(
            score_to_mass=args.score_to_mass, mass_to_score=args.mass_to_score,
            tau=args.tau, trace=args.trace, scores_path=args.scores,
            results_path=args.results, weights_path=args.weights,
            game_id_path=args.game_id, eval_weight_path=args.eval_weight)
        if args.report is not None:
            path = Path(args.report)
            if path.exists() and not args.force:
                raise CalibrationError(f"refusing to overwrite existing report: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n",
                            encoding="ascii", newline="\n")
    else:
        source = _resolve_trace(
            args.trace, args.scores, args.results, args.weights, args.game_id,
            args.eval_weight)
        if source is None:
            raise CalibrationError("a score/outcome trace is required for report")
        fit = None if args.tau is not None else fit_temperature(
            source, tau_bounds=(args.tau_min, args.tau_max),
            iterations=args.iterations)
        selected_tau = float(args.tau if args.tau is not None else fit.tau)
        reference = None
        if args.reference_scores is not None:
            try:
                reference = np.load(args.reference_scores, allow_pickle=True)
            except (OSError, ValueError) as exc:
                raise CalibrationError(
                    f"cannot read reference score array: {args.reference_scores}"
                ) from exc
        result = calibration_report(
            source.scores,
            source.outcomes,
            source.eval_weight,
            game_id=source.game_id,
            weight=source.weights,
            tau=selected_tau,
            reference_score=reference,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        result["fit"] = None if fit is None else asdict(fit)
        result["outputs"] = {
            "score_to_mass": str(Path(args.score_to_mass)),
            "mass_to_score": str(Path(args.mass_to_score)),
            "report": str(Path(args.report)),
        }
        if args.report.exists() and not args.force:
            raise CalibrationError(
                f"refusing to overwrite existing report: {args.report}"
            )
        write_probability_tables(
            build_probability_tables(selected_tau),
            args.score_to_mass,
            args.mass_to_score,
            force=args.force,
        )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(result, sort_keys=True, indent=2) + "\n",
            encoding="ascii",
            newline="\n",
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CalibrationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
