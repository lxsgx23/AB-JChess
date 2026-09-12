# AB-JChess V8.2 PyTorch trainer

This directory contains the PyTorch training and export source for the
AB-JChess V8.2 NNUE network. It includes feature encoding, the model,
dataset/loading code, distributed training helpers, checkpoint validation,
runtime serialization, and probability-table calibration.

## Environment

Install the dependencies from `requirements.txt` (or the CUDA-specific file)
with Python 3.10 or newer. Build the native data loader with CMake when the
training data path requires it:

```powershell
cmake -S . -B build-v8 -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-v8 --config RelWithDebInfo
Copy-Item build-v8\libtraining_data_loader.dll .\libtraining_data_loader.dll -Force
```

The loader and trainer are self-contained in this directory. Training input,
checkpoints, logs, and exported packages should be placed outside the source
tree.

## Train and export

Use `train_v8.py` for training and `serialize_v8.py` for an authenticated
V8.2 runtime package. A normal export supplies both probability-table files:

```powershell
python .\train_v8.py <train-data> <validation-data> `
  --features HalfKAv2_hm_jieqi_v8^ --gpus 1

python .\serialize_v8.py <checkpoint> <output.nnue> `
  --features HalfKAv2_hm_jieqi_v8^ `
  --probability-score-to-mass <score-to-mass.i32le> `
  --probability-mass-to-score <mass-to-score.i32le>
```

The exporter emits the V8.2 `ABJCHESSV82` container. The ABI keeps score input
at `-2000..2000`, uses 4001 `score-to-mass` entries, and supports effective
mass output `-950..950` with 1901 `mass-to-score` entries.

Use `scripts/calibrate_v8_probability.py` to fit and verify probability
tables from an explicit score/outcome trace. It never modifies an existing
NNUE package unless an explicit output path is supplied.

## Engine handoff

The engine does not infer a package name or search the source directory. Pass
the exported file explicitly through UCI:

```text
setoption name EvalFile value C:\path\to\output.nnue
```

The tuning scheduler is a separate package and is deliberately not included
in this source directory.
