"""Public entry point for V8.2 probability calibration.

The implementation lives in ``scripts.calibrate_v8_probability`` so it can
also be invoked directly as a command-line tool.  This shim keeps the public
array-oriented API importable from the project root.
"""

from scripts.calibrate_v8_probability import *
