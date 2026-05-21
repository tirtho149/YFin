"""finbench — a staged ML benchmark for stock close-price direction prediction.

Pipeline (run the scripts in order):
  1. download data        scripts/01_download_data.py
  2. build feature panels scripts/02_build_panels.py
  3. run the benchmark    scripts/03_run_benchmark.py
  4. walk-forward eval    scripts/04_walkforward.py

The headline task is **predicting the direction of the next close price**
(the ``Direction`` target); return and volatility are also benchmarked.
"""
__version__ = "1.0.0"
