"""Benchmark sweep definitions.

`base` holds the `BenchConfig` schema and a tiny quick-iteration grid; the other
modules are the sweeps that produced the paper's numbers. Select one by its
module name, without the package prefix:

    python run.py --config speedup --execute

See the repository README for which config produced which table or figure.
"""
