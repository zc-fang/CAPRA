# performance_comparison

This directory contains the benchmark-facing evaluation and wrapper scripts used
in the CAPRA study.

## Provenance

The code in `performance_comparison/scPerturBench/` is adapted from the public
`bm2-lab/scPerturBench` repository:

- repository: `https://github.com/bm2-lab/scPerturBench`
- benchmark paper: Wei et al., *Benchmarking algorithms for generalizable
  single-cell perturbation response prediction*, Nature Methods (2025)

Several files preserve the original benchmarking logic with local adjustments
for repository-relative path resolution, CAPRA method integration, and local
benchmark execution.

## Directory Intent

- `calPerformance_genetic.py`: benchmark evaluation entry point.
- `scPerturBench/`: adapted method wrappers and utility functions.

This subtree is not presented as a from-scratch reimplementation. It is a
benchmark-integration layer that depends on and should acknowledge the original
scPerturBench codebase.
