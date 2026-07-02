# Attribution for Benchmark-Comparison Code

The code under `performance_comparison/scPerturBench/` is adapted from the
public `bm2-lab/scPerturBench` repository and should not be presented as an
independent from-scratch implementation.

Original sources to cite:

1. Repository: `https://github.com/bm2-lab/scPerturBench`
2. Benchmark paper: Wei, Z. et al. *Benchmarking algorithms for generalizable
   single-cell perturbation response prediction*. Nature Methods (2025).

In this CAPRA release, local modifications mainly include:

- repository-relative path handling;
- CAPRA method integration;
- local benchmark execution wrappers;
- compatibility fixes for the current workspace layout.

If this subtree is redistributed, the original repository and benchmark paper
should remain cited in the README, manuscript, and derivative code
documentation.
