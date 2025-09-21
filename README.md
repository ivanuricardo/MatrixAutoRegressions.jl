
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://IvanRicardo.github.io/MatrixAutoRegressions.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://IvanRicardo.github.io/MatrixAutoRegressions.jl/dev/)
[![Build Status](https://github.com/IvanRicardo/MatrixAutoRegressions.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/IvanRicardo/MatrixAutoRegressions.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/IvanRicardo/MatrixAutoRegressions.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/IvanRicardo/MatrixAutoRegressions.jl)


# MAR.jl

**Matrix Autoregressive (MAR) models in Julia**

`MAR.jl` provides tools to estimate, simulate and work with first-order matrix autoregressive models (MAR(1)) where the transition matrix has a Kronecker-product structure: $\Phi = B \otimes A$. The package offers projection-based initialization, Alternating Least Squares (ALS) estimation, and a matrix-normal MLE routine that explicitly estimates row/column covariances.

---

## Features

* `MAR` model type for storing data, initial guesses and estimation results.
* `fit!` with three methods:

  * `:proj` — nearest Kronecker-product projection of an OLS estimate
  * `:ls` — Alternating Least Squares for least-squares estimation
  * `:mle` — iterative EM-like MLE for A, B, Σ₁, Σ₂
* Utilities for simulation (`simulate_mar`), coefficient generation (`generate_mar_coefs`), vectorize/matricize helpers, and stability checks.
* Functions to compute objective functions and to build VAR companion matrices.

---

## Installation

This README assumes you will ship the code as a standard Julia package. From the Julia REPL:

```julia
import Pkg
Pkg.develop(path="/path/to/MAR.jl") # or Pkg.add("MAR") when registered
```

Add dependencies (if not already present in the package manifest): `Distributions`, `LinearAlgebra`, `StatsBase`, and any matrix-normal helper (e.g. `MatrixDistributions`).

---

## Quick start

Create a `MAR` object from a 3D array `Y` with dimensions `(n1, n2, T)` where `T` is time (observations):

```julia
using MAR

# Y: n1 × n2 × T array
model = MAR(Y; p=1, method=:ls)
fit!(model)
println(model)
```

### Methods

* `method=:proj` — fast, closed-form projection of the OLS vectorized estimator onto a Kronecker product.
* `method=:ls` — ALS iterative solver (returns normalized estimates with `A` having Frobenius norm 1).
* `method=:mle` — estimates `A`, `B`, and covariance factors `Sigma1`, `Sigma2` via iterative updates.

The `fit!` function mutates the `MAR` object and stores estimated `A`, `B`, and, for `:mle`, `Sigma1` and `Sigma2`. For `:ls` and `:mle`, the number of iterations used is stored in `model.iters`.

---

## Example: simulate + estimate

```julia
# simulate
res = simulate_mar(200; n1=3, n2=4)
Y = res.Y

# build model and estimate
model = MAR(Y; method=:proj)
fit!(model)

# use the projection estimates as initialization for ALS
model_ls = MAR(Y; method=:ls)
fit!(model_ls)

# MLE (estimates Sigma1, Sigma2 too)
model_mle = MAR(Y; method=:mle, maxiter=500, tol=1e-8)
fit!(model_mle)
```

---

## Notes & Design decisions

* The package normalizes `A` to have Frobenius norm 1 and rescales `B` accordingly. This avoids scale indeterminacy in the Kronecker decomposition.
* Many routines assume `p = 1`. Extending to `p > 1` requires changes to how predictors (`pred`) and responses (`resp`) are built and how the companion matrix is constructed.
* `fit!` uses `projection` to initialize iterative methods when no `A`/`B` are provided.

---

## Tests & examples

Add unit tests for:

* `projection` reproducing `A, B` when `phi = kron(B, A)`.
* `als` convergence on simulated data.
* `mle` recovering covariance matrices on simulated data.

Provide short notebooks or REPL examples that walk through simulating, estimating (`:proj`, `:ls`, `:mle`), and plotting eigenvalue decay for the Kronecker coefficient.

---

## Contributing

Contributions welcome! Please open issues for bug reports and feature requests. For pull requests, follow these steps:

1. Fork the repo and create a feature branch.
2. Add tests for new functionality.
3. Ensure `Pkg.test("MAR")` passes locally.
4. Open a PR with a clear description of changes.

---

