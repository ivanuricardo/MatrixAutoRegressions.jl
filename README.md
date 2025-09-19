
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

## API reference (selected)

### `mutable struct MAR`

Fields:

* `A, B` — estimated coefficient matrices or `nothing` before fit.
* `p` — lag order (currently code assumes `p=1` in many places).
* `Sigma1, Sigma2` — row/column covariance factors for the matrix-normal error.
* `dims` — `(n1, n2)`
* `obs` — number of time observations
* `method` — `:proj`, `:ls`, or `:mle`
* `resp`, `pred` — internal demeaned response and predictor arrays
* `maxiter`, `tol` — control iterative solvers
* `iters` — number of iterations used (set after fitting)

### Constructors

* `MAR(data::AbstractArray; p::Int=1, method::Symbol=:ls, A=nothing, B=nothing, maxiter=100, tol=1e-6)`

  * `data` should be `n1 × n2 × T`. The constructor demeans across time and builds `resp`/`pred` (y₂..T, y₁..T-1).

### `fit!(model::MAR)`

Estimate parameters according to `model.method`. Returns the mutated `model`.

### `projection(phi, dims)`

Compute the nearest Kronecker-product projection of a vectorized coefficient matrix `phi` (or covariance) onto `B ⊗ A`. Returns a named tuple `(A, B, phi_est)`.

### `als(A_init, B_init, resp, pred; maxiter=100, tol=1e-6)`

Alternating Least Squares that returns `(A, B, track_obj, obj, num_iter)`.

### `mle(A_init, B_init, Sigma1_init, Sigma2_init, resp, pred; maxiter=100, tol=1e-6)`

Iterative MLE routine returning `(A, B, Sigma1, Sigma2, track_obj, obj, num_iter)`.

### `simulate_mar(obs; n1=3, n2=4, A=nothing, B=nothing, Sigma1=nothing, Sigma2=nothing, burnin=50, snr=0.7)`

Simulate `obs` observations from a MAR(1). If `A`/`B` not given, random stable coefficients are generated. Returns a named tuple with `Y, A, B, Sigma1, Sigma2, sorted_eigs`.

### Helpers

* `generate_mar_coefs` — generate normalized stable `A` and `B`.
* `vectorize`, `matricize` — reshape helpers.
* `makecompanion`, `isstable` — VAR companion and stability checks.
* `ls_objective`, `mle_objective` — objective functions for diagnostics.

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

## License

Choose and add a license (MIT/Apache-2.0 recommended).

---

If you'd like, I can also:

* add badges (CI, codecov),
* generate a `Project.toml` example,
* produce unit-test skeletons, or
* create usage notebooks with plots and comparisons between `:proj`, `:ls`, and `:mle` estimation.
