
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://IvanRicardo.github.io/MatrixAutoRegressions.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://IvanRicardo.github.io/MatrixAutoRegressions.jl/dev/)
[![Coverage](https://codecov.io/gh/IvanRicardo/MatrixAutoRegressions.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/IvanRicardo/MatrixAutoRegressions.jl)

# MatrixAutoRegressions.jl

A Julia package for estimation, inference, and impulse response analysis of Matrix Autoregressive (MAR) models, as introduced by [Chen, Xiao, and Yang (2021)](https://doi.org/10.1016/j.jeconom.2020.07.015).

## Overview

Matrix-valued time series arise when each observation is a matrix rather than a vector — for example, a panel of macroeconomic indicators across countries observed over time. The MAR model exploits this matrix structure through a bilinear formulation:

$$\mathbf{Y}_t = \sum_{j=1}^p \mathbf{A}_j \mathbf{Y}_{t-j} \mathbf{B}_j^\top + \mathbf{E}_t$$

which is equivalent to a restricted VAR with Kronecker-structured coefficient matrices $\mathbf{C}_j = \mathbf{B}_j \otimes \mathbf{A}_j$. This achieves substantial parameter reduction — $(N_1^2 + N_2^2)p$ parameters instead of $(N_1 N_2)^2 p$ for an unrestricted VAR — while maintaining interpretability along both the row and column dimensions.

This package provides:

- **MAR and VAR model estimation** via projection, alternating least squares (ALS), and maximum likelihood (MLE)
- **Impulse response functions** (reduced-form and Cholesky-identified)
- **Asymptotic inference** for IRFs via the delta method, accounting for Kronecker structure and normalization constraints
- **Bootstrap-after-bootstrap inference** extending [Kilian (1998)](https://doi.org/10.1162/003465398557465) to the MAR setting, with nearest-Kronecker-product projection
- **Bias correction** using both analytical (Pope–Kilian) and bootstrap approaches
- **Model selection** via AIC, BIC, and HQC
- **Simulation** utilities for MAR, VAR, and misspecified (two-term) MAR processes

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/ivanuricardo/MatrixAutoRegressions.jl")
```

## Quick Start

```julia
using MatrixAutoRegressions

# Simulate a MAR(1) process: 3×4 matrix observations, 200 time points
sim = simulate_mar(200; n1=3, n2=4, p=1)

# Fit a MAR model via MLE
model = MAR(sim.Y; p=1, method=:mle)
fit!(model)

# Compute impulse responses with delta-method standard errors
results = irf(model; hmax=20, shock_idx=[1, 1])
results.irfs    # point IRFs
results.irf_se  # standard errors

# Bootstrap confidence intervals (Kilian-style, with Kronecker projection)
boot = irf_bootstrap(model, Bootstrap(bias_runs=500);
                     boot_runs=2000, hmax=20, shock_idx=[1, 1])
boot.ci_lower   # lower confidence band
boot.ci_upper   # upper confidence band
```

## Models

### MAR — Matrix Autoregressive Model

```julia
# Create and fit
model = MAR(data; p=1, method=:mle)  # data is an n1 × n2 × T array
fit!(model)

# Available estimation methods
# :proj  — Nearest Kronecker product projection of OLS estimates
# :als   — Alternating least squares
# :mle   — Maximum likelihood with Kronecker-structured covariance
```

### VAR — Vector Autoregressive Model

```julia
# Create and fit
model = VAR(data; p=1)  # data is an n × T matrix
fit!(model)
```

## Impulse Response Analysis

### Point Estimates and Asymptotic Inference

```julia
# MAR: shock_idx is [row, column] of the matrix entry receiving the shock
results = irf(model; hmax=20, shock_idx=[1, 1])

# VAR: shock_idx is a scalar index
results = irf(var_model; hmax=20, shock_idx=1)

# Cholesky-identified structural IRFs
results = irf(model; hmax=20, shock_idx=[1, 1], ident=:cholesky)
```

### Bootstrap Inference

The package implements the bootstrap-after-bootstrap procedure with optional Kronecker projection (`project=true`), which ensures that bias-corrected coefficients retain the MAR structure:

```julia
# Analytical bias correction (Pope–Kilian closed-form, VAR only)
boot = irf_bootstrap(var_model, Analytical(); boot_runs=2000, hmax=20, shock_idx=1)

# Bootstrap bias correction
boot = irf_bootstrap(model, Bootstrap(bias_runs=500);
                     boot_runs=2000, hmax=20, shock_idx=[1, 1],
                     project=true)  # project onto Kronecker space
```

## Simulation

```julia
# MAR process with specified signal-to-noise ratio
sim = simulate_mar(200; n1=3, n2=4, p=1, snr=1.0)

# VAR process
sim = simulate_var(200; n=12, p=1)

# Two-term MAR (for misspecification studies):
# C = persistence * (B₁ ⊗ A₁) + persistence * η * (B₂ ⊗ A₂)
sim = simulate_two_term_mar(200; n1=3, n2=4, eta=0.1, persistence=0.8)
```

## Model Diagnostics

```julia
# Information criteria
aic(model)
bic(model)
hqc(model)

# Automatic lag selection
best_model, ic_table = fit_and_select!(model; ic_type=:bic)

# Log-likelihood (MLE only)
loglikelihood(model)

# Specification test (Kronecker structure)
p_value = specification_test(data)

# Coefficient standard errors
se = std_errors(model)
```

## Bias Correction

```julia
# Analytical (Pope–Kilian, VAR only)
corrected = bias_correction(var_model, Analytical())

# Bootstrap (works for both MAR and VAR)
corrected = bias_correction(model, Bootstrap(bias_runs=500))

# In-place
bias_correction!(model, Bootstrap(bias_runs=500))
```

## Key Utilities

| Function | Description |
|---|---|
| `make_companion(C)` | Build the VAR companion matrix |
| `isstable(A, B)` | Check stability of a MAR process |
| `mar_eigvals(A, B)` | Eigenvalues of the MAR companion matrix |
| `projection(Φ, dims)` | Nearest Kronecker product projection |
| `normalize_slices(A, B)` | Apply $\|\mathbf{A}_j\|_F = 1$ normalization |
| `vectorize(data)` | Reshape 3D array to 2D (vec each time slice) |
| `matricize(data, n1, n2)` | Reshape 2D back to 3D |
| `commutation_matrix(m, n)` | Commutation matrix $K_{m,n}$ |

## References

- Chen, R., Xiao, H., & Yang, D. (2021). Autoregressive models for matrix-valued time series. *Journal of Econometrics*, 222(1), 539–560.
- Kilian, L. (1998). Small-sample confidence intervals for impulse response functions. *Review of Economics and Statistics*, 80(2), 218–230.
