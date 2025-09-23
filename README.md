
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://IvanRicardo.github.io/MatrixAutoRegressions.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://IvanRicardo.github.io/MatrixAutoRegressions.jl/dev/)
[![Build Status](https://github.com/IvanRicardo/MatrixAutoRegressions.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/IvanRicardo/MatrixAutoRegressions.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/IvanRicardo/MatrixAutoRegressions.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/IvanRicardo/MatrixAutoRegressions.jl)

# MatrixAutoRegressions.jl

A lightweight Julia package for Matrix Autoregressive (MAR) models. This README introduces the main types and functions, shows quick examples for simulation and estimation (OLS projection, Alternating Least Squares — ALS, and Maximum Likelihood — MLE), and provides notes on stability and common utilities.

---

## Highlights

* `MAR` type: store MAR model configuration, parameters and data.
* Estimation methods: `:proj` (nearest-Kronecker projection of OLS VAR), `:ls` (alternating least squares), `:mle` (iterative MLE alternating A/B and covariances).
* Simulation: `simulate_mar` to generate synthetic MAR data.
* Helpers: `projection`, `als`, `mle`, `generate_mar_coefs`, `isstable`, `vectorize`/`matricize`, `residuals`, and more.

---

## Installation

This README assumes the package is available as `MatrixAutoRegressions` (replace with your actual package name). In the REPL:

```julia
] dev /path/to/repo
using MatrixAutoRegressions
```

---

## Quick examples

All examples below are self-contained and runnable in the REPL or a script.

### 1) Simulate a MAR(1) and fit with ALS (least squares)

```julia
using MatrixAutoRegressions, LinearAlgebra

# simulate a MAR(1) with n1=3, n2=4 and 300 observations
res = simulate_mar(300; n1=3, n2=4, p=1)
Y = res.Y         # data array: (n1, n2, obs)
A_true = res.A    # true A slice(s)
B_true = res.B    # true B slice(s)

# build MAR model (method = :ls for alternating least squares)
model = MAR(Y; p=1, method=:ls)
fit!(model)

println(model)            # brief summary printed by Base.show
resid = residuals(model)  # residuals array

# Compare to truth (example for p=1)
println("||A_est - A_true|| = ", norm(model.A[1] - A_true[1]))
println("||B_est - B_true|| = ", norm(model.B[1] - B_true[1]))
```

### 2) Fast projection (nearest Kronecker product) from OLS VAR

```julia
using MatrixAutoRegressions
res = simulate_mar(300; n1=3, n2=4, p=1)
Y = res.Y
A_true = res.A
B_true = res.B

# Use projection of OLS VAR directly
model_proj = MAR(Y; p=1, method=:proj)
fit!(model_proj)

# model_proj.A and model_proj.B are the projection estimators
println("Projected A size: ", size(model_proj.A[1]))
```

### 3) Maximum likelihood estimation (iterative)

```julia
using MatrixAutoRegressions
res = simulate_mar(300; n1=3, n2=4, p=1)
Y = res.Y
A_true = res.A
B_true = res.B

# Start from projection initialization; set method=:mle to run the MLE routine
model_mle = MAR(Y; p=1, method=:mle, maxiter=50, tol=1e-8)
fit!(model_mle)

# MLE returns fitted A/B and Sigma1/Sigma2
println("iters used: ", model_mle.iters)
println("Sigma1 size: ", size(model_mle.Sigma1))
```

### 4) Generate stable MAR coefficients

```julia
using MatrixAutoRegressions

coefs = generate_mar_coefs(3, 4; p=1)
A_gen, B_gen = coefs.A, coefs.B
println("stability eigenvalues: ", coefs.sorted_eigs)
println("isstable? ", isstable(A_gen, B_gen))
```

### 5) Use ALS directly if you already have starting slices

```julia
using MatrixAutoRegressions

coefs = generate_mar_coefs(3, 4; p=1)
println("stability eigenvalues: ", coefs.sorted_eigs)
println("isstable? ", isstable(coefs.A, coefs.B))

dgp = simulate_mar(300; n1=3, n2=4, p=1, A=coefs.A, B=coefs.B)

# Suppose A0 and B0 are initial guesses (vectors of matrices)
results = als(dgp.Y, coefs.A, coefs.B; maxiter=300, tol=1e-7)
A_est, B_est = results.A, results.B
```

### 6) Forecasting with a fitted MAR model

You can generate multi-step-ahead forecasts after fitting a MAR model.
Use `train_test_split` to hold out the last `h` observations for evaluation, then call `predict(model; h)`.

```julia
using MatrixAutoRegressions, LinearAlgebra

# simulate a MAR(2) with n1=3, n2=4 and 200 observations
sim = simulate_mar(200; n1=3, n2=4, p=2, snr=1000)
model = MAR(sim.Y, p=2)
fit!(model)

# split into training and test sets (last 5 observations are test)
train_data, test_data = train_test_split(model; h=5)
model.data = train_data

# forecast the next 5 steps
Yhat = predict(model; h=5)

println("Forecast array size: ", size(Yhat))   # (3, 4, 5)
println("Forecast error norm: ", norm(Yhat - test_data))
```
`Yhat` has shape `(n1, n2, h)` and contains the forecasted matrices.
You can compare directly to the held-out test data, or use it for downstream analysis.

---

## Practical notes & tips

* **Normalization**: Internally A slices are normalized (Frobenius norm) and B adjusted accordingly to reduce scale identifiability.
* **Initialization**: Good initialization speeds convergence. The package uses OLS + NKP projection as defaults.
* **Stability**: Generated coefficient sets aim to be stable; always check `isstable(A, B)` before trusting long-run simulations.
* **Convergence**: If ALS/MLE reaches `maxiter`, an `@warn` is emitted and the current estimates are returned. Tweak `maxiter` and `tol` if needed.

---

## Contributing

PRs are welcome. Please open an issue if you spot numerical instability or surprising behavior.

---

