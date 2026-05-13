using MatrixAutoRegressions
using BenchmarkTools, Random, Statistics, LinearAlgebra, RCall
Random.seed!(20250923)

@benchmark begin
    dgp = simulate_mar(1000; n1=3, n2=4, p=1)
    model = MAR(dgp.Y; p=1, method=:als)
    fit!(model)
end

R"""
set.seed(20250923)
library(tensorTS)
library(microbenchmark)

microbenchmark(
  dgp = tenAR.sim(1000, dim=c(3,4), R=1, P=1, rho=0.7, cov='iid'),
  est = {
    dgp <- tenAR.sim(1000, dim=c(3,4), R=1, P=1, rho=0.7, cov='iid')
    tenAR.est(dgp, R=1, P=1, method="LSE")
  },
  times = 300L
)
"""

# Median R code time:     126.2 ms
# Median Julia code time: 37.6 ms
# Mean R code time:       136.8 ms
# Mean Julia code time:   37.4 ms

