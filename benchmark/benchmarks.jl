using MatrixAutoRegressions
using BenchmarkTools, Random, Statistics, LinearAlgebra, RCall
Random.seed!(20250923)

@benchmark begin
    dgp = simulate_mar(1000; n1=3, n2=4, p=1)
    model = MAR(dgp.Y; p=1, method=:ls)
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

# Median R code time:     75.5 ms
# Median Julia code time: 16.3 ms
# Mean R code time:       80.5 ms
# Mean Julia code time:   16.8 ms

