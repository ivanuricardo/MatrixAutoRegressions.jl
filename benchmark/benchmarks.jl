using MatrixAutoRegressions
using BenchmarkTools

SUITE = BenchmarkGroup()
SUITE["rand"] = @benchmarkable rand(10)

# Write your benchmarks here.

using LinearAlgebra
using BenchmarkTools

# Dimensions
m, n, T = 50, 50, 500
A = randn(m, m)
B = randn(n, n)
resp = randn(m, n, T)
pred = randn(m, n, T)

# Safe version (bounds checks on)
function residuals_safe(resp, pred, A, B)
    m, n, obs = size(resp)
    res = similar(resp)
    for t in 1:obs
        res[:, :, t] = resp[:, :, t] - A * pred[:, :, t] * B'
    end
    return res
end

# Fast version (bounds checks off with @inbounds)
function residuals_fast(resp, pred, A, B)
    m, n, obs = size(resp)
    res = similar(resp)
    @inbounds for t in 1:obs
        res[:, :, t] = resp[:, :, t] - A * pred[:, :, t] * B'
    end
    return res
end

println("Check correctness: ",
    maximum(abs.(residuals_safe(resp, pred, A, B) .-
                 residuals_fast(resp, pred, A, B))))

println("\nBenchmark safe (with bounds checks):")
@benchmark residuals_safe($resp, $pred, $A, $B)

println("\nBenchmark fast (with @inbounds):")
@benchmark residuals_fast($resp, $pred, $A, $B)
