using MatrixAutoRegressions
using BenchmarkTools

SUITE = BenchmarkGroup()
SUITE["rand"] = @benchmarkable rand(10)

# Write your benchmarks here.

using LinearAlgebra

function make_array_form(N, p)
    randn(N, N, p)
end

function make_vector_form(N, p)
    [randn(N, N) for _ in 1:p]
end

# Access single lag
function access_array(A, j)
    return A[:, :, j]
end

function access_vector(V, j)
    return V[j]
end

# Loop through lags and multiply by vector
function apply_array(A, x)
    N, _, p = size(A)
    y = zeros(N)
    for j in 1:p
        y += A[:, :, j] * x
    end
    return y
end

function apply_vector(V, x)
    N = size(V[1], 1)
    y = zeros(N)
    for Aj in V
        y += Aj * x
    end
    return y
end

# Benchmark runner
function run_benchmarks(N, p)
    println("==== Benchmarks for N=$N, p=$p ====")
    A = make_array_form(N, p)
    V = make_vector_form(N, p)
    x = randn(N)

    println("\n--- Construction ---")
    @btime make_array_form($N, $p)
    @btime make_vector_form($N, $p)

    println("\n--- Access single lag ---")
    @btime access_array($A, 3)
    @btime access_vector($V, 3)

    println("\n--- Apply all lags (mat-vec multiply) ---")
    @btime apply_array($A, $x)
    @btime apply_vector($V, $x)
end

# Example runs
run_benchmarks(10, 5)
run_benchmarks(50, 5)
run_benchmarks(50, 20)
