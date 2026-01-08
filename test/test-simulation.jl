
@testset "generate_mar_coefs" begin
    n1, n2, p = 3, 4, 1
    result = generate_mar_coefs(n1, n2; p)

    @test isa(result.A, Vector{Matrix{Float64}})
    @test isa(result.B, Vector{Matrix{Float64}})
    @test isa(result.sorted_eigs, Vector{Float64})

    # --- Test sizes ---
    @test length(result.A) == p
    @test length(result.B) == p
    @test size(result.A[1]) == (n1, n1)
    @test size(result.B[1]) == (n2, n2)
    @test length(result.sorted_eigs) == n1 * n2 * p

    # --- Test stability ---
    @test isstable(result.A, result.B)

    n1, n2, p = 4, 5, 2
    result = generate_mar_coefs(n1, n2; p)

    # --- Test types ---
    @test isa(result.A, Vector{Matrix{Float64}})
    @test isa(result.B, Vector{Matrix{Float64}})
    @test isa(result.sorted_eigs, Vector{Float64})

    # --- Test sizes ---
    @test length(result.A) == p
    @test length(result.B) == p
    @test size(result.A[1]) == (n1, n1)
    @test size(result.B[1]) == (n2, n2)
    @test length(result.sorted_eigs) == n1 * n2 * p

    # --- Test stability ---
    @test isstable(result.A, result.B)

end

@testset "simulate mar" begin
    obs = 10
    n1, n2, p = 3, 4, 1

    # Default behavior (random A, B, identity Sigma)
    result = simulate_mar(obs)
    @test size(result.Y) == (n1, n2, obs)
    @test size(result.A[1]) == (n1, n1)
    @test size(result.B[1]) == (n2, n2)

    # With user-specified A, B
    A = [0.5 * I(n1)]
    B = [0.5 * I(n2)]
    result2 = simulate_mar(obs; A=A, B=B)
    @test all(result2.A .== A)
    @test all(result2.B .== B)
    @test size(result2.Y) == (n1, n2, obs)

    # Check that consecutive Y values are different (stochastic)
    @test any(result2.Y[:, :, 2] .!= result2.Y[:, :, 3])

    # Longer lag length
    obs = 50
    n1, n2, p = 3, 4, 5
    result = simulate_mar(obs; n1, n2, p)
    @test size(result.Y) == (n1, n2, obs)
    @test length(result.A) == p
    @test length(result.B) == p

    # With user-specified A, B
    n1, n2, p = 3, 4, 2
    coefs = generate_mar_coefs(n1, n2; p)

    result2 = simulate_mar(obs; A=coefs.A, B=coefs.B)
    @test all(result2.A .== coefs.A)
    @test all(result2.B .== coefs.B)
    @test size(result2.Y) == (n1, n2, obs)

end

@testset "generate_var_coefs: return structure" begin
    n, p = 4, 2
    out = generate_var_coefs(n, p)

    @test haskey(out, :C)
    @test haskey(out, :sorted_eigs)

    C = out.C
    @test length(C) == p
    @test all(size(C[j]) == (n, n) for j in 1:p)
end

@testset "generate_var_coefs: stability" begin
    n, p = 5, 3
    out = generate_var_coefs(n, p)

    @test isstable(out.C; mineigen=0.0, maxeigen=0.90)

    companion = make_companion(out.C)
    ρ = maximum(abs.(eigvals(companion)))
    @test ρ < 0.90
end

@testset "generate_var_coefs: eigenvalue consistency" begin
    n, p = 3, 2
    out = generate_var_coefs(n, p)

    eigs_direct = var_eigvals(out.C)

    @test length(out.sorted_eigs) == length(eigs_direct)
    @test sort(abs.(out.sorted_eigs)) ≈ sort(abs.(eigs_direct))
end

@testset "generate_var_coefs: deterministic with seed" begin
    n, p = 4, 2

    Random.seed!(123)
    out1 = generate_var_coefs(n, p)

    Random.seed!(123)
    out2 = generate_var_coefs(n, p)

    @test all(out1.C[j] ≈ out2.C[j] for j in 1:p)
    @test out1.sorted_eigs ≈ out2.sorted_eigs
end

@testset "generate_var_coefs: p = 1 edge case" begin
    n, p = 6, 1
    out = generate_var_coefs(n, p)

    @test length(out.C) == 1
    @test size(out.C[1]) == (n, n)

    eigs = eigvals(out.C[1])
    @test maximum(abs.(eigs)) < 0.90
end

@testset "generate_var_coefs: failure throws error" begin
    n, p = 20, 5

    @test_throws ErrorException generate_var_coefs(n, p; maxiter=0)
end

@testset "generate_var_coefs: bounded operator norm" begin
    n, p = 4, 2
    out = generate_var_coefs(n, p)

    # crude but effective bound
    opnorm_sum = sum(opnorm, out.C)
    @test opnorm_sum < p * 1.5
end

@testset "MAR eigvals max iters" begin
    n1, n2 = 3, 4

    @test_warn  "Reached the maximum number of iterations! May not be stable." begin
        out = generate_mar_coefs(n1, n2; maxiter=2)
    end
end
