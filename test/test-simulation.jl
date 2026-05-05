
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

@testset "simulate_two_term_mar" begin

    # Basic output structure
    res = simulate_two_term_mar(100; n1=3, n2=2, eta=0.0)
    @test size(res.Y) == (3, 2, 100)
    @test size(res.A1) == (3, 3)
    @test size(res.B1) == (2, 2)
    @test size(res.A2) == (3, 3)
    @test size(res.B2) == (2, 2)
    @test size(res.C) == (6, 6)
    @test res.eta == 0.0

    # eta = 0 means C = 0.5 * kron(B1, A1)
    res0 = simulate_two_term_mar(100; n1=3, n2=2, eta=0.0)
    @test res0.C ≈ 0.8 * kron(res0.B1, res0.A1)

    # eta > 0 adds second term
    res1 = simulate_two_term_mar(100; n1=3, n2=2, eta=0.3)
    expected_C = 0.8 * kron(res1.B1, res1.A1) + 0.8 * 0.3 * kron(res1.B2, res1.A2)
    @test res1.C ≈ expected_C

    # Spectral radius of coefficient matrices should be 1
    @test maximum(abs.(eigvals(res1.A1))) ≈ 1.0
    @test maximum(abs.(eigvals(res1.B1))) ≈ 1.0
    @test maximum(abs.(eigvals(res1.A2))) ≈ 1.0
    @test maximum(abs.(eigvals(res1.B2))) ≈ 1.0

    # Stochastic: consecutive observations differ
    @test any(res1.Y[:, :, 2] .!= res1.Y[:, :, 3])

    # User-supplied coefficients
    A1 = [1.0 0.0; 0.0 0.5; 0.0 0.0]  # not square, should we test square?
    A1 = [0.5 0.0 0.0; 0.0 0.3 0.0; 0.0 0.0 0.2]
    B1 = [0.4 0.0; 0.0 0.6]
    A2 = [0.1 0.0 0.0; 0.0 0.2 0.0; 0.0 0.0 0.3]
    B2 = [0.3 0.0; 0.0 0.1]
    res_custom = simulate_two_term_mar(200; n1=3, n2=2, eta=0.5,
                                        A1=A1, B1=B1, A2=A2, B2=B2)
    @test res_custom.A1 === A1
    @test res_custom.B1 === B1
    @test res_custom.A2 === A2
    @test res_custom.B2 === B2
    @test size(res_custom.Y) == (3, 2, 200)

    # Warn if not stationary
    A1_big = 5.0 * I(3)
    B1_big = 5.0 * I(2)
    @test_warn "not stationary" simulate_two_term_mar(50; n1=3, n2=2, eta=0.0,
                                                       A1=A1_big, B1=B1_big)

end
