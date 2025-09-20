
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


