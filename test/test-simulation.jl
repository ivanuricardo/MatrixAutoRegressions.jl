@testset "simulate mar" begin
    obs = 10
    n1, n2 = 3, 4

    # Test 1: Default behavior (random A, B, identity Sigma)
    result = simulate_mar(obs)
    @test size(result.Y) == (n1, n2, obs)
    @test size(result.A) == (n1, n1)
    @test size(result.B) == (n2, n2)

    # Test 2: With user-specified A, B
    A = 0.5 * I(n1)
    B = 0.5 * I(n2)
    result2 = simulate_mar(obs; A=A, B=B)
    @test all(result2.A .== A)
    @test all(result2.B .== B)
    @test size(result2.Y) == (n1, n2, obs)

    # Test 4: Check that consecutive Y values are different (stochastic)
    is_stochastic = any(result2.Y[:, :, 2] .!= result2.Y[:, :, 3])
    @test is_stochastic

end


@testset "generate_mar_coefs" begin
    n1, n2, p = 3, 4, 1
    result = generate_mar_coefs(n1, n2; p)

    # Test types
    @test isa(result.A, Array{Float64})
    @test isa(result.B, Array{Float64})
    @test isa(result.phi, Array{Float64})
    @test isa(result.eig_phi, Vector{Float64})

    # Test sizes
    @test size(result.A) == (n1, n1, p)
    @test size(result.B) == (n2, n2, p)
    @test size(result.phi) == (n1*n2*p, n1*n2*p)
    @test length(result.eig_phi) == n1*n2*p

    # Test stability
    @test isstable(result.A, result.B)

    n1, n2, p = 4, 5, 2
    result = generate_mar_coefs(n1, n2; p)

    # Test types
    @test isa(result.A, Array{Float64})
    @test isa(result.B, Array{Float64})
    @test isa(result.phi, Array{Float64})
    @test isa(result.eig_phi, Vector{Float64})

    # Test sizes
    @test size(result.A) == (n1, n1, p)
    @test size(result.B) == (n2, n2, p)
    @test size(result.phi) == (n1*n2*p, n1*n2*p)
    @test length(result.eig_phi) == n1*n2*p

    # Test stability
    @test isstable(result.A, result.B)

end
