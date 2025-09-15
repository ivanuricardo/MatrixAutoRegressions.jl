@test "simulate mar" begin
    obs = 100
    A = randn(4,4)
end


@testset "generate_mar_coefs" begin
    using Random
    Random.seed!(20250915)
    n1, n2 = 3, 4
    result = generate_mar_coefs(n1, n2; maxiters=100)

    # Test types
    @test isa(result.A, Matrix{Float64})
    @test isa(result.B, Matrix{Float64})
    @test isa(result.phi, Matrix{Float64})
    @test isa(result.eig_phi, Vector{Float64})

    # Test sizes
    @test size(result.A) == (n1, n1)
    @test size(result.B) == (n2, n2)
    @test size(result.phi) == (n1*n2, n1*n2)
    @test length(result.eig_phi) == n1*n2

    # Test stability
    @test isstable(result.A, result.B)

end
