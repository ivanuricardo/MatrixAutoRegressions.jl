
@testset "vectorize and matricize data" begin
    n1 = 3
    n2 = 4
    obs = 10
    data = randn(n1, n2, obs)
    first_obs = data[:, :, 1]
    second_obs = data[:, :, 2]
    third_obs = data[:, :, 3]

    # In place vectorization
    vec_data = vectorize(data)

    @test vec(first_obs) == vec_data[:, 1]
    @test vec(second_obs) == vec_data[:, 2]
    @test vec(third_obs) == vec_data[:, 3]

    mat_data = matricize(vec_data, n1, n2)
    @test data == mat_data

end

@testset "stability of the MAR(p)" begin
    n1 = 3
    n2 = 4
    A1 = randn(n1, n1, 1)
    B1 = randn(n2, n2, 1)
    @test isstable(A1, B1) isa Bool

    A2 = randn(n1, n1, 2)
    B2 = randn(n2, n2, 2)
    @test isstable(A2, B2) isa Bool

end

@testset "normalizing slices" begin

    A = randn(3,3,2)
    normalize_slices!(A)
    @test isapprox(norm(A[:, :, 1]), 1)
    @test isapprox(norm(A[:, :, 2]), 1)

    B = randn(5,5,3)
    normalize_slices!(B)
    @test isapprox(norm(B[:, :, 1]), 1)
    @test isapprox(norm(B[:, :, 2]), 1)

end
