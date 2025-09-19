
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

    A = randn(3, 3, 2)
    B = randn(4, 4, 2)
    A_new, B_new = normalize_slices(A, B)
    @test isapprox(norm(A_new[:, :, 1]), 1)
    @test isapprox(norm(A_new[:, :, 2]), 1)
    @test isapprox(kron(B[:, :, 1], A[:, :, 1]), kron(B_new[:, :, 1], A_new[:, :, 1]))
    @test isapprox(kron(B[:, :, 2], A[:, :, 2]), kron(B_new[:, :, 2], A_new[:, :, 2]))

end
