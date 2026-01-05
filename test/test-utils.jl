
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
    A1 = [randn(n1, n1)]
    B1 = [randn(n2, n2)]
    @test isstable(A1, B1) isa Bool

    A2 = randn(n1, n1)
    push!(A1, A2)
    B2 = randn(n2, n2)
    push!(B1, B2)
    @test isstable(A1, B1) isa Bool

end

@testset "normalizing slices" begin

    A = [randn(3, 3), randn(3,3)]
    B = [randn(4, 4), randn(4,4)]
    A_new, B_new = normalize_slices(A, B)
    @test isapprox(norm(A_new[1]), 1)
    @test isapprox(norm(A_new[2]), 1)
    @test A_new[1][1] > 0
    @test A_new[2][1] > 0
    @test isapprox(kron(B[1], A[1]), kron(B_new[1], A_new[1]))
    @test isapprox(kron(B[2], A[2]), kron(B_new[2], A_new[2]))

end

@testset " Commuting B' ⊗ A into B ⊗ A with a permutation matrix" begin
    A = randn(3,3)
    K = commutation_matrix(A)
    @test vec(A) == K * vec(A')
    @test K * vec(A) == vec(A')

    A = randn(3,3)
    B = randn(4,4)
    P = vectorize_kronecker(B, A)

    @test vec(kron(B, A)) == P * kron(vec(B), vec(A))
    @test P' * vec(kron(B, A)) == kron(vec(B), vec(A))

    obs = 100
    dgp = simulate_mar(obs; p=3)
    matdata = dgp.Y

    model = MAR(matdata; method = :proj, p=3)
    fit!(model)
    Astack, Bstack = stack_coefs(model)
    P = vectorize_kronecker(Bstack, Astack)
    @test vec(kron(Bstack, Astack)) == P * kron(vec(Bstack), vec(Astack))
    @test P' * vec(kron(Bstack, Astack)) == kron(vec(Bstack), vec(Astack))

end

@testset "Large commutation matrix" begin
    obs = 100
    n1 = 3
    p = 3
    dgp = simulate_mar(obs; p=3)
    matdata = dgp.Y

    model = MAR(matdata; method = :proj, p=3)
    fit!(model)
    Astack, Bstack = stack_coefs(model)

    transposed_B = vcat(vec(Astack), vec(Bstack'))
    large_comm = large_commutation_matrix(Bstack', n1, p)

    fixed_B = large_comm * transposed_B
    @test isapprox(norm(fixed_B - vcat(vec(Astack), vec(Bstack))), 0.0)
end

@testset "Q selection" begin
    obs = 100
    n1 = 3
    n2 = 4
    p = 3
    dgp = simulate_mar(obs; p=3)
    matdata = dgp.Y

    model = MAR(matdata; method = :mle, p=3)
    fit!(model)
    structured_data = structure_lagged_data(model)

    first_obs = structured_data[:, :, 1]
    vec_obs = vec(first_obs)
    correct_structure = vcat(vec(first_obs[1:3, 1:4]), vec(first_obs[4:6, 5:8]), vec(first_obs[7:end, 9:end]))
    Q = Q_matrix(n1, n2, p)

    @test isapprox(norm(correct_structure - Q * vec_obs), 0)

    A, B = stack_coefs(model)
    full_kron = kron(B, A)
    selected_kron = full_kron * Q'

    @test selected_kron[1:12, 1:12] == kron(model.B[1], model.A[1])
    @test selected_kron[1:12, 13:24] == kron(model.B[2], model.A[2])
    @test selected_kron[1:12, 25:end] == kron(model.B[3], model.A[3])

end

@testset "Index formula" begin
    n1, n2 = 10, 15
    idx = [3,2]

    my_mat = reshape(1:n1*n2, n1, n2)
    selected_val = my_mat[idx[1], idx[2]]
    formula_val = idx[1] + (idx[2] - 1) * n1
    @test selected_val == formula_val
end



