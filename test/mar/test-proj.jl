
@testset "Nearest Kronecker Product" begin
    n1 = 4
    n2 = 3
    A1 = randn(n1,n1)
    B1 = randn(n2,n2)

    phi = kron(B1, A1)

    est = projection(phi, (n1, n2))
    scale_term = norm(A1)
    true_a1 = A1 / scale_term
    true_b1 = B1 * scale_term
    if true_a1[1] < 0
        true_a1 .= - true_a1
        true_b1 .= - true_b1
    end
    @test isapprox(true_a1, est.A; atol=1e-8)
    @test isapprox(true_b1, est.B; atol=1e-8)

    true_product = kron(B1, A1)
    est_product = kron(est.B, est.A)
    @test isapprox(est_product, true_product, atol=1e-08)

    # Adding another coef as a lag
    n1 = 3
    n2 = 4
    p = 3
    dims = (n1, n2)

    Astack = [randn(n1,n1), randn(n1, n1), randn(n1, n1)]
    Bstack = [randn(n2,n2), randn(n2, n2), randn(n2, n2)]
    phi = [kron(Bstack[1], Astack[1]), kron(Bstack[2], Astack[2]), kron(Bstack[3], Astack[3])]

    phi_est = projection(phi, dims)

    scale_term = norm(Astack[1])
    norm_Astack = Astack[1] ./ scale_term
    norm_Bstack = Bstack[1] .* scale_term
    if norm_Astack[1] < 0
        norm_Astack .= - norm_Astack
        norm_Bstack .= - norm_Bstack
    end

    @test isapprox(phi_est.A[1], norm_Astack, atol=sqrt(eps()))
    @test isapprox(phi_est.B[1], norm_Bstack, atol=sqrt(eps()))

    scale_term = norm(Astack[2])
    norm_Astack = Astack[2] ./ scale_term
    norm_Bstack = Bstack[2] .* scale_term
    if norm_Astack[1] < 0
        norm_Astack .= - norm_Astack
        norm_Bstack .= - norm_Bstack
    end

    @test isapprox(phi_est.A[2], norm_Astack, atol=sqrt(eps()))
    @test isapprox(phi_est.B[2], norm_Bstack, atol=sqrt(eps()))

end

@testset "proj algorithm behavior" begin
    obs = 10000
    dgp = simulate_mar(obs)
    matdata = dgp.Y
    A_init = dgp.A
    B_init = dgp.B

    model = MAR(matdata; method = :proj)
    fit!(model)

    @test norm(model.A[1] - A_init[1]) < 0.1
    @test norm(model.B[1] - B_init[1]) < 0.1

    dgp2 = simulate_mar(obs; p=2)
    matdata2 = dgp2.Y
    A2 = dgp2.A
    B2 = dgp2.B

    model2 = MAR(matdata2; method = :proj, p = 2)
    fit!(model2)

    kron1 = kron(B2[1], A2[1])
    est_kron1 = kron(model2.B[1], model2.A[1])
    @test norm(kron1 - est_kron1) < 0.1

    kron2 = kron(B2[2], A2[2])
    est_kron2 = kron(model2.B[2], model2.A[2])
    @test norm(kron2 - est_kron2) < 0.1

end

@testset "Permutation matrix" begin
    n1 = 3
    n2 = 4
    phi = randn(12, 12)
    dims = (n1,n2)

    perm_mat = permutation_matrix(dims)
    tensor_phi = reshape(phi, (n1, n2, n1, n2))
    R = reshape(permutedims(tensor_phi, (1, 3, 2, 4)), n1 * n1, n2 * n2)

    @test norm(vec(R) - perm_mat * vec(phi)) == 0

    p = 2
    phi = randn(n1 * n2, n1 * n2 * p)
    dims = (n1, n2)

    perm_mat = permutation_matrix(dims)
    new_perm = kron(I(p), perm_mat)
    tensor_phi = reshape(phi, (n1, n2, n1, n2, p))
    R = reshape(permutedims(tensor_phi, (1, 3, 2, 4, 5)), n1 * n1, n2 * n2 * p)

    @test norm(vec(R) - new_perm * vec(phi)) == 0
end
