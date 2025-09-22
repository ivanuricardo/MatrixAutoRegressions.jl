
@testset "Nearest Kronecker Product" begin
    n1 = 4
    n2 = 3
    A1 = randn(n1,n1)
    B1 = randn(n2,n2)

    phi = [kron(B1, A1)]

    est = projection(phi, (n1, n2))
    true_a1 = A1 / norm(A1)
    @test isapprox(true_a1, est.A[1]; atol=1e-8) || isapprox(true_a1, -est.A[1]; atol=1e-8)

    true_b1 = B1 * norm(A1)
    @test isapprox(true_b1, est.B[1]; atol=1e-8) || isapprox(true_b1, -est.B[1]; atol=1e-8)

    true_product = kron(B1, A1)
    est_product = kron(est.B[1], est.A[1])
    @test isapprox(est_product, true_product, atol=1e-08)

    # Adding another coef as a lag
    A2 = randn(n1, n1)
    B2 = randn(n2, n2)

    phi2 = kron(B2, A2)
    push!(phi, phi2)

    est2 = projection(phi, (n1, n2))
    true_a2 = A2 / norm(A2)
    @test isapprox(true_a1, est2.A[1]; atol=1e-8) || isapprox(true_a1, -est2.A[1]; atol=1e-8)
    @test isapprox(true_a2, est2.A[2]; atol=1e-8) || isapprox(true_a2, -est2.A[2]; atol=1e-8)

    true_b2 = B2 * norm(A2)
    @test isapprox(true_b1, est2.B[1]; atol=1e-8) || isapprox(true_b1, -est2.B[1]; atol=1e-8)
    @test isapprox(true_b2, est2.B[2]; atol=1e-8) || isapprox(true_b2, -est2.B[2]; atol=1e-8)

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

